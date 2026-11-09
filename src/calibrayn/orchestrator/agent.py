from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.models.documents import Chunk
from calibrayn.models.learning import Attempt, Question, Support
from calibrayn.orchestrator.cognitive_load import Answer, CognitiveLoadEstimator, LoadEstimate
from calibrayn.orchestrator.metacognition import (
    CalibrationReading,
    MetacognitionTracker,
    Rating,
)
from calibrayn.orchestrator.policy import Plan, make_plan
from calibrayn.orchestrator.selector import choose_chunk
from calibrayn.orchestrator.zpd import ZPDStatus, ZPDTracker
from calibrayn.scheduler.ease_adjuster import adjust_for_confidence
from calibrayn.scheduler.sm2 import Schedule, is_due, next_schedule

HISTORY_LIMIT = 300
EXTRA_PRACTICE = (
    "Nothing is due right now, so this is extra practice. A correct answer will not change "
    "the schedule."
)


class NothingToStudy(Exception):
    pass


class StaleQuestion(Exception):
    pass


@dataclass(frozen=True)
class AskedQuestion:
    question: Question
    chunk: Chunk
    plan: Plan
    asked_at: datetime


@dataclass(frozen=True)
class Outcome:
    score: float
    correct: bool
    feedback: str
    marked_by: str
    answer_key: str
    grade: int
    schedule: Schedule
    calibration: CalibrationReading
    load: LoadEstimate
    level: int
    queued_recall: bool
    counted: bool = True


@dataclass(frozen=True)
class Snapshot:
    level: int
    zpd: ZPDStatus
    load: LoadEstimate
    calibration: CalibrationReading
    due_now: int
    pending_recalls: int
    session_minutes: float


def grade_for(score, confidence, config):
    floors = ((0.9, 5), (0.75, 4), (config.correct_from, 3), (0.4, 2), (0.2, 1))
    grade = 0
    for floor, value in floors:
        if score >= floor:
            grade = value
            break
    if confidence is Confidence.GUESSING:
        grade = min(grade, config.guessing_grade_cap)
    return grade


class StudyAgent:
    def __init__(self, library, progress, writer, grader, tuning, clock=datetime.now):
        self.library = library
        self.progress = progress
        self.writer = writer
        self.grader = grader
        self.tuning = tuning
        self.clock = clock
        self.session_started = clock()
        self.reload()

    def reload(self):
        self.recent = []
        history = self.progress.attempts(limit=HISTORY_LIMIT)
        stored = self.progress.get_state("level")
        level = int(stored) if stored else self.tuning.zpd.start_level
        level = min(max(level, self.tuning.zpd.min_level), self.tuning.zpd.max_level)
        at_level = [attempt.score for attempt in history if attempt.level == level]
        self.zpd = ZPDTracker(
            self.tuning.zpd, level=level, scores=at_level[-self.tuning.zpd.window :]
        )
        self.load = CognitiveLoadEstimator(
            self.tuning.load,
            attempts=[Answer(a.score, max(a.response_seconds, 0.1)) for a in history][
                -self.tuning.load.baseline_window :
            ],
        )
        self.meta = MetacognitionTracker(
            self.tuning.meta,
            ratings=[Rating(a.confidence, a.correct) for a in history][-self.tuning.meta.window :],
        )

    def start_session(self, now=None):
        self.session_started = now or self.clock()

    def session_minutes(self, now=None):
        moment = now or self.clock()
        return max(0.0, (moment - self.session_started).total_seconds() / 60)

    def snapshot(self, now=None):
        moment = now or self.clock()
        schedules = self.progress.schedules()
        chunks = self.library.chunks()
        due = sum(
            1
            for chunk in chunks
            if chunk.id not in schedules or is_due(schedules[chunk.id], moment)
        )
        minutes = self.session_minutes(moment)
        return Snapshot(
            level=self.zpd.level,
            zpd=self.zpd.status(),
            load=self.load.estimate(minutes),
            calibration=self.meta.reading(),
            due_now=due,
            pending_recalls=len(self.progress.pending_recalls()),
            session_minutes=minutes,
        )

    def ask(self, now=None):
        moment = now or self.clock()
        chunks = self.library.chunks()
        if not chunks:
            raise NothingToStudy("Add some notes first, then come back to the quiz.")
        by_id = {chunk.id: chunk for chunk in chunks}
        recalls = [chunk_id for chunk_id in self.progress.due_recalls(moment) if chunk_id in by_id]

        plan = make_plan(
            level=self.zpd.level,
            zone=self.zpd.status().zone,
            load=self.load.estimate(self.session_minutes(moment)),
            intervention=self.meta.intervention(),
            recall_due=bool(recalls),
            max_level=self.tuning.zpd.max_level,
            config=self.tuning.agent,
        )
        if plan.level != self.zpd.level:
            self.zpd.set_level(plan.level)
            self.progress.set_state("level", plan.level)

        schedules = self.progress.schedules()
        order = [chunk.id for chunk in chunks]
        chosen = choose_chunk(order, schedules, moment, recalls, tuple(self.recent))
        self.recent = [*(item for item in self.recent if item != chosen), chosen][-len(order) :]
        chunk = by_id[chosen]
        waiting = schedules.get(chosen)
        if waiting is not None and not is_due(waiting, moment) and chosen not in recalls:
            plan = replace(plan, reasons=(*plan.reasons, EXTRA_PRACTICE))
        generated = self.writer.question(chunk, plan.level, plan.support)
        example = ""
        if plan.support is Support.WORKED_EXAMPLE:
            example = self.writer.worked_example(chunk)
        question = Question(
            chunk_id=chunk.id,
            text=generated.text,
            answer_key=generated.answer_key,
            level=plan.level,
            support=plan.support,
            worked_example=example,
            delayed_recall=plan.delayed_recall,
        )
        return AskedQuestion(question, chunk, plan, moment)

    def answer(self, asked, student_answer, confidence, response_seconds, now=None):
        moment = now or self.clock()
        question = asked.question
        if self.library.chunk(question.chunk_id) is None:
            raise StaleQuestion("The notes for this question were removed. Ask for a new question.")
        verdict = self.grader.grade(question.text, question.answer_key, student_answer, asked.chunk)
        config = self.tuning.agent
        correct = verdict.score >= config.correct_from
        grade = grade_for(verdict.score, confidence, config)
        seconds = max(float(response_seconds), 0.1)

        self.meta.record(confidence, correct)
        reading = self.meta.reading()

        current = self.progress.schedule_for(question.chunk_id)
        early = current is not None and not is_due(current, moment)
        counted = not (early and grade >= self.tuning.sm2.pass_grade)
        if counted:
            schedule = next_schedule(current, grade, moment, self.tuning.sm2)
            schedule = adjust_for_confidence(
                schedule, confidence, correct, reading.state, self.tuning.ease
            )
        else:
            schedule = current
        self.progress.save_schedule(question.chunk_id, schedule)
        self.progress.record_attempt(
            Attempt(
                chunk_id=question.chunk_id,
                asked_at=moment,
                level=question.level,
                confidence=confidence,
                score=verdict.score,
                correct=correct,
                response_seconds=seconds,
                delayed_recall=question.delayed_recall,
            )
        )

        self.zpd.record(verdict.score)
        self.load.record(verdict.score, seconds)

        if question.delayed_recall:
            self.progress.clear_recall(question.chunk_id)
        queued = (
            reading.state is CalibrationState.OVERCONFIDENT
            and confidence is Confidence.SURE
            and not correct
        )
        if queued:
            due_at = moment + timedelta(minutes=config.delayed_recall_minutes)
            self.progress.queue_recall(question.chunk_id, due_at)

        level = self.zpd.follow_suggestion()
        self.progress.set_state("level", level)

        return Outcome(
            score=verdict.score,
            correct=correct,
            feedback=verdict.feedback,
            marked_by=verdict.marked_by,
            answer_key=question.answer_key,
            grade=grade,
            schedule=schedule,
            calibration=reading,
            load=self.load.estimate(self.session_minutes(moment)),
            level=level,
            queued_recall=queued,
            counted=counted,
        )
