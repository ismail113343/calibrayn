import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from statistics import fmean

from calibrayn.config import Tuning
from calibrayn.evaluation.personas import PERSONAS
from calibrayn.evaluation.stats import Interval, bootstrap_interval
from calibrayn.models.calibration import CalibrationState, Confidence
from calibrayn.orchestrator.metacognition import MetacognitionTracker
from calibrayn.orchestrator.selector import choose_chunk
from calibrayn.scheduler.ease_adjuster import adjust_for_confidence
from calibrayn.scheduler.sm2 import is_due, new_schedule, next_schedule

SURE_FROM = 0.75
UNSURE_FROM = 0.4
MINUTES_PER_DAY = 1440.0

METRICS = (
    "retention",
    "known_share",
    "reviews",
    "known_per_100_reviews",
    "confident_errors",
    "recalls",
    "mean_easiness",
)

PARAMETERS = {
    "confident_error_penalty": ("ease", "confident_error_penalty"),
    "persistent_overconfidence_extra": ("ease", "persistent_overconfidence_extra"),
    "overconfidence_threshold": ("meta", "overconfidence_threshold"),
    "delayed_recall_minutes": ("agent", "delayed_recall_minutes"),
}


@dataclass(frozen=True)
class Condition:
    name: str
    uses_confidence: bool


PLAIN = Condition("plain SM-2", False)
ADJUSTED = Condition("confidence-adjusted", True)


@dataclass(frozen=True)
class Setup:
    topics: int = 40
    days: int = 60
    start: datetime = datetime(2026, 1, 5, 9, 0)
    known_from: float = 0.8

    def __post_init__(self) -> None:
        if self.topics < 1:
            raise ValueError("topics must be at least 1")
        if self.days < 1:
            raise ValueError("days must be at least 1")
        if not 0.0 < self.known_from <= 1.0:
            raise ValueError("known_from must be above 0 and at most 1")


@dataclass(frozen=True)
class RunResult:
    retention: float
    known_share: float
    reviews: int
    known_per_100_reviews: float
    confident_errors: int
    recalls: int
    mean_easiness: float
    weekly_retention: tuple[float, ...]


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    plain: Interval
    adjusted: Interval
    difference: Interval


@dataclass(frozen=True)
class PersonaComparison:
    persona: str
    seeds: int
    metrics: tuple[MetricComparison, ...]
    plain_curve: tuple[float, ...]
    adjusted_curve: tuple[float, ...]


@dataclass(frozen=True)
class SensitivityRow:
    parameter: str
    value: float
    retention: Interval
    difference: Interval


def grade_from(success, probability):
    if success:
        if probability >= 0.85:
            return 5
        return 4 if probability >= 0.6 else 3
    if probability >= 0.5:
        return 2
    return 1 if probability >= 0.25 else 0


def confidence_from(persona, probability, rng):
    feeling = probability + persona.confidence_bias + rng.gauss(0.0, persona.confidence_noise)
    if feeling >= SURE_FROM:
        return Confidence.SURE
    if feeling >= UNSURE_FROM:
        return Confidence.UNSURE
    return Confidence.GUESSING


def recall_probability(memory, moment, persona):
    if memory is None:
        return persona.new_topic_recall
    half_life, seen_at = memory
    elapsed_days = (moment - seen_at).total_seconds() / (MINUTES_PER_DAY * 60.0)
    return 0.5 ** (elapsed_days / half_life)


class Learner:
    def __init__(self, persona, condition, rng, setup, tuning):
        self.persona = persona
        self.condition = condition
        self.rng = rng
        self.setup = setup
        self.tuning = tuning
        self.order = list(range(1, setup.topics + 1))
        self.schedules = {}
        self.memories = {}
        self.recalls = {}
        self.tracker = MetacognitionTracker(tuning.meta)
        self.reviews = 0
        self.confident_errors = 0
        self.recall_reviews = 0

    def retention_at(self, moment):
        total = sum(
            recall_probability(memory, moment, self.persona) for memory in self.memories.values()
        )
        return total / self.setup.topics

    def known_at(self, moment):
        return sum(
            1
            for memory in self.memories.values()
            if recall_probability(memory, moment, self.persona) >= self.setup.known_from
        )

    def study(self, now):
        for _ in range(self.persona.reviews_per_session):
            waiting = [topic for topic, due in self.recalls.items() if due <= now]
            topic = choose_chunk(self.order, self.schedules, now, waiting)
            scheduled = self.schedules.get(topic)
            if scheduled is not None and topic not in waiting and not is_due(scheduled, now):
                return
            self.review(topic, now, topic in waiting)
            now += timedelta(minutes=self.persona.minutes_per_review)

    def review(self, topic, now, was_recall):
        persona = self.persona
        memory = self.memories.get(topic)
        probability = recall_probability(memory, now, persona)
        success = self.rng.random() < probability
        confidence = confidence_from(persona, probability, self.rng)

        grade = grade_from(success, probability)
        if confidence is Confidence.GUESSING:
            grade = min(grade, self.tuning.agent.guessing_grade_cap)

        half_life = memory[0] if memory else persona.starting_half_life
        if success:
            half_life *= 1.0 + persona.memory_gain * (1.0 - probability)
        else:
            half_life = max(persona.starting_half_life * 0.5, half_life * 0.4)
        self.memories[topic] = (half_life, now)

        schedule = self.schedules.get(topic) or new_schedule(self.tuning.sm2)
        schedule = next_schedule(schedule, grade, now, self.tuning.sm2)

        if self.condition.uses_confidence:
            self.tracker.record(confidence, success)
            state = self.tracker.reading().state
            schedule = adjust_for_confidence(schedule, confidence, success, state, self.tuning.ease)
            if was_recall:
                del self.recalls[topic]
            if (
                state is CalibrationState.OVERCONFIDENT
                and confidence is Confidence.SURE
                and not success
            ):
                delay = timedelta(minutes=self.tuning.agent.delayed_recall_minutes)
                self.recalls[topic] = now + delay

        self.schedules[topic] = schedule
        self.reviews += 1
        self.recall_reviews += 1 if was_recall else 0
        self.confident_errors += 1 if confidence is Confidence.SURE and not success else 0


DEFAULT_SETUP = Setup()
DEFAULT_TUNING = Tuning()


def simulate(persona, condition, seed, setup=DEFAULT_SETUP, tuning=DEFAULT_TUNING):
    learner = Learner(persona, condition, random.Random(seed), setup, tuning)
    weekly = []
    for day in range(setup.days):
        moment = setup.start + timedelta(days=day)
        if moment.weekday() in persona.study_days:
            learner.study(moment)
        if day % 7 == 6:
            weekly.append(learner.retention_at(moment + timedelta(days=1)))

    end = setup.start + timedelta(days=setup.days)
    known = learner.known_at(end)
    easiness = [schedule.easiness for schedule in learner.schedules.values()]
    return RunResult(
        retention=learner.retention_at(end),
        known_share=known / setup.topics,
        reviews=learner.reviews,
        known_per_100_reviews=known / learner.reviews * 100.0 if learner.reviews else 0.0,
        confident_errors=learner.confident_errors,
        recalls=learner.recall_reviews,
        mean_easiness=fmean(easiness) if easiness else tuning.sm2.initial_easiness,
        weekly_retention=tuple(weekly),
    )


def average_curve(runs):
    length = len(runs[0].weekly_retention)
    return tuple(fmean(run.weekly_retention[week] for run in runs) for week in range(length))


def compare(persona, seeds, setup=DEFAULT_SETUP, tuning=DEFAULT_TUNING):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    plain_runs = [simulate(persona, PLAIN, seed, setup, tuning) for seed in seeds]
    adjusted_runs = [simulate(persona, ADJUSTED, seed, setup, tuning) for seed in seeds]
    metrics = []
    for name in METRICS:
        plain = [getattr(run, name) for run in plain_runs]
        adjusted = [getattr(run, name) for run in adjusted_runs]
        gaps = [after - before for before, after in zip(plain, adjusted, strict=True)]
        metrics.append(
            MetricComparison(
                name,
                bootstrap_interval(plain, seed=1),
                bootstrap_interval(adjusted, seed=2),
                bootstrap_interval(gaps, seed=3),
            )
        )
    return PersonaComparison(
        persona=persona.name,
        seeds=len(seeds),
        metrics=tuple(metrics),
        plain_curve=average_curve(plain_runs),
        adjusted_curve=average_curve(adjusted_runs),
    )


def run_personas(names, seed_count, setup=DEFAULT_SETUP, tuning=DEFAULT_TUNING):
    chosen = []
    for name in names:
        if name not in PERSONAS:
            raise KeyError(f"Unknown persona '{name}'. Choose from: {', '.join(sorted(PERSONAS))}")
        chosen.append(PERSONAS[name])
    seeds = range(seed_count)
    return tuple(compare(persona, seeds, setup, tuning) for persona in chosen)


def sweep(persona, parameter, values, seeds, setup=DEFAULT_SETUP, tuning=DEFAULT_TUNING):
    if parameter not in PARAMETERS:
        raise KeyError(f"Unknown parameter '{parameter}'. Choose from: {', '.join(PARAMETERS)}")
    seeds = list(seeds)
    block, field = PARAMETERS[parameter]
    plain_runs = [simulate(persona, PLAIN, seed, setup, tuning) for seed in seeds]
    rows = []
    for value in values:
        changed = replace(tuning, **{block: replace(getattr(tuning, block), **{field: value})})
        runs = [simulate(persona, ADJUSTED, seed, setup, changed) for seed in seeds]
        retention = [run.retention for run in runs]
        gaps = [
            after.retention - before.retention
            for before, after in zip(plain_runs, runs, strict=True)
        ]
        rows.append(
            SensitivityRow(
                parameter,
                value,
                bootstrap_interval(retention, seed=4),
                bootstrap_interval(gaps, seed=5),
            )
        )
    return tuple(rows)
