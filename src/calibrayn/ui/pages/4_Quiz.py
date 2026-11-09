import time

import streamlit as st

from calibrayn.llm.base import LLMError
from calibrayn.orchestrator.agent import NothingToStudy, StaleQuestion
from calibrayn.ui.components.labels import (
    CONFIDENCE_CHOICES,
    LOAD_TEXT,
    ZONE_TEXT,
    describe_calibration,
    describe_next_review,
)
from calibrayn.ui.components.sidebar import show_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Quiz", page_icon=":brain:", layout="wide")

workspace = get_workspace()
agent = workspace.agent
show_sidebar(workspace)

st.title("Quiz")
state = st.session_state
state.setdefault("quiz_stage", "idle")
state.setdefault("quiz_round", 0)
state.setdefault("quiz_log", [])


def forget_question(notice):
    state["quiz_stage"] = "idle"
    state["quiz_notice"] = notice
    state.pop("quiz_asked", None)
    state.pop("quiz_outcome", None)


if state["quiz_stage"] in ("asking", "feedback"):
    shown = state.get("quiz_asked")
    if shown is None or workspace.library.chunk(shown.chunk.id) is None:
        forget_question("Your notes changed, so that question is gone. Start a new session.")


def next_question():
    try:
        asked = agent.ask()
    except NothingToStudy as error:
        state["quiz_notice"] = str(error)
        return
    except LLMError as error:
        state["quiz_notice"] = f"The model could not write a question. {error}"
        return
    state["quiz_round"] += 1
    state["quiz_asked"] = asked
    state["quiz_clock"] = time.monotonic()
    state["quiz_stage"] = "asking"
    state["quiz_notice"] = ""


def begin_session():
    state["quiz_log"] = []
    agent.start_session()
    next_question()


snapshot = agent.snapshot()
needed = workspace.tuning.meta.min_ratings_before_use
log = state["quiz_log"]
with st.sidebar:
    st.subheader("Right now")
    st.metric("Level", snapshot.level, help="Difficulty level from 1 (easiest) to 5")
    st.write(f"Difficulty: {ZONE_TEXT[snapshot.zpd.zone]}")
    st.write(f"Mental load: {LOAD_TEXT[snapshot.load.level]}")
    st.write(describe_calibration(snapshot.calibration, needed))
    if log:
        average = sum(entry["score"] for entry in log) / len(log)
        noun = "answer" if len(log) == 1 else "answers"
        st.write(f"This session: {len(log)} {noun}, average {average:.0%}.")

if state.get("quiz_notice"):
    st.info(state["quiz_notice"])

stage = state["quiz_stage"]

if stage == "idle":
    st.write("Questions come from your notes. Say how sure you are, then answer.")
    if st.button("Start a session", key="start_quiz"):
        begin_session()
        st.rerun()

elif stage == "asking":
    asked = state["quiz_asked"]
    question = asked.question
    round_number = state["quiz_round"]

    if asked.plan.suggest_break:
        st.warning("You have worked for a while and it is getting hard. A short break may help.")
    if question.worked_example:
        st.info(question.worked_example)

    st.subheader(question.text)
    kind = "Recall check, no help" if question.delayed_recall else f"Level {question.level}"
    st.caption(f"{asked.chunk.title} ({workspace.citation(asked.chunk)}). {kind}.")
    with st.expander("Why this question?"):
        for reason in asked.plan.reasons:
            st.write(reason)

    choice = st.radio(
        "How sure are you before you answer?",
        list(CONFIDENCE_CHOICES),
        index=None,
        horizontal=True,
        key=f"confidence_{round_number}",
    )
    answer = st.text_area("Your answer", key=f"answer_{round_number}", height=150)
    if st.button("Submit answer", disabled=choice is None, key="submit_answer"):
        seconds = time.monotonic() - state["quiz_clock"]
        try:
            outcome = agent.answer(asked, answer, CONFIDENCE_CHOICES[choice], seconds)
        except StaleQuestion as error:
            forget_question(str(error))
        else:
            state["quiz_outcome"] = outcome
            state["quiz_stage"] = "feedback"
            log.append(
                {"score": outcome.score, "correct": outcome.correct, "title": asked.chunk.title}
            )
        st.rerun()

elif stage == "feedback":
    asked = state["quiz_asked"]
    outcome = state["quiz_outcome"]
    if outcome.correct:
        st.success(f"Good work. Score: {outcome.score:.0%}")
    else:
        st.error(f"Not yet. Score: {outcome.score:.0%}")
    st.write(outcome.feedback)
    if outcome.marked_by == "keywords":
        st.caption("Marked by comparing key words, because the model was not available.")
    st.markdown("**From your notes:**")
    st.write(outcome.answer_key)

    when = describe_next_review(outcome.schedule, workspace.clock())
    st.write(f"You will see this topic again {when}.")
    if not outcome.counted:
        st.caption("This was extra practice, so the schedule did not change.")
    st.caption(describe_calibration(outcome.calibration, needed))
    if outcome.level != asked.question.level:
        direction = "up" if outcome.level > asked.question.level else "down"
        st.info(f"Your level moved {direction} to {outcome.level}.")
    if outcome.queued_recall:
        minutes = workspace.tuning.agent.delayed_recall_minutes
        st.warning(f"You felt sure but missed this. It will come back in about {minutes} minutes.")

    again, finish = st.columns(2)
    if again.button("Next question", key="next_question"):
        next_question()
        st.rerun()
    if finish.button("End session", key="end_session"):
        state["quiz_stage"] = "summary"
        st.rerun()

else:
    st.subheader("Session summary")
    correct = sum(1 for entry in log if entry["correct"])
    average = sum(entry["score"] for entry in log) / len(log) if log else 0.0
    first, second, third, fourth = st.columns(4)
    first.metric("Answers", len(log))
    second.metric("Correct", correct)
    third.metric("Average score", f"{average:.0%}")
    fourth.metric("Minutes", f"{agent.session_minutes():.0f}")
    weakest = sorted((entry for entry in log if not entry["correct"]), key=lambda e: e["score"])
    if weakest:
        st.write("**Topics to revisit:**")
        for entry in weakest[:3]:
            st.write(f"{entry['title']} ({entry['score']:.0%})")
    else:
        st.write("Every answer was correct. Well done.")
    st.write(describe_calibration(snapshot.calibration, needed))
    if st.button("Start a new session", key="new_session"):
        begin_session()
        st.rerun()
