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


def forget_question(notice):
    state["quiz_stage"] = "idle"
    state["quiz_notice"] = notice
    state.pop("quiz_asked", None)
    state.pop("quiz_outcome", None)


if state["quiz_stage"] != "idle":
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


snapshot = agent.snapshot()
needed = workspace.tuning.meta.min_ratings_before_use
with st.sidebar:
    st.subheader("Right now")
    st.metric("Level", snapshot.level)
    st.write(f"Difficulty: {ZONE_TEXT[snapshot.zpd.zone]}")
    st.write(f"Mental load: {LOAD_TEXT[snapshot.load.level]}")
    st.write(describe_calibration(snapshot.calibration, needed))

if state.get("quiz_notice"):
    st.info(state["quiz_notice"])

if state["quiz_stage"] == "idle":
    st.write("Questions come from your notes. Say how sure you are, then answer.")
    if st.button("Start a session", key="start_quiz"):
        agent.start_session()
        next_question()
        st.rerun()

elif state["quiz_stage"] == "asking":
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
        st.rerun()

else:
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
    st.caption(describe_calibration(outcome.calibration, needed))
    if outcome.level != asked.question.level:
        direction = "up" if outcome.level > asked.question.level else "down"
        st.info(f"Your level moved {direction} to {outcome.level}.")
    if outcome.queued_recall:
        minutes = workspace.tuning.agent.delayed_recall_minutes
        st.warning(f"You felt sure but missed this. It will come back in about {minutes} minutes.")

    if st.button("Next question", key="next_question"):
        next_question()
        st.rerun()
