import pandas as pd
import streamlit as st

from calibrayn.ui.components.labels import LOAD_TEXT, ZONE_TEXT, describe_calibration
from calibrayn.ui.components.palette import AMBER, BLUE, GREEN, TEAL
from calibrayn.ui.components.sidebar import show_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Dashboard", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_sidebar(workspace)

st.title("Dashboard")
analytics = workspace.analytics
summary = analytics.summary(workspace.clock())

first, second, third, fourth = st.columns(4)
first.metric("Topics", summary.topics)
second.metric("Started", summary.started, help="Topics you have answered at least once")
third.metric("Due now", summary.due_now, help="Topics ready to study")
fourth.metric(
    "Average score", "none yet" if summary.accuracy is None else f"{summary.accuracy:.0%}"
)

if summary.attempts == 0:
    st.info("Answer a few quiz questions and your progress will show here.")
    st.stop()

with st.expander("How to read this page"):
    st.write("Score is how well your answers matched your notes, from 0 to 100 percent.")
    st.write(
        "Level is the difficulty. It rises when answers are easy and falls when they are hard."
    )
    st.write(
        "The confidence charts compare how sure you said you were with how often you were right."
    )

snapshot = workspace.agent.snapshot()
st.subheader("Where you are now")
st.write(f"Level {snapshot.level}. Difficulty: {ZONE_TEXT[snapshot.zpd.zone]}.")
st.write(f"Mental load: {LOAD_TEXT[snapshot.load.level]}.")
st.write(describe_calibration(snapshot.calibration, workspace.tuning.meta.min_ratings_before_use))

st.subheader("Score by day")
daily = pd.DataFrame(
    [{"Day": point.day, "Average score (%)": point.accuracy * 100} for point in analytics.daily()]
)
st.line_chart(daily.set_index("Day"), color=BLUE)

st.subheader("Confidence and results")
trend = analytics.calibration_trend()
if trend:
    left, right = st.columns(2)
    scores = pd.DataFrame(
        [{"Answer": p.number, "Calibration score (%)": p.score * 100} for p in trend]
    )
    biases = pd.DataFrame([{"Answer": p.number, "Bias (%)": p.bias * 100} for p in trend])
    left.line_chart(scores.set_index("Answer"), color=GREEN)
    right.line_chart(biases.set_index("Answer"), color=AMBER)
    st.caption(
        "Left: 100 means your confidence matches your results. "
        "Right: above 0 means overconfident, below 0 means under-confident."
    )
else:
    st.write("These charts start after ten answers.")

st.subheader("Difficulty level")
levels = pd.DataFrame(
    [{"Answer": number, "Level": level} for number, level in analytics.level_history()]
)
st.line_chart(levels.set_index("Answer"), color=TEAL)

st.subheader("Topics to revisit")
weak = analytics.weakest(limit=5)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Topic": topic.title,
                "Document": topic.document,
                "Answers": topic.attempts,
                "Average score": f"{topic.accuracy:.0%}",
            }
            for topic in weak
        ]
    ),
    hide_index=True,
)
