import pandas as pd
import streamlit as st

from calibrayn.ui.components.labels import LOAD_TEXT, ZONE_TEXT, describe_calibration
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
second.metric("Started", summary.started)
third.metric("Due now", summary.due_now)
fourth.metric(
    "Average score", "none yet" if summary.accuracy is None else f"{summary.accuracy:.0%}"
)

if summary.attempts == 0:
    st.info("Answer a few quiz questions and your progress will show here.")
    st.stop()

snapshot = workspace.agent.snapshot()
st.subheader("Where you are now")
st.write(f"Level {snapshot.level}. Difficulty: {ZONE_TEXT[snapshot.zpd.zone]}.")
st.write(f"Mental load: {LOAD_TEXT[snapshot.load.level]}.")
st.write(describe_calibration(snapshot.calibration, workspace.tuning.meta.min_ratings_before_use))

st.subheader("Score by day")
daily = pd.DataFrame(
    [{"Day": point.day, "Average score": point.accuracy} for point in analytics.daily()]
)
st.line_chart(daily.set_index("Day"))

st.subheader("Confidence and results")
trend = analytics.calibration_trend()
if trend:
    frame = pd.DataFrame(
        [{"Answer": p.number, "Calibration score": p.score, "Bias": p.bias} for p in trend]
    )
    st.line_chart(frame.set_index("Answer"))
    st.caption(
        "Calibration score: 1 means your confidence matches your results. "
        "A bias above 0 means overconfident."
    )
else:
    st.write("This chart starts after ten answers.")

st.subheader("Difficulty level")
levels = pd.DataFrame(
    [{"Answer": number, "Level": level} for number, level in analytics.level_history()]
)
st.line_chart(levels.set_index("Answer"))

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
