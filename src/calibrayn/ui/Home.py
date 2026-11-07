import streamlit as st

from calibrayn.ui.components.guidance import next_step, steps
from calibrayn.ui.components.sidebar import show_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Calibrayn", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_sidebar(workspace)

st.title("Calibrayn")
st.write(
    "A study companion that works on your own notes. It plans your revision, adjusts how hard "
    "the questions are, and checks whether your confidence matches your results."
)

summary = workspace.analytics.summary(workspace.clock())
first, second, third, fourth = st.columns(4)
first.metric("Documents", summary.documents, help="Files you have added")
second.metric("Topics", summary.topics, help="Each document is split into topics")
third.metric("Due now", summary.due_now, help="Topics ready to study")
fourth.metric("Answers so far", summary.attempts)

step = next_step(summary, workspace.client.name)
st.subheader("What to do next")
st.info(f"**{step.title}.** {step.text}")

with st.expander("Getting started", expanded=summary.documents == 0):
    for item in steps(summary, workspace.client.name):
        label = "Done" if item.done else "To do"
        st.write(f"**{label}: {item.title}.** {item.text}")

st.subheader("The pages")
st.write("**Upload** adds notes. **Chat** answers questions from them, with sources.")
st.write("**Tutor** explains a topic with a worked example. **Quiz** is the main practice.")
st.write(
    "**Dashboard** shows progress. **Evaluation** tests the system. **Setup** connects a model."
)
