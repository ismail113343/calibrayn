import streamlit as st

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
first.metric("Documents", summary.documents)
second.metric("Topics", summary.topics)
third.metric("Due now", summary.due_now)
fourth.metric("Answers so far", summary.attempts)

if summary.documents == 0:
    st.info("Start on the Upload page. Add a PDF, Word, PowerPoint, text or Markdown file.")
else:
    st.write(
        "Use Quiz to practise, Tutor to learn a topic, Chat to ask questions, "
        "and Dashboard to see your progress."
    )
