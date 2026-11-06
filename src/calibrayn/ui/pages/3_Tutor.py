import streamlit as st

from calibrayn.llm.base import LLMError
from calibrayn.ui.components.sidebar import show_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Tutor", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_sidebar(workspace)

st.title("Tutor")
chunks = workspace.library.chunks()
if not chunks:
    st.info("Add some notes on the Upload page first.")
    st.stop()

labels = {chunk.id: f"{chunk.title} ({workspace.citation(chunk)})" for chunk in chunks}
chosen = st.selectbox("Pick a topic", list(labels), format_func=labels.get, key="tutor_topic")

if st.button("Explain this topic", key="explain_button"):
    try:
        with st.spinner("Writing the lesson..."):
            st.session_state["lesson"] = workspace.lesson(chosen)
    except LLMError as error:
        st.error(f"The model could not write the lesson. {error}")

lesson = st.session_state.get("lesson")
if lesson is not None and lesson.chunk_id == chosen:
    st.subheader("Explanation")
    st.write(lesson.explanation)
    st.subheader("Worked example")
    st.write(lesson.worked_example)
    st.caption(f"Source: {lesson.citation}")
