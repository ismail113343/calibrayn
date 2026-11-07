import streamlit as st

from calibrayn.ui.components.sidebar import show_console_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Researcher console", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_console_sidebar(workspace)

st.title("Researcher console")
st.write(
    "Tools for building and testing Calibrayn: connect models, check search, test the grader and "
    "run simulations. Students use the main app and never need this."
)

first, second, third = st.columns(3)
first.metric("Language model", workspace.client.name)
second.metric("Embeddings", workspace.retriever.embedder.name)
third.metric("Privacy mode", workspace.settings.privacy_mode)

st.subheader("Pages")
st.write("**Setup** shows what is connected, downloads Hugging Face models and tests the model.")
st.write(
    "**Evaluation** checks search on your notes and on public datasets, and runs the learner "
    "simulation."
)

st.subheader("Command line")
st.code(
    "python -m calibrayn.evaluation questions\n"
    "python -m calibrayn.evaluation retrieval\n"
    "python -m calibrayn.evaluation simulate\n"
    "python -m calibrayn.hub benchmark organisation/dataset --question-field q --context-field c\n"
    "python -m calibrayn.hub grader nkazi/Beetle --split test_ua ...",
    language="bash",
)
st.write("The student app is started with: streamlit run src/calibrayn/ui/Home.py")
