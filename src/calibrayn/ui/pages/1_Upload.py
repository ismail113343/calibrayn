import streamlit as st

from calibrayn.ingestion.loaders import NoTextFound, UnreadableFile, UnsupportedFile
from calibrayn.ui.components.sidebar import show_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Upload notes", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_sidebar(workspace)

st.title("Upload notes")
files = st.file_uploader(
    "Add your notes",
    type=["pdf", "docx", "pptx", "txt", "md"],
    accept_multiple_files=True,
    key="uploader",
)
if st.button("Add to library", disabled=not files, key="add_button"):
    for file in files:
        try:
            with st.spinner(f"Reading {file.name}..."):
                _, chunks = workspace.add_document(file.name, file.getvalue())
            st.success(f"{file.name}: {len(chunks)} topics created.")
        except (UnsupportedFile, UnreadableFile, NoTextFound) as error:
            st.error(f"{file.name}: {error}")

st.subheader("Your library")
documents = workspace.library.documents()
if not documents:
    st.write("Nothing here yet.")
for document in documents:
    topics = len(workspace.library.chunks(document.id))
    left, right = st.columns([5, 1])
    left.write(f"**{document.name}**: {topics} topics, added {document.added_at:%d %b %Y}")
    if right.button("Remove", key=f"remove_{document.id}"):
        workspace.remove_document(document.id)
        st.session_state["quiz_stage"] = "idle"
        st.session_state.pop("quiz_asked", None)
        st.session_state.pop("quiz_outcome", None)
        st.session_state.pop("lesson", None)
        st.rerun()
