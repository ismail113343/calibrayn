import streamlit as st

from calibrayn.ingestion.loaders import NoTextFound, UnreadableFile, UnsupportedFile
from calibrayn.ui.components.samples import sample_files
from calibrayn.ui.components.sidebar import show_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Upload notes", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_sidebar(workspace)

st.title("Upload notes")
st.caption("PDF, Word, PowerPoint, text and Markdown files are supported. Scanned pages are not.")

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

known = {document.name for document in workspace.library.documents()}
if st.button("Try the sample notes (biology)", key="sample_button"):
    added = 0
    for name, content in sample_files():
        if name not in known:
            workspace.add_document(name, content)
            added += 1
    st.success(
        f"Added {added} sample documents." if added else "The sample notes are already here."
    )
    st.rerun()

st.subheader("Your library")
documents = workspace.library.documents()
if not documents:
    st.write("Nothing here yet. Add a file above, or try the sample notes.")
for document in documents:
    chunks = workspace.library.chunks(document.id)
    left, right = st.columns([5, 1])
    left.write(f"**{document.name}**: {len(chunks)} topics, added {document.added_at:%d %b %Y}")
    if right.button("Remove", key=f"remove_{document.id}"):
        workspace.remove_document(document.id)
        st.session_state["quiz_stage"] = "idle"
        st.session_state["quiz_log"] = []
        st.session_state.pop("quiz_asked", None)
        st.session_state.pop("quiz_outcome", None)
        st.session_state.pop("lesson", None)
        st.rerun()
    with st.expander("Show topics"):
        for chunk in chunks:
            st.text(f"{chunk.title} ({workspace.citation(chunk)})")
