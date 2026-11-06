import streamlit as st

from calibrayn.llm.base import LLMError
from calibrayn.ui.components.sidebar import show_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Chat", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_sidebar(workspace)

st.title("Chat")
st.caption("Ask a question. Answers come only from your notes, with sources.")

history = st.session_state.setdefault("chat_history", [])
for message in history:
    with st.chat_message(message["role"]):
        st.write(message["text"])
        if message.get("sources"):
            st.caption("Sources: " + "; ".join(message["sources"]))
        if message.get("warning"):
            st.warning(message["warning"])

question = st.chat_input("Ask about your notes")
if question:
    history.append({"role": "user", "text": question})
    try:
        reply = workspace.chat(question)
    except LLMError as error:
        history.append({"role": "assistant", "text": f"The model could not answer. {error}"})
    else:
        warning = None
        if reply.citations and not reply.grounded:
            warning = "Parts of this answer may not come from your notes. Check the sources."
        history.append(
            {
                "role": "assistant",
                "text": reply.text,
                "sources": list(dict.fromkeys(reply.citations)),
                "warning": warning,
            }
        )
    st.rerun()
