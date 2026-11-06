import streamlit as st


def show_sidebar(workspace):
    client = workspace.client
    with st.sidebar:
        st.header("Calibrayn")
        st.caption(f"Model: {client.name}")
        st.caption(f"Embeddings: {workspace.retriever.embedder.name}")
        if client.hosted:
            st.caption("Privacy: your notes are sent to a hosted model.")
        else:
            st.caption("Privacy: your notes stay on this computer.")
        if client.name == "mock":
            st.warning(
                "This is the offline demo model. It builds simple questions and answers from "
                "your own sentences. Set LLM_PROVIDER in your .env file to use a real model."
            )
