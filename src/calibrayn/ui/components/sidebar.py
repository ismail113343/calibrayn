import streamlit as st


def show_sidebar(workspace):
    client = workspace.client
    with st.sidebar:
        st.header("Calibrayn")
        if client.hosted:
            st.caption("Privacy: your notes are sent to an online AI model.")
        else:
            st.caption("Privacy: your notes stay on this computer.")
        if client.name == "mock":
            st.caption("Basic mode: no AI model is connected, so questions are simple.")


def show_console_sidebar(workspace):
    client = workspace.client
    with st.sidebar:
        st.header("Researcher console")
        st.caption(f"Model: {client.name}")
        st.caption(f"Embeddings: {workspace.retriever.embedder.name}")
        st.caption(f"Privacy mode: {workspace.settings.privacy_mode}")
        if client.hosted:
            st.caption("Notes are sent to a hosted model.")
        else:
            st.caption("Notes stay on this computer.")
        if client.name == "mock":
            st.warning("No language model is connected. The offline test model is answering.")
