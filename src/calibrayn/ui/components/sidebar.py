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
                "No language model is connected, so the offline model is answering. It builds "
                "simple questions from your own sentences. Open Setup to connect a real model."
            )
