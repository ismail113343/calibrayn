import time

import pandas as pd
import streamlit as st

from calibrayn.hub.models import download_model
from calibrayn.hub.rows import HubError
from calibrayn.hub.status import hub_status
from calibrayn.llm.base import LLMError
from calibrayn.llm.discovery import installed_ollama_models
from calibrayn.ui.components.sidebar import show_sidebar
from calibrayn.ui.state import get_workspace

st.set_page_config(page_title="Setup", page_icon=":brain:", layout="wide")

workspace = get_workspace()
show_sidebar(workspace)
client = workspace.client
settings = workspace.settings

st.title("Model setup")
st.write(
    "Calibrayn writes questions and marks answers with a language model. "
    "Here is how to connect one."
)

st.subheader("What is connected now")
where = "A hosted server" if client.hosted else "This computer"
st.dataframe(
    pd.DataFrame(
        [
            ("Language model", client.name),
            ("Runs on", where),
            ("Embeddings", workspace.retriever.embedder.name),
            ("Privacy mode", settings.privacy_mode),
        ],
        columns=["Setting", "Value"],
    ),
    hide_index=True,
)
if client.name == "mock":
    st.warning("The offline model is a stand-in for testing. Connect a real model below.")

st.subheader("Option 1: a model on this computer (recommended)")
installed = installed_ollama_models(settings.ollama_base_url)
if installed:
    st.success("Ollama is running here with: " + ", ".join(installed))
    st.write("Restart the app and it will use one of these automatically.")
else:
    st.info(f"Ollama was not found at {settings.ollama_base_url}. Install it, then pull a model.")
st.code("winget install Ollama.Ollama\nollama pull qwen2.5:3b", language="powershell")
st.write("Your notes never leave this computer. A computer with 8 GB of memory or more is advised.")

st.subheader("Option 2: Hugging Face models on this computer")
status = hub_status(settings)
st.write(
    "The app can use models from the Hugging Face Hub. Download them once and it finds them by "
    "itself the next time it starts."
)
packages = "Installed" if status.packages_ready else "Not installed"
st.dataframe(
    pd.DataFrame(
        [
            ("Packages for local models", packages),
            (
                f"Embedding model: {settings.hf_embedding_model}",
                "Downloaded" if status.embedding_cached else "Not downloaded",
            ),
            (
                f"Language model: {settings.hf_llm_model}",
                "Downloaded" if status.llm_cached else "Not downloaded",
            ),
            (
                "Hugging Face token",
                "Set" if status.token_set else "Not set (needed only for private or gated models)",
            ),
        ],
        columns=["Item", "Status"],
    ),
    hide_index=True,
)
if not status.packages_ready:
    st.info("Install the packages first: python -m pip install -r requirements-ml.txt")
first, second = st.columns(2)
for column, label, key, repo in (
    (
        first,
        "Download the embedding model (small)",
        "download_embedding",
        settings.hf_embedding_model,
    ),
    (second, "Download the language model (large)", "download_llm", settings.hf_llm_model),
):
    if column.button(label, key=key):
        try:
            with st.spinner(f"Downloading {repo}..."):
                saved = download_model(repo, settings.hf_token)
        except HubError as error:
            st.error(str(error))
        else:
            st.success(f"Downloaded {repo}. Restart the app to use it.")
            st.caption(f"Saved in {saved}")
st.write(
    "The language model is several gigabytes, so check its size on its Hugging Face page first."
)

st.subheader("Other options")
with st.expander("Hugging Face model on this computer"):
    st.write("Install the extra packages, then set these lines in the .env file.")
    st.code("python -m pip install -r requirements-ml.txt", language="powershell")
    st.code("LLM_PROVIDER=hf_local\nEMBEDDER=huggingface", language="ini")
with st.expander("A hosted model (Hugging Face API, DeepSeek or similar)"):
    st.write(
        "Hosted models send your notes to another server. They are refused unless you allow it. "
        "Use them only with notes that are not sensitive."
    )
    st.code(
        "LLM_PROVIDER=openai\nPRIVACY_MODE=remote_allowed\nOPENAI_BASE_URL=https://api.deepseek.com\n"
        "OPENAI_API_KEY=your key\nOPENAI_MODEL=the model name from the provider",
        language="ini",
    )
    st.code(
        "LLM_PROVIDER=hf_inference\nPRIVACY_MODE=remote_allowed\nHF_TOKEN=your token",
        language="ini",
    )
st.write("After changing the .env file, stop the app with Ctrl+C and start it again.")

st.subheader("Test the connection")
if st.button("Test the model", key="test_model"):
    if client.name == "mock":
        st.info("The offline model needs no test. Connect a real model to test it.")
    else:
        started = time.monotonic()
        try:
            with st.spinner("Asking the model..."):
                reply = client.complete(
                    "Answer in one short sentence.", "What is two plus two?", 40, 0.0
                )
        except LLMError as error:
            st.error(f"The model did not answer. {error}")
        else:
            st.success(f"The model answered in {time.monotonic() - started:.1f} seconds.")
            st.write(reply)
