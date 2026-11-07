import streamlit as st

from calibrayn.config import Settings
from calibrayn.hub import use_token
from calibrayn.llm.base import LLMError
from calibrayn.services.workspace import Workspace


@st.cache_resource(show_spinner="Opening your study space...")
def open_workspace():
    settings = Settings()
    use_token(settings.hf_token)
    return Workspace.open(settings)


def get_workspace():
    try:
        return open_workspace()
    except (LLMError, ValueError) as error:
        st.error(f"Calibrayn could not start. {error}")
        st.stop()
