from calibrayn.llm.base import LLMError, PrivacyError
from calibrayn.llm.hf import HuggingFaceInferenceClient, HuggingFaceLocalClient
from calibrayn.llm.http import OllamaClient, OpenAICompatibleClient
from calibrayn.llm.mock import MockClient

PROVIDERS = ("mock", "hf_local", "hf_inference", "ollama", "openai")


def build_client(settings):
    provider = settings.llm_provider.strip().lower()
    if provider == "mock":
        client = MockClient()
    elif provider == "hf_local":
        client = HuggingFaceLocalClient(settings.hf_llm_model)
    elif provider == "hf_inference":
        if not settings.hf_token:
            raise LLMError("hf_inference needs HF_TOKEN in your .env file.")
        client = HuggingFaceInferenceClient(settings.hf_llm_model, settings.hf_token)
    elif provider == "ollama":
        client = OllamaClient(settings.ollama_base_url, settings.ollama_model)
    elif provider == "openai":
        client = OpenAICompatibleClient(
            settings.openai_base_url, settings.openai_api_key, settings.openai_model
        )
        if client.hosted and not settings.openai_api_key:
            raise LLMError("openai needs OPENAI_API_KEY in your .env file.")
    else:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use one of: " + ", ".join(PROVIDERS)
        )

    if client.hosted and settings.privacy_mode.strip().lower() != "remote_allowed":
        raise PrivacyError(
            f"{client.name} sends your notes to another server, but PRIVACY_MODE is 'local'. "
            "Set PRIVACY_MODE=remote_allowed to allow it, or choose a local provider."
        )
    return client
