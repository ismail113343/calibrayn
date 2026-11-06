from typing import Protocol


class LLMError(Exception):
    pass


class PrivacyError(LLMError):
    pass


class LLMClient(Protocol):
    name: str
    hosted: bool

    def complete(
        self, system: str, prompt: str, max_tokens: int = 400, temperature: float = 0.2
    ) -> str: ...


def require_text(text, source):
    if not isinstance(text, str) or not text.strip():
        raise LLMError(f"{source} sent back an empty reply.")
    return text.strip()
