from dataclasses import dataclass

from calibrayn.hub.cache import is_cached, packages_available


@dataclass(frozen=True)
class HubStatus:
    packages_ready: bool
    embedding_cached: bool
    llm_cached: bool
    token_set: bool
    missing_packages: tuple


def hub_status(settings, cached=is_cached, available=packages_available):
    found = available()
    needed = ("transformers", "sentence_transformers")
    missing = tuple(name for name in needed if not found.get(name))
    return HubStatus(
        packages_ready=not missing,
        embedding_cached=cached(settings.hf_embedding_model),
        llm_cached=cached(settings.hf_llm_model),
        token_set=bool(settings.hf_token),
        missing_packages=missing,
    )
