from calibrayn.llm.base import LLMError
from calibrayn.llm.http import get_json


def installed_ollama_models(base_url, fetch=get_json):
    try:
        reply = fetch(f"{base_url.rstrip('/')}/api/tags", 2.0)
    except LLMError:
        return []
    listed = reply.get("models") if isinstance(reply, dict) else None
    if not isinstance(listed, list):
        return []
    return [
        entry["name"]
        for entry in listed
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]


def chat_models(names):
    return [name for name in names if "embed" not in name.lower()]


def pick_model(wanted, installed):
    available = chat_models(installed)
    if wanted in available:
        return wanted
    family = wanted.split(":")[0]
    for name in available:
        if name.split(":")[0] == family:
            return name
    return available[0] if available else None
