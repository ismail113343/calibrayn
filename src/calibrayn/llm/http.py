import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from calibrayn.llm.base import LLMError, require_text

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_local_url(url):
    return (urlparse(url).hostname or "") in LOCAL_HOSTS


def post_json(url, payload, headers=None, timeout=120.0):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise LLMError(f"The model server answered with status {error.code}.") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise LLMError(f"Could not get a reply from {url}: {error}") from error


def get_json(url, timeout=5.0):
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise LLMError(f"The server answered with status {error.code}.") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise LLMError(f"Could not get a reply from {url}: {error}") from error


class OllamaClient:
    def __init__(self, base_url, model, transport=post_json):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport = transport
        self.name = f"ollama:{model}"
        self.hosted = not is_local_url(base_url)

    def complete(self, system, prompt, max_tokens=400, temperature=0.2):
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        reply = self.transport(f"{self.base_url}/api/chat", body, {}, 120.0)
        try:
            text = reply["message"]["content"]
        except (KeyError, TypeError) as error:
            raise LLMError("Ollama sent a reply in an unexpected shape.") from error
        return require_text(text, "Ollama")


class OpenAICompatibleClient:
    def __init__(self, base_url, api_key, model, transport=post_json):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.transport = transport
        self.name = f"openai:{model}"
        self.hosted = not is_local_url(base_url)

    def complete(self, system, prompt, max_tokens=400, temperature=0.2):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        reply = self.transport(f"{self.base_url}/chat/completions", body, headers, 120.0)
        try:
            text = reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("The model server sent a reply in an unexpected shape.") from error
        return require_text(text, "The model server")
