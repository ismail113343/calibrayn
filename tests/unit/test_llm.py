import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from calibrayn.llm.base import LLMError, PrivacyError, require_text
from calibrayn.llm.discovery import chat_models, installed_ollama_models, pick_model
from calibrayn.llm.factory import build_client
from calibrayn.llm.hf import HuggingFaceInferenceClient, HuggingFaceLocalClient
from calibrayn.llm.http import (
    OllamaClient,
    OpenAICompatibleClient,
    get_json,
    is_local_url,
    post_json,
)
from calibrayn.llm.mock import MockClient, block

PASSAGE = (
    "Photosynthesis turns light into chemical energy. Plants use chlorophyll to absorb light. "
    "The process makes glucose and oxygen. Oxygen is released into the air."
)


def settings(**changes):
    values = {
        "llm_provider": "mock",
        "privacy_mode": "local",
        "hf_token": "",
        "hf_llm_model": "some/model",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama3.2",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_api_key": "",
        "openai_model": "chat-model",
    }
    values.update(changes)
    return SimpleNamespace(**values)


class TestRequireText:
    def test_text_is_trimmed(self):
        assert require_text("  hello \n", "x") == "hello"

    @pytest.mark.parametrize("value", ["", "   ", None, 5])
    def test_empty_or_wrong_values_are_rejected(self, value):
        with pytest.raises(LLMError):
            require_text(value, "The model")


class TestMock:
    def test_block_finds_the_text_between_markers(self):
        prompt = "A\nPASSAGE:\nline one\nline two\nEND PASSAGE\nB"
        assert block(prompt, "PASSAGE") == "line one\nline two"
        assert block(prompt, "MISSING") == ""

    def test_a_question_comes_back_as_json_with_an_answer_key(self):
        prompt = f"Difficulty level: 2 out of 5.\nPASSAGE:\n{PASSAGE}\nEND PASSAGE"
        reply = json.loads(MockClient().complete("TASK: question", prompt))
        assert "Photosynthesis turns light into" in reply["question"]
        assert reply["answer"].startswith("Photosynthesis turns light")

    @pytest.mark.parametrize(
        ("level", "start"), [(1, "What do"), (3, "Explain in"), (5, "Explain why")]
    )
    def test_the_question_style_follows_the_level(self, level, start):
        prompt = f"Difficulty level: {level} out of 5.\nPASSAGE:\n{PASSAGE}\nEND PASSAGE"
        reply = json.loads(MockClient().complete("TASK: question", prompt))
        assert reply["question"].startswith(start)

    def test_a_full_answer_scores_high(self):
        prompt = (
            f"ANSWER KEY:\n{PASSAGE}\nEND ANSWER KEY\n"
            f"STUDENT ANSWER:\n{PASSAGE}\nEND STUDENT ANSWER"
        )
        reply = json.loads(MockClient().complete("TASK: grade", prompt))
        assert reply["score"] == 1.0

    def test_an_unrelated_answer_scores_zero_and_lists_missing_ideas(self):
        prompt = (
            f"ANSWER KEY:\n{PASSAGE}\nEND ANSWER KEY\n"
            "STUDENT ANSWER:\nBananas are yellow.\nEND STUDENT ANSWER"
        )
        reply = json.loads(MockClient().complete("TASK: grade", prompt))
        assert reply["score"] == 0.0
        assert reply["feedback"].startswith("Try to include:")

    def test_grading_without_a_key_scores_zero(self):
        prompt = "ANSWER KEY:\n\nEND ANSWER KEY\nSTUDENT ANSWER:\nanything\nEND STUDENT ANSWER"
        assert json.loads(MockClient().complete("TASK: grade", prompt))["score"] == 0.0

    def test_explanation_and_example_use_the_passage(self):
        prompt = f"PASSAGE:\n{PASSAGE}\nEND PASSAGE"
        assert MockClient().complete("TASK: explain", prompt).startswith("In short: Photosynthesis")
        example = MockClient().complete("TASK: example", prompt)
        assert "Step 1: Photosynthesis" in example
        assert "Step 4:" in example

    def test_an_answer_picks_the_best_passage_and_cites_it(self):
        prompt = (
            "QUESTION:\nWhat does chlorophyll absorb?\nEND QUESTION\n"
            "PASSAGES:\n[1] Cells divide by mitosis.\n[2] Plants use chlorophyll to absorb light.\n"
            "END PASSAGES"
        )
        reply = MockClient().complete("TASK: answer", prompt)
        assert reply.endswith("[2]")
        assert "chlorophyll" in reply

    def test_an_answer_admits_when_nothing_matches(self):
        prompt = (
            "QUESTION:\nquantum gravity\nEND QUESTION\nPASSAGES:\n[1] Cells divide.\nEND PASSAGES"
        )
        assert (
            MockClient().complete("TASK: answer", prompt) == "I could not find this in your notes."
        )

    def test_a_missing_task_is_an_error(self):
        with pytest.raises(LLMError):
            MockClient().complete("no task here", "prompt")

    def test_an_unknown_task_is_an_error(self):
        with pytest.raises(LLMError):
            MockClient().complete("TASK: dance", "prompt")

    def test_the_mock_is_local(self):
        assert MockClient().hosted is False


def fake_reply(text):
    message = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestHuggingFaceInference:
    def test_it_sends_chat_messages_and_returns_the_text(self):
        seen = {}

        class FakeClient:
            def chat_completion(self, messages, max_tokens, temperature):
                seen.update(messages=messages, max_tokens=max_tokens, temperature=temperature)
                return fake_reply("  the reply ")

        client = HuggingFaceInferenceClient("some/model", "token", client=FakeClient())
        assert client.complete("be brief", "hello", max_tokens=50, temperature=0.5) == "the reply"
        assert seen["messages"][0] == {"role": "system", "content": "be brief"}
        assert seen["messages"][1] == {"role": "user", "content": "hello"}
        assert seen["max_tokens"] == 50
        assert client.hosted is True
        assert client.name == "hf_inference:some/model"

    def test_a_server_failure_becomes_an_llm_error(self):
        class FakeClient:
            def chat_completion(self, **_):
                raise RuntimeError("rate limit")

        client = HuggingFaceInferenceClient("m", "t", client=FakeClient())
        with pytest.raises(LLMError, match="rate limit"):
            client.complete("s", "p")

    def test_a_strange_reply_becomes_an_llm_error(self):
        class FakeClient:
            def chat_completion(self, **_):
                return SimpleNamespace(choices=[])

        client = HuggingFaceInferenceClient("m", "t", client=FakeClient())
        with pytest.raises(LLMError):
            client.complete("s", "p")

    def test_an_empty_reply_becomes_an_llm_error(self):
        class FakeClient:
            def chat_completion(self, **_):
                return fake_reply("  ")

        client = HuggingFaceInferenceClient("m", "t", client=FakeClient())
        with pytest.raises(LLMError):
            client.complete("s", "p")

    def test_the_real_client_can_be_created_without_a_network_call(self):
        client = HuggingFaceInferenceClient("some/model", "token")
        assert client.hosted is True


class TestHuggingFaceLocal:
    def test_it_runs_the_pipeline_and_returns_the_last_message(self):
        calls = []

        def pipe(messages, **options):
            calls.append((messages, options))
            return [{"generated_text": [*messages, {"role": "assistant", "content": " done "}]}]

        client = HuggingFaceLocalClient("some/model", pipeline=pipe)
        assert client.complete("sys", "user", max_tokens=20, temperature=0.0) == "done"
        assert calls[0][1]["max_new_tokens"] == 20
        assert calls[0][1]["do_sample"] is False
        assert client.hosted is False

    def test_sampling_is_on_when_the_temperature_is_positive(self):
        seen = {}

        def pipe(messages, **options):
            seen.update(options)
            return [{"generated_text": [{"role": "assistant", "content": "x"}]}]

        HuggingFaceLocalClient("m", pipeline=pipe).complete("s", "p", temperature=0.7)
        assert seen["do_sample"] is True
        assert seen["temperature"] == 0.7

    def test_a_pipeline_failure_becomes_an_llm_error(self):
        def pipe(messages, **options):
            raise RuntimeError("out of memory")

        with pytest.raises(LLMError, match="out of memory"):
            HuggingFaceLocalClient("m", pipeline=pipe).complete("s", "p")

    def test_a_strange_output_becomes_an_llm_error(self):
        with pytest.raises(LLMError):
            HuggingFaceLocalClient("m", pipeline=lambda messages, **options: []).complete("s", "p")


class TestOllama:
    def test_it_posts_to_the_chat_endpoint(self):
        seen = {}

        def transport(url, body, headers, timeout):
            seen.update(url=url, body=body)
            return {"message": {"content": " hi "}}

        client = OllamaClient("http://localhost:11434/", "llama3.2", transport)
        assert client.complete("sys", "user", max_tokens=30, temperature=0.1) == "hi"
        assert seen["url"] == "http://localhost:11434/api/chat"
        assert seen["body"]["model"] == "llama3.2"
        assert seen["body"]["stream"] is False
        assert seen["body"]["options"] == {"temperature": 0.1, "num_predict": 30}

    def test_a_local_server_is_not_hosted(self):
        assert OllamaClient("http://localhost:11434", "m").hosted is False
        assert OllamaClient("http://192.168.1.5:11434", "m").hosted is True

    def test_a_strange_reply_becomes_an_llm_error(self):
        client = OllamaClient("http://localhost:11434", "m", lambda *args: {"oops": 1})
        with pytest.raises(LLMError):
            client.complete("s", "p")


class TestOpenAICompatible:
    def test_it_posts_with_a_bearer_key(self):
        seen = {}

        def transport(url, body, headers, timeout):
            seen.update(url=url, body=body, headers=headers)
            return {"choices": [{"message": {"content": "ok"}}]}

        client = OpenAICompatibleClient("https://api.example.com/v1/", "secret", "m", transport)
        assert client.complete("s", "p") == "ok"
        assert seen["url"] == "https://api.example.com/v1/chat/completions"
        assert seen["headers"] == {"Authorization": "Bearer secret"}
        assert seen["body"]["messages"][1]["content"] == "p"

    def test_no_key_means_no_header(self):
        seen = {}

        def transport(url, body, headers, timeout):
            seen["headers"] = headers
            return {"choices": [{"message": {"content": "ok"}}]}

        OpenAICompatibleClient("http://localhost:8000/v1", "", "m", transport).complete("s", "p")
        assert seen["headers"] == {}

    def test_hosting_depends_on_the_address(self):
        assert OpenAICompatibleClient("http://127.0.0.1:8000/v1", "", "m").hosted is False
        assert OpenAICompatibleClient("https://api.openai.com/v1", "k", "m").hosted is True

    def test_a_strange_reply_becomes_an_llm_error(self):
        client = OpenAICompatibleClient(
            "http://localhost/v1", "", "m", lambda *args: {"choices": []}
        )
        with pytest.raises(LLMError):
            client.complete("s", "p")


class QuietServer(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        received = json.loads(self.rfile.read(length))
        if self.path == "/fail":
            self.send_response(500)
            self.end_headers()
            return
        if self.path == "/broken":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"not json")
            return
        body = json.dumps({"echo": received, "agent": self.headers.get("X-Test")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            body = json.dumps(
                {"models": [{"name": "llama3.2:latest"}, {"name": "nomic-embed-text:latest"}]}
            ).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def server_url():
    server = HTTPServer(("127.0.0.1", 0), QuietServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


class TestRealTransport:
    def test_post_json_sends_and_reads_json(self, server_url):
        reply = post_json(f"{server_url}/ok", {"a": 1}, {"X-Test": "yes"}, timeout=5)
        assert reply == {"echo": {"a": 1}, "agent": "yes"}

    def test_an_error_status_becomes_an_llm_error(self, server_url):
        with pytest.raises(LLMError, match="500"):
            post_json(f"{server_url}/fail", {}, timeout=5)

    def test_a_reply_that_is_not_json_becomes_an_llm_error(self, server_url):
        with pytest.raises(LLMError):
            post_json(f"{server_url}/broken", {}, timeout=5)

    def test_an_unreachable_server_becomes_an_llm_error(self):
        with pytest.raises(LLMError):
            post_json("http://127.0.0.1:9/none", {}, timeout=1)

    def test_is_local_url(self):
        assert is_local_url("http://localhost:1234")
        assert is_local_url("http://127.0.0.1")
        assert not is_local_url("https://example.com")


class TestFactory:
    def test_the_mock_is_the_default_choice(self):
        assert isinstance(build_client(settings()), MockClient)

    def test_provider_names_are_not_case_sensitive(self):
        assert isinstance(build_client(settings(llm_provider=" MOCK ")), MockClient)

    def test_a_local_hugging_face_model_is_allowed_in_local_mode(self):
        client = build_client(settings(llm_provider="hf_local"))
        assert isinstance(client, HuggingFaceLocalClient)

    def test_ollama_on_this_machine_is_allowed_in_local_mode(self):
        assert isinstance(build_client(settings(llm_provider="ollama")), OllamaClient)

    def test_hosted_hugging_face_is_blocked_in_local_mode(self):
        with pytest.raises(PrivacyError, match="PRIVACY_MODE"):
            build_client(settings(llm_provider="hf_inference", hf_token="abc"))

    def test_hosted_hugging_face_works_when_remote_is_allowed(self):
        chosen = settings(
            llm_provider="hf_inference", hf_token="abc", privacy_mode="remote_allowed"
        )
        assert isinstance(build_client(chosen), HuggingFaceInferenceClient)

    def test_hosted_hugging_face_needs_a_token(self):
        with pytest.raises(LLMError, match="HF_TOKEN"):
            build_client(settings(llm_provider="hf_inference", privacy_mode="remote_allowed"))

    def test_a_hosted_openai_service_needs_a_key_and_permission(self):
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            build_client(settings(llm_provider="openai", privacy_mode="remote_allowed"))
        with pytest.raises(PrivacyError):
            build_client(settings(llm_provider="openai", openai_api_key="k"))
        chosen = settings(llm_provider="openai", openai_api_key="k", privacy_mode="remote_allowed")
        assert isinstance(build_client(chosen), OpenAICompatibleClient)

    def test_an_openai_style_server_on_this_machine_needs_no_key(self):
        chosen = settings(llm_provider="openai", openai_base_url="http://localhost:8000/v1")
        assert isinstance(build_client(chosen), OpenAICompatibleClient)

    def test_an_unknown_provider_lists_the_choices(self):
        with pytest.raises(LLMError, match="mock, hf_local"):
            build_client(settings(llm_provider="magic"))


class TestGetJson:
    def test_it_reads_json_from_a_real_server(self, server_url):
        reply = get_json(f"{server_url}/api/tags", timeout=5)
        assert [entry["name"] for entry in reply["models"]] == [
            "llama3.2:latest",
            "nomic-embed-text:latest",
        ]

    def test_an_error_status_becomes_an_llm_error(self, server_url):
        with pytest.raises(LLMError, match="404"):
            get_json(f"{server_url}/missing", timeout=5)

    def test_an_unreachable_server_becomes_an_llm_error(self):
        with pytest.raises(LLMError):
            get_json("http://127.0.0.1:9/none", timeout=1)


class TestDiscovery:
    def test_models_are_listed_from_a_real_server(self, server_url):
        assert installed_ollama_models(server_url) == ["llama3.2:latest", "nomic-embed-text:latest"]

    def test_a_server_that_is_not_running_gives_an_empty_list(self):
        assert installed_ollama_models("http://127.0.0.1:9") == []

    def test_a_reply_in_the_wrong_shape_gives_an_empty_list(self):
        assert installed_ollama_models("http://x", fetch=lambda url, timeout: {"models": 5}) == []
        assert installed_ollama_models("http://x", fetch=lambda url, timeout: ["a"]) == []

    def test_entries_without_a_name_are_skipped(self):
        reply = {"models": [{"name": "a:1"}, {"id": 2}, "text", {"name": 5}]}
        assert installed_ollama_models("http://x", fetch=lambda url, timeout: reply) == ["a:1"]

    def test_the_address_is_cleaned_before_asking(self):
        seen = []
        installed_ollama_models("http://x:1/", fetch=lambda url, timeout: seen.append(url) or {})
        assert seen == ["http://x:1/api/tags"]

    def test_embedding_models_are_not_chat_models(self):
        assert chat_models(["llama3.2:latest", "nomic-embed-text"]) == ["llama3.2:latest"]

    def test_the_wanted_model_is_chosen_when_installed(self):
        assert pick_model("qwen2.5:3b", ["llama3.2:latest", "qwen2.5:3b"]) == "qwen2.5:3b"

    def test_a_model_of_the_same_family_is_the_next_choice(self):
        assert pick_model("llama3.2", ["mistral:7b", "llama3.2:latest"]) == "llama3.2:latest"

    def test_otherwise_the_first_chat_model_is_used(self):
        assert pick_model("qwen2.5:3b", ["nomic-embed-text", "mistral:7b"]) == "mistral:7b"

    def test_nothing_is_chosen_when_nothing_is_installed(self):
        assert pick_model("llama3.2", []) is None
        assert pick_model("llama3.2", ["nomic-embed-text"]) is None


class TestAutoProvider:
    def test_a_running_local_model_is_used(self):
        chosen = settings(llm_provider="auto")
        client = build_client(chosen, discover=lambda url: ["llama3.2:latest"])
        assert isinstance(client, OllamaClient)
        assert client.name == "ollama:llama3.2:latest"

    def test_the_configured_model_wins_when_it_is_installed(self):
        chosen = settings(llm_provider="auto", ollama_model="qwen2.5:3b")
        client = build_client(chosen, discover=lambda url: ["llama3.2:latest", "qwen2.5:3b"])
        assert client.name == "ollama:qwen2.5:3b"

    def test_the_offline_model_is_the_fallback(self):
        client = build_client(settings(llm_provider="auto"), discover=lambda url: [])
        assert isinstance(client, MockClient)

    def test_embedding_only_installs_do_not_count(self):
        client = build_client(
            settings(llm_provider="auto"), discover=lambda url: ["nomic-embed-text"]
        )
        assert isinstance(client, MockClient)

    def test_the_address_from_the_settings_is_asked(self):
        asked = []
        chosen = settings(llm_provider="auto", ollama_base_url="http://localhost:9999")
        build_client(chosen, discover=lambda url: asked.append(url) or [])
        assert asked == ["http://localhost:9999"]

    def test_a_remote_server_is_still_blocked_in_local_mode(self):
        chosen = settings(llm_provider="auto", ollama_base_url="http://192.168.1.5:11434")
        with pytest.raises(PrivacyError):
            build_client(chosen, discover=lambda url: ["llama3.2:latest"])


class TestHuggingFaceLocalToken:
    def test_the_token_reaches_the_pipeline(self, monkeypatch):
        seen = {}

        def fake_pipeline(task, model, token=None):
            seen.update(task=task, model=model, token=token)
            return lambda messages, max_new_tokens, temperature, do_sample: [
                {"generated_text": [{"role": "assistant", "content": "Done."}]}
            ]

        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
        client = HuggingFaceLocalClient("org/model", token="secret")
        assert client.complete("system", "prompt") == "Done."
        assert seen == {"task": "text-generation", "model": "org/model", "token": "secret"}

    def test_an_empty_token_is_passed_as_none(self, monkeypatch):
        seen = {}

        def fake_pipeline(task, model, token=None):
            seen["token"] = token
            return lambda messages, max_new_tokens, temperature, do_sample: [
                {"generated_text": [{"role": "assistant", "content": "Done."}]}
            ]

        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
        HuggingFaceLocalClient("org/model").complete("s", "p")
        assert seen["token"] is None

    def test_a_missing_package_gives_a_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(LLMError, match="requirements-ml.txt"):
            HuggingFaceLocalClient("org/model").complete("s", "p")


class TestAutoWithHuggingFace:
    def pick(self, **options):
        chosen = settings(llm_provider="auto", hf_llm_model="org/model", hf_token="tok")
        defaults = {
            "discover": lambda url: [],
            "cached": lambda repo: True,
            "available": lambda: {"transformers": True},
        }
        defaults.update(options)
        return build_client(chosen, **defaults)

    def test_a_downloaded_model_is_used_when_ollama_is_not_running(self):
        client = self.pick()
        assert isinstance(client, HuggingFaceLocalClient)
        assert client.name == "hf_local:org/model"

    def test_ollama_comes_first(self):
        assert isinstance(self.pick(discover=lambda url: ["llama3.2:latest"]), OllamaClient)

    def test_a_model_that_is_not_downloaded_is_not_used(self):
        assert isinstance(self.pick(cached=lambda repo: False), MockClient)

    def test_without_the_package_the_offline_model_is_used(self):
        assert isinstance(self.pick(available=lambda: {"transformers": False}), MockClient)

    def test_the_chosen_model_name_is_checked(self):
        asked = []
        self.pick(cached=lambda repo: asked.append(repo) or False)
        assert asked == ["org/model"]

    def test_the_explicit_local_choice_passes_the_token_on(self):
        client = build_client(settings(llm_provider="hf_local", hf_token="tok"))
        assert client._token == "tok"
