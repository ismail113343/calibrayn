import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from calibrayn.config import ROOT
from calibrayn.evaluation.dataset import QAItem, load_dataset, save_dataset
from calibrayn.hub import cache, card, publish
from calibrayn.hub.cli import main
from calibrayn.hub.models import download_model
from calibrayn.hub.rows import HubError, default_loader, describe_rows, load_rows
from calibrayn.hub.status import hub_status


def make_cache(folder, repo="org/name", filename="config.json"):
    base = folder / f"models--{repo.replace('/', '--')}"
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "main").write_text("abc123", encoding="utf-8")
    snapshot = base / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / filename).write_text("{}", encoding="utf-8")


class TestCache:
    def test_a_downloaded_model_is_found(self, tmp_path):
        make_cache(tmp_path)
        assert cache.is_cached("org/name", cache_dir=tmp_path)

    def test_a_model_that_was_never_downloaded_is_not_found(self, tmp_path):
        assert not cache.is_cached("org/other", cache_dir=tmp_path)

    def test_a_missing_file_is_not_found(self, tmp_path):
        make_cache(tmp_path)
        assert not cache.is_cached("org/name", filename="missing.json", cache_dir=tmp_path)

    def test_installed_packages_are_reported_by_name(self):
        found = cache.packages_available()
        assert set(found) == {"transformers", "sentence_transformers", "datasets", "torch"}
        assert all(isinstance(value, bool) for value in found.values())


class TestRows:
    def test_rows_come_from_the_loader(self):
        seen = {}

        def loader(repo, config, split, limit):
            seen.update(repo=repo, config=config, split=split, limit=limit)
            return [{"a": 1}]

        assert load_rows("org/data", "sub", "train", 5, loader=loader) == [{"a": 1}]
        assert seen == {"repo": "org/data", "config": "sub", "split": "train", "limit": 5}

    def test_no_rows_is_an_error_naming_the_split(self):
        with pytest.raises(HubError, match="'test'"):
            load_rows("org/data", loader=lambda *args: [])

    def test_the_columns_are_described_with_short_examples(self):
        rows = [{"question": "Why?", "documents": ["a", "b"], "text": "x" * 300}]
        described = {entry["column"]: entry for entry in describe_rows(rows, width=20)}
        assert described["question"]["type"] == "str"
        assert described["documents"]["type"] == "list"
        assert described["text"]["example"].endswith("...")

    def test_the_default_loader_reads_a_stream_up_to_the_limit(self, monkeypatch):
        def fake(repo, config, split, streaming):
            assert streaming is True
            return iter([{"n": number} for number in range(10)])

        monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake))
        assert default_loader("org/data", None, "test", 3) == [{"n": 0}, {"n": 1}, {"n": 2}]

    def test_the_default_loader_without_a_limit_reads_everything(self, monkeypatch):
        fake = lambda repo, config, split, streaming: iter([{"n": 1}, {"n": 2}])  # noqa: E731
        monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake))
        assert len(default_loader("org/data", None, "test", None)) == 2

    def test_a_failure_while_reading_becomes_a_clear_error(self, monkeypatch):
        def fake(repo, config, split, streaming):
            raise ConnectionError("no network")

        monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake))
        with pytest.raises(HubError, match="Could not read org/data: no network"):
            default_loader("org/data", None, "test", 3)

    def test_a_missing_datasets_package_tells_you_what_to_install(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        with pytest.raises(HubError, match="requirements-ml.txt"):
            default_loader("org/data", None, "test", 3)


class TestDownload:
    def test_the_model_is_downloaded_with_the_token(self):
        calls = []

        def downloader(repo, token):
            calls.append((repo, token))
            return "/cache/model"

        assert download_model("org/model", "secret", downloader) == "/cache/model"
        assert calls == [("org/model", "secret")]

    def test_no_token_is_passed_as_none(self):
        calls = []
        download_model("org/model", "", lambda repo, token: calls.append(token) or "x")
        assert calls == [None]

    def test_a_failure_becomes_a_clear_error(self):
        def downloader(repo, token):
            raise OSError("offline")

        with pytest.raises(HubError, match="Could not download org/model: offline"):
            download_model("org/model", None, downloader)


def items(split=None, verified=0, total=6):
    made = []
    for number in range(1, total + 1):
        made.append(
            QAItem(
                number,
                f"Question {number}?",
                "An answer.",
                number,
                f"doc{number % 2}.txt",
                1,
                2 + number % 4,
                "model-x",
                kind="recall",
                verified=number <= verified,
                split=split(number) if split else "",
            )
        )
    return made


def front_matter(text):
    return yaml.safe_load(text.split("---")[1])


class TestCard:
    def test_the_front_matter_is_valid_and_complete(self):
        meta = front_matter(card.dataset_card(items(), "me/notes-qa", "cc-by-4.0", "Sam"))
        assert meta["license"] == "cc-by-4.0"
        assert meta["language"] == ["en"]
        assert meta["task_categories"] == ["question-answering"]
        assert meta["size_categories"] == ["n<1K"]
        assert meta["pretty_name"] == "notes-qa"

    def test_the_counts_are_filled_in(self):
        text = card.dataset_card(items(verified=2), "me/notes-qa", "mit", "Sam")
        assert "6 questions from 2 documents" in text
        assert "Checked by a person: 2 of 6" in text
        assert "model-x" in text

    def test_splits_become_data_files(self):
        data = items(split=lambda n: "validation" if n < 3 else "test")
        meta = front_matter(card.dataset_card(data, "me/notes-qa", "mit", "Sam"))
        files = meta["configs"][0]["data_files"]
        assert files == [
            {"split": "test", "path": "test.jsonl"},
            {"split": "validation", "path": "validation.jsonl"},
        ]

    def test_without_splits_there_is_no_file_list(self):
        assert "configs" not in front_matter(card.dataset_card(items(), "me/x", "mit", "Sam"))

    def test_the_citation_names_the_author_and_repository(self):
        text = card.dataset_card(items(), "me/notes-qa", "mit", "Sam Lee")
        assert "Sam Lee (" in text
        assert "me/notes-qa" in text

    @pytest.mark.parametrize("bad", ["", "CC BY", "mit license", "-x"])
    def test_a_badly_written_licence_is_refused(self, bad):
        with pytest.raises(HubError, match="licence"):
            card.dataset_card(items(), "me/x", bad, "Sam")

    def test_no_questions_are_refused(self):
        with pytest.raises(HubError):
            card.dataset_card([], "me/x", "mit", "Sam")

    @pytest.mark.parametrize(
        ("count", "band"), [(10, "n<1K"), (999, "n<1K"), (1000, "1K<n<10K"), (10000, "10K<n<100K")]
    )
    def test_the_size_band(self, count, band):
        assert card.size_band(count) == band


class FakeApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, **options):
        self.created.append((repo_id, options))

    def upload_folder(self, **options):
        folder = options["folder_path"]
        import os

        options["files"] = sorted(os.listdir(folder))
        readme = os.path.join(folder, "README.md")
        options["card"] = open(readme, encoding="utf-8").read() if os.path.exists(readme) else ""
        self.uploaded.append(options)


class TestPublishDataset:
    def test_it_refuses_without_confirmation(self):
        api = FakeApi()
        with pytest.raises(HubError, match="--confirm-rights"):
            publish.publish_dataset(api, "me/qa", items(), "mit", "Sam")
        assert api.created == []

    def test_a_private_dataset_with_one_file_is_created_by_default(self):
        api = FakeApi()
        url = publish.publish_dataset(api, "me/qa", items(), "mit", "Sam", rights_confirmed=True)
        assert url == "https://huggingface.co/datasets/me/qa"
        repo, options = api.created[0]
        assert repo == "me/qa"
        assert options == {"repo_type": "dataset", "private": True, "exist_ok": True}
        assert api.uploaded[0]["files"] == ["README.md", "questions.jsonl"]
        assert api.uploaded[0]["repo_type"] == "dataset"

    def test_a_public_dataset_can_be_requested(self):
        api = FakeApi()
        publish.publish_dataset(
            api, "me/qa", items(), "mit", "Sam", private=False, rights_confirmed=True
        )
        assert api.created[0][1]["private"] is False

    def test_each_split_gets_its_own_file(self):
        api = FakeApi()
        data = items(split=lambda n: "validation" if n < 3 else "test")
        publish.publish_dataset(api, "me/qa", data, "mit", "Sam", rights_confirmed=True)
        assert api.uploaded[0]["files"] == ["README.md", "test.jsonl", "validation.jsonl"]

    def test_the_card_travels_with_the_data(self):
        api = FakeApi()
        publish.publish_dataset(
            api,
            "me/qa",
            items(),
            "cc-by-4.0",
            "Sam",
            rights_confirmed=True,
            description="About it.",
        )
        assert "license: cc-by-4.0" in api.uploaded[0]["card"]
        assert "About it." in api.uploaded[0]["card"]

    def test_a_bad_licence_stops_before_anything_is_created(self):
        api = FakeApi()
        with pytest.raises(HubError):
            publish.publish_dataset(
                api, "me/qa", items(), "bad licence", "Sam", rights_confirmed=True
            )
        assert api.created == []


class TestSpace:
    def test_the_needed_files_are_staged(self, tmp_path):
        staged = publish.stage_space(ROOT, tmp_path)
        assert "Dockerfile" in staged
        assert "README.md" in staged
        assert "requirements.txt" in staged
        assert "src/calibrayn/ui/Home.py" in staged
        assert ".streamlit/config.toml" in staged
        assert "configs/settings.yaml" in staged
        assert not any("\\" in path for path in staged)
        assert not any("__pycache__" in path for path in staged)
        assert not any(path.endswith(".pyc") for path in staged)
        assert not any(path.startswith("tests") for path in staged)

    def test_the_space_card_selects_docker_and_the_right_port(self, tmp_path):
        publish.stage_space(ROOT, tmp_path)
        meta = front_matter((tmp_path / "README.md").read_text(encoding="utf-8"))
        assert meta["sdk"] == "docker"
        assert meta["app_port"] == 7860
        assert len(meta["short_description"]) <= 60

    def test_the_dockerfile_serves_the_app_on_the_right_port(self, tmp_path):
        publish.stage_space(ROOT, tmp_path)
        text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
        assert "EXPOSE 7860" in text
        assert "src/calibrayn/ui/Home.py" in text
        assert "--server.address=0.0.0.0" in text

    def test_a_missing_source_folder_is_reported(self, tmp_path):
        with pytest.raises(HubError, match="was not found"):
            publish.stage_space(tmp_path, tmp_path / "out")

    def test_a_missing_template_is_reported(self, tmp_path):
        for name in publish.SPACE_PATHS:
            source = ROOT / name
            target = tmp_path / name
            if source.is_dir():
                target.mkdir()
            else:
                target.write_text("x", encoding="utf-8")
        with pytest.raises(HubError, match="deploy"):
            publish.stage_space(tmp_path, tmp_path / "out")

    def test_deploying_creates_a_docker_space_with_its_settings(self, tmp_path):
        api = FakeApi()
        url = publish.deploy_space(api, "me/app", tmp_path, variables={"EMBEDDER": "hashing"})
        assert url == "https://huggingface.co/spaces/me/app"
        repo, options = api.created[0]
        assert repo == "me/app"
        assert options["repo_type"] == "space"
        assert options["space_sdk"] == "docker"
        assert options["private"] is False
        keys = {entry["key"]: entry["value"] for entry in options["space_variables"]}
        assert keys["PRIVACY_MODE"] == "remote_allowed"
        assert keys["LLM_PROVIDER"] == "hf_inference"
        assert keys["EMBEDDER"] == "hashing"
        assert api.uploaded[0]["repo_type"] == "space"

    def test_variables_can_be_overridden_and_the_space_made_private(self, tmp_path):
        api = FakeApi()
        publish.deploy_space(
            api, "me/app", tmp_path, variables={"LLM_PROVIDER": "openai"}, private=True
        )
        keys = {entry["key"]: entry["value"] for entry in api.created[0][1]["space_variables"]}
        assert keys["LLM_PROVIDER"] == "openai"
        assert api.created[0][1]["private"] is True


class TestHubCommands:
    def rows(self):
        return [
            {
                "question": "Who built long roads for armies?",
                "context": "The Roman empire built long roads for armies.",
            },
            {
                "question": "How do atoms share electrons?",
                "context": "Atoms bond by sharing electrons in covalent bonds.",
            },
            {
                "question": "What does chlorophyll absorb?",
                "context": "Chlorophyll absorbs light for photosynthesis.",
            },
        ]

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        values = {
            "LLM_PROVIDER": "mock",
            "EMBEDDER": "hashing",
            "INDEX": "numpy",
            "DATA_DIR": str(tmp_path / "data"),
            "HF_TOKEN": "token-123",
        }
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        return tmp_path

    def test_inspect_lists_the_columns(self, env, monkeypatch, capsys):
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: self.rows())
        assert main(["inspect", "org/data"]) == 0
        shown = capsys.readouterr().out
        assert "question (str)" in shown
        assert "context (str)" in shown

    def test_a_failure_to_read_gives_a_clear_error(self, env, monkeypatch, capsys):
        def broken(*args):
            raise HubError("Could not read org/data: offline")

        monkeypatch.setattr("calibrayn.hub.rows.default_loader", broken)
        assert main(["inspect", "org/data"]) == 1
        assert "offline" in capsys.readouterr().err

    def test_the_benchmark_scores_embeddings_against_keyword_search_and_chance(
        self, env, monkeypatch, capsys
    ):
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: self.rows())
        out = env / "reports"
        arguments = ["benchmark", "org/data", "--question-field", "question"]
        arguments += ["--context-field", "context", "--out", str(out)]
        assert main(arguments) == 0
        shown = capsys.readouterr().out
        assert "3 questions over 3 passages" in shown
        assert "keyword search (BM25)" in shown
        assert "random guessing" in shown
        assert "offline hashing embedder" in shown
        assert (out / "benchmark_org_data.csv").is_file()
        assert (out / "benchmark_org_data.md").is_file()

    def test_a_wrong_column_name_lists_the_real_ones(self, env, monkeypatch, capsys):
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: self.rows())
        arguments = ["benchmark", "org/data", "--question-field", "q", "--context-field", "context"]
        assert main(arguments) == 1
        assert "context, question" in capsys.readouterr().err

    def test_publishing_needs_the_rights_flag(self, env, monkeypatch, capsys):
        api = FakeApi()
        monkeypatch.setattr("calibrayn.hub.cli.make_api", lambda token: api)
        data = save_dataset(items(), env / "qa.jsonl")
        arguments = [
            "publish",
            "me/qa",
            "--dataset",
            str(data),
            "--license",
            "mit",
            "--author",
            "Sam",
        ]
        assert main(arguments) == 1
        assert "--confirm-rights" in capsys.readouterr().err
        assert api.created == []

    def test_publishing_with_the_flag_creates_a_private_dataset(self, env, monkeypatch, capsys):
        api = FakeApi()
        tokens = []
        monkeypatch.setattr("calibrayn.hub.cli.make_api", lambda token: tokens.append(token) or api)
        data = save_dataset(items(), env / "qa.jsonl")
        arguments = ["publish", "me/qa", "--dataset", str(data), "--license", "mit"]
        arguments += ["--author", "Sam", "--confirm-rights"]
        assert main(arguments) == 0
        assert tokens == ["token-123"]
        assert api.created[0][1]["private"] is True
        assert "private dataset" in capsys.readouterr().out

    def test_publishing_can_be_made_public(self, env, monkeypatch):
        api = FakeApi()
        monkeypatch.setattr("calibrayn.hub.cli.make_api", lambda token: api)
        data = save_dataset(items(), env / "qa.jsonl")
        arguments = ["publish", "me/qa", "--dataset", str(data), "--license", "mit"]
        arguments += ["--author", "Sam", "--confirm-rights", "--public"]
        assert main(arguments) == 0
        assert api.created[0][1]["private"] is False

    def test_deploying_uploads_the_staged_files(self, env, monkeypatch, capsys):
        api = FakeApi()
        monkeypatch.setattr("calibrayn.hub.cli.make_api", lambda token: api)
        assert main(["deploy", "me/app"]) == 0
        shown = capsys.readouterr().out
        assert "https://huggingface.co/spaces/me/app" in shown
        assert "HF_TOKEN" in shown
        assert api.created[0][1]["space_sdk"] == "docker"
        assert api.uploaded[0]["repo_type"] == "space"

    def test_download_fetches_both_models_by_default(self, env, monkeypatch, capsys):
        calls = []
        monkeypatch.setattr(
            "calibrayn.hub.cli.download_model", lambda repo, token: calls.append(repo) or "/saved"
        )
        assert main(["download"]) == 0
        assert calls == ["BAAI/bge-small-en-v1.5", "Qwen/Qwen2.5-1.5B-Instruct"]

    def test_download_can_fetch_one_model(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "calibrayn.hub.cli.download_model", lambda repo, token: calls.append(repo) or "/saved"
        )
        main(["download", "--embedding"])
        main(["download", "--llm"])
        assert calls == ["BAAI/bge-small-en-v1.5", "Qwen/Qwen2.5-1.5B-Instruct"]

    def test_a_step_is_required(self):
        with pytest.raises(SystemExit):
            main([])

    def test_the_dataset_file_written_for_publishing_can_be_reloaded(self, env):
        path = save_dataset(items(verified=3), env / "qa.jsonl")
        assert [entry.verified for entry in load_dataset(path)][:4] == [True, True, True, False]
        assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["verified"] is True


class TestHubStatus:
    def settings(self, token="secret"):
        return SimpleNamespace(hf_embedding_model="a/embed", hf_llm_model="c/chat", hf_token=token)

    def ready(self):
        return {"transformers": True, "sentence_transformers": True}

    def test_everything_ready(self):
        status = hub_status(self.settings(), cached=lambda repo: True, available=self.ready)
        assert status.packages_ready
        assert status.embedding_cached and status.llm_cached and status.token_set
        assert status.missing_packages == ()

    def test_each_model_is_checked_separately(self):
        status = hub_status(
            self.settings(), cached=lambda repo: repo == "a/embed", available=self.ready
        )
        assert status.embedding_cached
        assert not status.llm_cached

    def test_missing_packages_are_named(self):
        status = hub_status(
            self.settings(),
            cached=lambda repo: False,
            available=lambda: {"transformers": False, "sentence_transformers": True},
        )
        assert not status.packages_ready
        assert status.missing_packages == ("transformers",)

    def test_no_token_is_reported(self):
        status = hub_status(
            self.settings(token=""), cached=lambda repo: False, available=self.ready
        )
        assert not status.token_set
