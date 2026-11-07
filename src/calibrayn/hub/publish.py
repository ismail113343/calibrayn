import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from calibrayn.hub.card import dataset_card
from calibrayn.hub.rows import HubError

DATASET_URL = "https://huggingface.co/datasets/"
SPACE_URL = "https://huggingface.co/spaces/"
SPACE_PATHS = ("src", "configs", ".streamlit", "requirements.txt", "pyproject.toml", "LICENSE")
SKIPPED = shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc", ".pytest_cache")
SPACE_DEFAULTS = {
    "PRIVACY_MODE": "remote_allowed",
    "LLM_PROVIDER": "hf_inference",
    "EMBEDDER": "hashing",
    "DATA_DIR": "/tmp/calibrayn",
}


def write_json_lines(items, path):
    lines = [json.dumps(asdict(item), ensure_ascii=False) for item in items]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_dataset(
    api, repo_id, items, license_id, author, private=True, rights_confirmed=False, description=""
):
    if not rights_confirmed:
        raise HubError(
            "Publishing shares the questions and the sentences taken from your notes. "
            "Confirm that you may share them by adding --confirm-rights."
        )
    card = dataset_card(items, repo_id, license_id, author, description)
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        (base / "README.md").write_text(card, encoding="utf-8")
        names = sorted({item.split for item in items if item.split})
        if names:
            for name in names:
                write_json_lines(
                    [item for item in items if item.split == name], base / f"{name}.jsonl"
                )
        else:
            write_json_lines(items, base / "questions.jsonl")
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(base),
            commit_message="Add the question set and its card",
        )
    return DATASET_URL + repo_id


def stage_space(root, target):
    root = Path(root)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name in SPACE_PATHS:
        source = root / name
        if source.is_dir():
            shutil.copytree(source, target / name, ignore=SKIPPED, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, target / name)
        else:
            raise HubError(f"{source} is needed for the Space but was not found.")
    for name in ("Dockerfile", "README.md"):
        template = root / "deploy" / "huggingface" / name
        if not template.is_file():
            raise HubError(f"{template} is needed for the Space but was not found.")
        shutil.copy2(template, target / name)
    return sorted(
        path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
    )


def deploy_space(api, repo_id, staging, variables=None, private=False):
    chosen = {**SPACE_DEFAULTS, **(variables or {})}
    api.create_repo(
        repo_id,
        repo_type="space",
        space_sdk="docker",
        private=private,
        exist_ok=True,
        space_variables=[{"key": key, "value": value} for key, value in chosen.items()],
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(staging),
        commit_message="Deploy Calibrayn",
    )
    return SPACE_URL + repo_id
