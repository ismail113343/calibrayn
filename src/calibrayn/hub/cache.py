from importlib.util import find_spec

from huggingface_hub import try_to_load_from_cache

PACKAGES = ("transformers", "sentence_transformers", "datasets", "torch")


def is_cached(repo_id, filename="config.json", cache_dir=None):
    found = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
    return isinstance(found, str)


def packages_available():
    return {name: find_spec(name) is not None for name in PACKAGES}
