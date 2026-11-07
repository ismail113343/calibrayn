import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def use_token(token):
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
