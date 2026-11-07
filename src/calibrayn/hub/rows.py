from calibrayn.text import shorten


class HubError(Exception):
    pass


def default_loader(repo, config, split, limit):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise HubError(
            "Reading Hugging Face datasets needs the datasets package. "
            "Run: pip install -r requirements-ml.txt"
        ) from error
    try:
        stream = load_dataset(repo, config, split=split, streaming=True)
        rows = []
        for row in stream:
            rows.append(dict(row))
            if limit and len(rows) >= limit:
                break
    except Exception as error:
        raise HubError(f"Could not read {repo}: {error}") from error
    return rows


def load_rows(repo, config=None, split="test", limit=None, loader=None):
    rows = (loader or default_loader)(repo, config, split, limit)
    if not rows:
        raise HubError(f"{repo} returned no rows for the split '{split}'.")
    return rows


def describe_rows(rows, width=80):
    first = rows[0]
    return [
        {
            "column": name,
            "type": type(value).__name__,
            "example": shorten(str(value), width),
        }
        for name, value in first.items()
    ]
