import re
from collections import Counter
from datetime import date

from calibrayn.hub.rows import HubError

LICENSE_FORM = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")
FIELD_TEXT = (
    ("id", "Number of the question"),
    ("question", "The question"),
    ("answer", "The reference answer, taken from the notes"),
    ("evidence", "The sentence in the notes that supports the answer"),
    ("chunk_id", "The topic the question comes from"),
    ("document", "The source file"),
    ("page", "Page, slide or section where the topic starts"),
    ("level", "Difficulty level from 2 (recall) to 5 (apply)"),
    ("kind", "recall, explain, reason or apply"),
    ("answerable", "False if the notes do not contain the answer"),
    ("verified", "True once a person has checked the question"),
    ("reviewer", "Who checked it"),
    ("written_by", "The model that wrote the question"),
    ("split", "validation or test"),
)


def check_license(license_id):
    if not LICENSE_FORM.match(license_id or ""):
        raise HubError(
            "Give a licence such as cc-by-4.0, mit or cc-by-sa-4.0. "
            "Use 'other' only if the notes carry their own terms."
        )
    return license_id


def size_band(count):
    if count < 1000:
        return "n<1K"
    return "1K<n<10K" if count < 10000 else "10K<n<100K"


def dataset_card(items, repo_id, license_id, author, description=""):
    check_license(license_id)
    if not items:
        raise HubError("There are no questions to describe.")
    name = repo_id.split("/")[-1]
    splits = Counter(item.split for item in items if item.split)
    verified = sum(1 for item in items if item.verified)
    models = ", ".join(sorted({item.written_by for item in items}))
    files = sorted(splits) if splits else ["questions"]
    config_lines = "".join(
        f"  - split: {name_}\n    path: {name_}.jsonl\n" for name_ in files if name_ != "questions"
    )
    config_block = (
        f"configs:\n- config_name: default\n  data_files:\n{config_lines}" if config_lines else ""
    )
    levels = dict(sorted(Counter(item.level for item in items).items()))
    kinds = dict(sorted(Counter(item.kind for item in items).items()))
    documents = len({item.document for item in items})
    year = date.today().year
    fields = "\n".join(f"| `{field}` | {text} |" for field, text in FIELD_TEXT)
    split_text = (
        ", ".join(f"{count} {label}" for label, count in sorted(splits.items())) or "not split"
    )
    return f"""---
license: {license_id}
language:
- en
pretty_name: {name}
size_categories:
- {size_band(len(items))}
task_categories:
- question-answering
tags:
- education
- retrieval
- study-notes
{config_block}---

# {name}

{description or "Questions written from study notes, each linked to the topic it came from."}

## Summary

- {len(items)} questions from {documents} documents
- Difficulty levels: {levels}
- Kinds: {kinds}
- Splits: {split_text}
- Checked by a person: {verified} of {len(items)}

## Fields

| Field | Meaning |
| --- | --- |
{fields}

## How it was made

Questions were written from the notes by: {models}. Each one keeps the topic it came from, so the
data can be used to test whether a search system finds the right notes. Questions that a person
has not checked may be wrong, unclear or too easy.

## Intended use

Testing search and question answering over study notes, and studying how well retrieval works for
short educational documents.

## Limits

- The collection is small and comes from a limited set of documents.
- Questions share many words with their source, so they are easier than questions a student
  would ask in their own words.
- Do not use it to train a model for a different setting without checking that it fits.

## Licence and consent

The licence is {license_id}. The notes were shared by their owner, who confirmed the right to
publish them.

## Citation

{author} ({year}). {name}. Hugging Face dataset, {repo_id}.
"""
