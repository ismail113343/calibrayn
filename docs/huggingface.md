# Using Hugging Face

| Part of the project | How Hugging Face is used |
| --- | --- |
| Language model | A model from the Hub runs on your computer (`hf_local`), or through hosted inference (`hf_inference`) |
| Embeddings | BAAI/bge-small-en-v1.5 through sentence-transformers |
| Evaluation data | Public datasets are read from the Hub and used to score search, the grounding check and the answer grader |
| Your dataset | Published to the Hub with a dataset card |
| Demo | Deployed as a Docker Space |

## 1. Models on your computer

```
python -m pip install -r requirements-ml.txt
python -m calibrayn.hub download
```

This downloads the embedding model and the language model chosen in `.env`
(`HF_EMBEDDING_MODEL`, `HF_LLM_MODEL`). Use `--embedding` or `--llm` for one of them. The language
model is several gigabytes, so check its page first. The Setup page has the same two buttons.

With `LLM_PROVIDER=auto` and `EMBEDDER=auto` (the defaults) the app uses the downloaded models the
next time it starts. For a private or gated model, put your token in `.env` as `HF_TOKEN=`.

## 2. Score search on a public dataset

Look at the columns first:

```
python -m calibrayn.hub inspect organisation/dataset --split test
```

Then score search, naming the question column and the passage column you saw:

```
python -m calibrayn.hub benchmark organisation/dataset --question-field question --context-field context --limit 500
```

Each row's passage is treated as the right answer, and the corpus is all passages. The result
shows your embeddings, a keyword search (BM25) and random guessing side by side. Report all three,
and use `EMBEDDER=huggingface` so the embeddings are a real model.

This fits datasets of the question and context kind. It does not fit RAGBench, see below.

## 3. Test the grounding check on RAGBench

RAGBench (Friel, Belyi and Sanyal, 2024) labels whether an answer is faithful to its documents.
That is what the app's grounding check tries to do, so it can be tested directly:

```
python -m calibrayn.hub inspect rungalileo/ragbench --config <configuration> --split test
python -m calibrayn.hub groundedness rungalileo/ragbench --config <configuration> --response-field <response column> --documents-field <documents column> --label-field <faithful label column>
```

Take the configuration and column names from the dataset page and the `inspect` output. The tool
reports the area under the ROC curve (0.5 is chance), the accuracy at the app's threshold and the
best accuracy. The grounding check compares words, so expect it to catch invented content but miss
subtle errors. Report that honestly.

## 4. Test the answer grader

The grader marks a student's answer against a reference answer. The standard benchmark for that is
SemEval-2013 Student Response Analysis (SciEntsBank and Beetle), which is on the Hub:

```
python -m calibrayn.hub inspect nkazi/Beetle --split test_ua
python -m calibrayn.hub grader nkazi/Beetle --split test_ua --question-field question --reference-field reference_answer --answer-field student_answer --label-field label --correct-labels 0 --limit 200
```

The result compares the grader with keyword overlap and with always guessing the majority label.
Use the official test splits (`test_ua` for unseen answers, `test_uq` for unseen questions). For a
dataset with a numeric grade, use `--score-field` and `--max-score` instead of the label options.
See docs/models.md for the details.

## 5. Publish your question set

Review the questions first (see docs/evaluation.md). Then:

```
python -m calibrayn.hub publish yourname/notes-qa --dataset evaluation/datasets/qa_100.jsonl --license cc-by-4.0 --author "Your Name" --confirm-rights
```

The dataset is private by default. Add `--public` only when you are sure. The command refuses to
run without `--confirm-rights`, because the questions and evidence sentences come from your notes.
Check that you may share them and choose a licence that matches. A dataset card is generated and
uploaded with the data. Set `HF_TOKEN` to a token with write access.

## 6. Deploy a Space

```
python -m calibrayn.hub deploy yourname/calibrayn
```

This creates a Docker Space, uploads the app and sets its variables. Then open the Space settings
and add `HF_TOKEN` as a secret if the app should use hosted inference. The first build takes a few
minutes. The demo sends text to a hosted model, so tell people to use the sample notes only.

## What has not been tested live

The network steps were tested with stand-ins for the Hub. Run each command once on a real
connection. The Space build has not been tested on Hugging Face.
