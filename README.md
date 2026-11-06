# Calibrayn: An Agentic Study Companion (ASC)

![CI](https://github.com/ismail113343/calibrayn/actions/workflows/ci.yml/badge.svg)

Calibrayn is a study companion that works on a student's own notes. It plans revision with the
SM-2 algorithm, adjusts question difficulty using estimates of cognitive load and the Zone of
Proximal Development, and checks whether the student's confidence matches their results.

## What it does

- Reads PDF, Word, PowerPoint, text and Markdown notes and splits them into topics.
- Asks questions made from the notes. Before answering, the student says how sure they are.
- Schedules each topic. A topic answered wrongly returns in 10 minutes.
- Lowers a topic's easiness when the student was sure and wrong, and asks for a recall check later.
- Explains topics with a worked example, and answers questions with sources from the notes.
- Shows progress, calibration and weak topics on a dashboard.

## Set up (Windows)

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
copy .env.example .env
```

## Run the app

```
streamlit run src/calibrayn/ui/Home.py
```

With the default settings the app uses an offline demo model, so it runs without any account or
download. The sidebar says when the demo model is in use.

## Use a real model

Edit `.env`:

| Goal | Settings |
| --- | --- |
| Hugging Face model on this computer | `LLM_PROVIDER=hf_local` and `EMBEDDER=huggingface` |
| Hugging Face hosted model | `LLM_PROVIDER=hf_inference`, `HF_TOKEN=...`, `PRIVACY_MODE=remote_allowed` |
| Ollama on this computer | `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=...` |
| Any server with the OpenAI chat format | `LLM_PROVIDER=openai` and the `OPENAI_` settings |

Local models and the Hugging Face embedder need `python -m pip install -r requirements-ml.txt`.
Hosted models send your notes to another server, so they are refused unless
`PRIVACY_MODE=remote_allowed`.

## Check the code

```
python -m pytest
ruff check .
ruff format --check .
```

## Layout

| Folder | Purpose |
| --- | --- |
| `src/calibrayn/scheduler` | SM-2 scheduling and the confidence-based easiness adjustment |
| `src/calibrayn/orchestrator` | ZPD tracker, cognitive load, metacognition, policy, topic selector, study agent |
| `src/calibrayn/ingestion` | File loaders, chunker and the upload pipeline |
| `src/calibrayn/rag` | Embeddings, search index, retriever, prompts, question writer, grounding check |
| `src/calibrayn/llm` | One interface to mock, Hugging Face, Ollama and OpenAI-style models |
| `src/calibrayn/storage` | SQLite database, library and progress |
| `src/calibrayn/services` | Grader, analytics and the workspace that wires everything together |
| `src/calibrayn/ui` | Streamlit pages |
| `configs` | `settings.yaml` with every tunable number, and the prompt files |
| `docs/decisions` | A short record for each design decision |
| `tests` | Unit and interface tests |

## Author

Muhammad Ismail Fiaz (2509837)

## License

MIT
