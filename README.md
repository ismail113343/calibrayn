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
- Tests itself: reviewed question sets, search accuracy against keyword and chance baselines, a
  grounding check test on RAGBench, and simulated learners, from the app or the command line.

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

The Home page shows what to do next. Open **Upload** and add notes, or use the sample biology
notes. Open **Quiz** to practise.

## Connect a language model

The app looks for a model by itself, in this order: Ollama if it is running, then a Hugging Face
model that has been downloaded, then an offline model that only builds simple questions from your
own sentences. The sidebar says which is in use. The **Setup** page shows what is connected, lets
you download Hugging Face models, gives the exact commands, and tests the connection.

| Goal | Settings in `.env` |
| --- | --- |
| Hugging Face model on this computer | `python -m pip install -r requirements-ml.txt`, then `python -m calibrayn.hub download` |
| A model on this computer through Ollama | `LLM_PROVIDER=auto` (the default), then `ollama pull qwen2.5:3b` |
| Hugging Face hosted model | `LLM_PROVIDER=hf_inference`, `HF_TOKEN=...`, `PRIVACY_MODE=remote_allowed` |
| Any server with the OpenAI chat format, such as DeepSeek | `LLM_PROVIDER=openai` and the `OPENAI_` settings |

Hosted models send your notes to another server, so they are refused unless
`PRIVACY_MODE=remote_allowed`.

## Hugging Face

Hugging Face is the main platform: models and embeddings run from the Hub, public datasets are
scored, your own question set can be published as a dataset with a card, and the app can be
deployed as a Space. See [docs/huggingface.md](docs/huggingface.md). No model is trained here: the
pre-trained models and the benchmarks that check each task are described in
[docs/models.md](docs/models.md).

```
python -m calibrayn.hub download
python -m calibrayn.hub inspect organisation/dataset
python -m calibrayn.hub benchmark organisation/dataset --question-field question --context-field context
python -m calibrayn.hub grader nkazi/Beetle --split test_ua --question-field question --reference-field reference_answer --answer-field student_answer --label-field label --correct-labels 0
python -m calibrayn.hub deploy yourname/calibrayn
```

## Evaluate

Open the **Evaluation** page, or use the command line. See [docs/evaluation.md](docs/evaluation.md).

```
python -m calibrayn.evaluation questions --count 100
python -m calibrayn.evaluation quality
python -m calibrayn.evaluation retrieval
python -m calibrayn.evaluation simulate --seeds 20
```

## Check the code

```
python -m pytest
ruff check .
ruff format --check .
```

## Layout

See [docs/architecture.md](docs/architecture.md) for the diagrams and the folder guide.
Every design choice has a short record in `docs/decisions`.

## Colours

The interface uses one palette, defined in `src/calibrayn/ui/components/palette.py` and in
`.streamlit/config.toml`. A test keeps the two in step.

| Colour | Code | Use |
| --- | --- | --- |
| Deep blue | `#065A82` | Main colour, links, first chart line |
| Navy | `#21295C` | Headings in slides and documents |
| Teal | `#1C7293` | Secondary charts |
| Green | `#2F9E6F` | Success, calibration chart |
| Amber | `#E8A33D` | Warnings, second chart line |
| Red | `#C8553D` | Errors |
| Ink | `#1B2340` | Text |
| Mist | `#E8EEF6` | Panels |
| Paper | `#F7F9FC` | Background |

## Author

Muhammad Ismail Fiaz (2509837)

## License

MIT
