# Models and the data behind them

Calibrayn trains no model. It uses two pre-trained models from the Hugging Face Hub, and rule-based
code for scheduling and monitoring. This page says what each part rests on and which benchmark
checks it.

## What is used

| Part | What it is | Evidence it rests on | Licence |
| --- | --- | --- | --- |
| Language model | Qwen/Qwen2.5-1.5B-Instruct, by the Qwen team | The model card points to the Qwen2.5 blog for results on public benchmarks | Apache 2.0, checked on the model card |
| Embedding model | BAAI/bge-small-en-v1.5, by BAAI | Its model card lists results on the MTEB benchmark tasks. Reference: Xiao et al. (2024), C-Pack, SIGIR | MIT, confirm on the model card |
| Scheduling | SM-2 (Wozniak, 1990) with a confidence adjustment | Published algorithm. The adjustment is this project's own and is tested by simulation | Own code |
| Monitors | ZPD tracker, cognitive load estimate, metacognition | Rule-based. No training data. Settings are documented assumptions | Own code |
| Optional hosted models | Hugging Face inference, DeepSeek or similar | Providers' own reports | Check each provider |

A note on licences: Qwen2.5-3B is released under the Qwen Research License, not Apache 2.0, while
the 1.5B and 7B models are Apache 2.0 (Qwen Team, 2024). The default here is the 1.5B model. If you
choose the 3B model through Ollama, say so in the report and check that your use fits its terms.

## Which benchmark checks which task

| Task | Benchmark data | Command | Notes |
| --- | --- | --- | --- |
| Finding the right notes | Any Hub dataset with a question column and a passage column. MTEB results for the embedding model are published | `python -m calibrayn.hub benchmark` | Reports embeddings, keyword search (BM25) and random guessing side by side |
| Judging whether an answer is supported by the notes | RAGBench (Friel et al., 2024) | `python -m calibrayn.hub groundedness` | Tests the grounding check as a detector of unfaithful answers |
| Grading a student's answer | SemEval-2013 Student Response Analysis: SciEntsBank and Beetle (Dzikovska et al., 2013), on the Hub as `nkazi/SciEntsBank` and `nkazi/Beetle`, CC BY 4.0. ASAG2024 combines seven short-answer datasets (Meyer, Breuer and Furst, 2024) | `python -m calibrayn.hub grader` | Reports the grader against keyword overlap and the majority label |
| Questions written from notes | No standard benchmark exists | review sheets and agreement | People check the questions. See docs/evaluation.md |
| Memory model in the simulation | None used | sensitivity runs | The numbers are assumptions. They could be fitted to a public spaced-repetition dataset such as the one in Settles and Meeder (2016), if it can be obtained |

## Testing the grader on SemEval-2013

Look at the data first:

```
python -m calibrayn.hub inspect nkazi/Beetle --split test_ua
```

Then grade the unseen-answers test set. The dataset card lists the default labels as 0 correct,
1 contradictory, 2 partially correct, 3 irrelevant and 4 non-domain, so only 0 counts as correct:

```
python -m calibrayn.hub grader nkazi/Beetle --split test_ua --question-field question --reference-field reference_answer --answer-field student_answer --label-field label --correct-labels 0 --limit 200
```

Repeat with `--split test_uq` (unseen questions) and with `nkazi/SciEntsBank`. Read the split names
from each dataset page. Use only the official test splits for reported numbers, and keep the
`--limit` the same across systems.

For context, a study of GPT-4 on these data reported F1 scores of about 0.74 on SciEntsBank and
0.61 on Beetle for the two-way task (arXiv 2309.09338). Its F1 may be defined differently from the
one here, so compare with care. A small local model should be expected to do worse.

## If a model is ever trained

Nothing is trained now. If a grader is fine-tuned later, train only on the official training split,
choose settings on a held-out part of it, report once on the official test splits, and record the
model and dataset versions. Never tune on a test split.
