# Running the evaluation

Everything here can be run from the app (Evaluation page) or from the command line.

## What the evaluation answers

1. Does search find the right notes? (question set, search scores, keyword and chance baselines)
2. Does the grounding check catch answers the notes do not support? (RAGBench)
   Does the grader mark student answers like a human does? (SemEval-2013 SciEntsBank and Beetle)
3. Does using confidence in the schedule help? (learner simulation, sensitivity)
4. Does it help a real student? (one informal test, separate)

## 1. Questions from the notes

```
python -m calibrayn.evaluation questions --count 100 --out evaluation/datasets/qa_100.jsonl
```

Add your notes in the app first. Use a real model, not the offline model. The tool says how many
different questions it could make.

## 2. Make the question set benchmark-grade

A set of model-written questions is a draft. These steps turn it into something you can defend.

```
python -m calibrayn.evaluation quality --dataset evaluation/datasets/qa_100.jsonl
python -m calibrayn.evaluation review-export --dataset evaluation/datasets/qa_100.jsonl --out evaluation/datasets/review_a.csv
```

A person opens the sheet and fills the `verdict` column with `keep`, `fix` or `reject`. For `fix`,
write the corrected text in the `*_fixed` columns. Then apply it:

```
python -m calibrayn.evaluation review-import --sheet evaluation/datasets/review_a.csv --reviewer "Your Name"
```

For agreement, give the same sheet to a second person and compare:

```
python -m calibrayn.evaluation agreement --a evaluation/datasets/review_a.csv --b evaluation/datasets/review_b.csv
```

Finally divide the questions into validation and test sets:

```
python -m calibrayn.evaluation split --validation-share 0.3
```

Targets worth aiming for:

- at least 10 documents and at least 100 questions, with a corpus much larger than the 10 results
  searched, so the scores can fail
- every question checked by a person, and 30 or more checked by a second person, with the kappa
  reported (about 0.6 or more is usually called good agreement)
- a mix of difficulty levels, and questions in the student's own words, not only the notes' words
- the `quality` check showing few very easy questions and no near-duplicates
- notes you are allowed to publish, with the licence and the date recorded

## 3. Search accuracy

```
python -m calibrayn.evaluation retrieval --dataset evaluation/datasets/qa_100.jsonl
```

Writes `retrieval.json`, `retrieval.md` and `retrieval_baselines.csv`. The baselines show a keyword
search (BM25) and random guessing next to your embeddings. A system is only worth reporting if it
clearly beats both. With 10 topics or fewer every system looks good, and the tool says so.

## 4. Public benchmarks for the models

See docs/huggingface.md and docs/models.md: `inspect`, `benchmark` (search), `groundedness`
(RAGBench) and `grader` (SemEval-2013 SciEntsBank and Beetle). Each is scored next to simple
baselines.

## 5. Learner simulation

```
python -m calibrayn.evaluation simulate --seeds 20 --days 60 --topics 40
```

Writes `persona_comparison.csv` and `.md`, and `retention_curves.csv` and `.md`. Each row compares
plain SM-2 with the confidence-adjusted schedule. A positive difference means the adjusted
schedule kept more. Low and high are the 95 percent range of that difference.

## 6. Sensitivity

```
python -m calibrayn.evaluation sensitivity --persona overconfident --parameter confident_error_penalty --values 0 0.1 0.25 0.4
```

Parameters: `confident_error_penalty`, `persistent_overconfidence_extra`,
`overconfidence_threshold`, `delayed_recall_minutes`.

## Reading the results honestly

- The learners and their memory are assumptions, not measurements of real students.
- A small difference is a valid result. Report it as it is.
- Model-written questions can make search look easier than it is. Say so, and say how many were
  checked by a person.
- The student test is separate. Ask for consent, store no personal data, and compare a single
  mental effort rating with the cognitive load estimate.
