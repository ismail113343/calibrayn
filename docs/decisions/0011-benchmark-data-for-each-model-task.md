# 0011: Benchmark data for each model task

Date: 2026-11-06
Status: Accepted

## Context

The supervisor asked that the model use a benchmark dataset. The project trains no model. It uses
pre-trained models, so the benchmark question is where to show that each model-dependent task
works, on data that others recognise.

## Decision

- No model is trained or fine-tuned for now. The two pre-trained models are chosen for their
  published benchmark results and permissive licences, and the choice is documented in
  docs/models.md.
- Every task that depends on a model is checked on a recognised benchmark from the Hub, next to
  simple baselines:
  - finding notes: a question and passage dataset, against BM25 and chance
  - judging whether an answer is supported: RAGBench, as a detector with area under the ROC curve
  - grading student answers: SemEval-2013 SciEntsBank and Beetle, against keyword overlap and the
    majority label, using the official unseen-answer and unseen-question test splits
- The project's own question set stays for in-domain checks and is reviewed by people.
- Published numbers are quoted for context only, because metrics and settings differ.
- If a model is trained later, only the official training split is used for training, only a
  held-out part of it for choosing settings, and the official test splits once for reporting.

## Why

- A benchmark with baselines shows whether a result means anything. A score alone does not.
- Official splits make the numbers comparable with other work.

## To check

- The commands were tested with stand-ins for the Hub. Run each once on a real connection and read
  the column names and split names from the dataset page.
- A small local model may grade much worse than the published large-model results. Report the
  model used, the threshold and the number of answers graded.
- Qwen2.5-3B has a research licence. Prefer the 1.5B or 7B models if the licence matters.
