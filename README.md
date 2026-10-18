# Calibrayn: An Agentic Study Companion (ASC)

![CI](https://github.com/ismail113343/calibrayn/actions/workflows/ci.yml/badge.svg)

A cognitively-grounded agentic AI study tool. Students upload their own notes; Calibrayn
schedules revision with SM-2, adapts difficulty using Cognitive Load and Zone of Proximal
Development estimates, and tracks how accurately the student judges their own understanding
(Metacognition Module).

**Status:** early development.

## Setup (Windows)

py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
copy .env.example .env
pytest


For the RAG pipeline and local models: `pip install -r requirements-ml.txt`.

## Layout
`src/calibrayn/` holds `ingestion`, `rag`, `llm`, `orchestrator`, `scheduler`, `services`,
`storage` and `ui`. Evaluation code lives in `evaluation/`.

## Author
Muhammad Ismail Fiaz (2509837)

## License
MIT
