# Architecture

## Layers

```mermaid
flowchart TD
    pages["Student app: Upload, Chat, Tutor, Quiz, Dashboard"]
    console["Researcher console: Setup, Evaluation"]
    workspace["Workspace: wires everything together"]
    agent["Study agent: plans each question and handles each answer"]
    monitors["Monitors: ZPD tracker, cognitive load, metacognition"]
    sched["Scheduler: SM-2 and the confidence adjustment"]
    pipeline["Notes pipeline: loaders, chunker, embeddings, search index, retriever"]
    writer["Writer and grader"]
    gateway["Model gateway: offline, Hugging Face, Ollama, hosted"]
    store["SQLite: notes, schedules, every answer"]

    pages --> workspace
    console --> workspace
    workspace --> agent
    workspace --> pipeline
    agent --> monitors
    agent --> sched
    agent --> writer
    writer --> gateway
    pipeline --> store
    agent --> store
```

## One question

```mermaid
flowchart LR
    pick["Pick the topic"] --> read["Read the monitors"]
    read --> plan["Set level and help"]
    plan --> write["Write the question from the notes"]
    write --> rate["Student rates confidence"]
    rate --> mark["Answer is marked"]
    mark --> update["Schedule and monitors update"]
    update --> recall{"Sure but wrong while overconfident?"}
    recall -- yes --> queue["Queue a recall check with no help"]
    recall -- no --> pick
    queue --> pick
```

## Where things live

| Folder | Purpose |
| --- | --- |
| `src/calibrayn/scheduler` | SM-2 and the confidence-based easiness adjustment |
| `src/calibrayn/orchestrator` | ZPD tracker, cognitive load, metacognition, policy, topic selector, study agent |
| `src/calibrayn/ingestion` | File loaders, chunker and the upload pipeline |
| `src/calibrayn/rag` | Embeddings, search index, retriever, prompts, writer, grounding check |
| `src/calibrayn/llm` | One interface to every model, plus detection of a local model |
| `src/calibrayn/storage` | SQLite database, library and progress |
| `src/calibrayn/services` | Grader, analytics and the workspace |
| `src/calibrayn/evaluation` | Question dataset, review and quality tools, search scores with baselines, grounding test, learner simulation, reports and command line |
| `src/calibrayn/hub` | Hugging Face: dataset reading, model download, dataset cards, publishing, Space deployment and command line |
| `deploy/huggingface` | The Dockerfile and card used for the Space |
| `src/calibrayn/ui` | The student app: Streamlit pages, palette, guidance and sample notes |
| `src/calibrayn/ui/console` | The researcher console: model setup and evaluation pages |
| `configs` | Every tunable number and the prompt files |
| `docs/decisions` | One short record per design decision |
