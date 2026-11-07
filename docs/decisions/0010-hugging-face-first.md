# 0010: Hugging Face as the main platform

Date: 2026-11-06
Status: Accepted

## Context

The supervisor asked for Hugging Face to be used especially. The project already had a Hugging Face
model provider. This record makes Hugging Face the main path for models, embeddings, datasets and
deployment.

## Decision

Models and embeddings
- The default language model is Qwen/Qwen2.5-1.5B-Instruct and the default embedding model is
  BAAI/bge-small-en-v1.5, both run on the student's computer through the transformers and
  sentence-transformers packages.
- With LLM_PROVIDER=auto the app uses Ollama if it is running, then a Hugging Face model that has
  already been downloaded, then the offline model. EMBEDDER=auto uses the Hugging Face embedding
  model once it is downloaded, otherwise the offline hashing embedder.
- Nothing is downloaded silently. The Setup page of the researcher console and the hub download command do it on request,
  because the language model is several gigabytes.
- A Hugging Face token is read from HF_TOKEN and passed to every download and load, so private or
  gated models work. It is never written to a file by the app.
- Hosted inference stays available but is refused in the default privacy mode.

Datasets
- Public datasets are read from the Hub with the datasets package in streaming mode, so only the
  rows needed are fetched.
- Search is scored on any dataset that has a question column and a passage column, next to a
  keyword search (BM25) and chance. The tool shows the columns first, so nothing is guessed.
- RAGBench is not used for search. Its documents column holds everything given to the answering
  system, and its labels describe the answers. It is used to test the grounding check as a
  detector of unfaithful answers, reported as area under the ROC curve and accuracy.
- The student's own question set can be published as a dataset with a generated card. Publishing
  is private by default and refuses to run without an explicit confirmation of the right to share
  the notes, and a stated licence.

Deployment
- The app is deployed as a Docker Space on port 7860. The deploy command stages only the needed
  files and sets the Space variables. The Hugging Face token for hosted inference is added by the
  owner in the Space settings.

## Why

- It follows the supervisor's direction and keeps the privacy default: models run locally.
- Streaming, explicit downloads and explicit confirmation avoid surprises in size, cost and rights.

## To check

- The Hugging Face parts that need the network were tested with stand-ins for the Hub, not live.
  Run each command once on a real connection.
- The Space build, and the Streamlit flags that allow file upload inside the Space frame, have not
  been tested on Hugging Face.
- Dataset column names and licences must be read from each dataset page before use.
- Small local models write weaker questions than large hosted ones. Report the model used.
