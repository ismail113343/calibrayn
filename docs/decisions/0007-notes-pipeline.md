# 0007: Notes pipeline

Date: 2026-11-03
Status: Accepted

## Context

Sections 3.4 and 3.5 of the proposal describe loading notes, splitting them into chunks of about
500 words, embedding them and retrieving passages for each question.

## Decision

- Supported files: PDF, DOCX, PPTX, TXT and MD. Word and PowerPoint text is read straight from the
  file, so no extra package is needed.
- Text is split at sentence ends into chunks of about 500 words with 50 words of overlap. A chunk
  records the page, slide or section where its new text begins, and that is what citations show.
- Scanned PDFs are refused with a clear message. Reading them needs OCR, which is not built.
- The default embedder is a small hashing embedder that works offline. The Hugging Face embedder
  (BAAI/bge-small-en-v1.5 by default) is chosen with EMBEDDER=huggingface.
- Search is exact cosine similarity over a matrix. FAISS can be chosen with INDEX=faiss and gives
  the same results.
- Chat answers are checked against the retrieved text. If too few of the answer's content words
  appear in the sources, the interface warns the student.
- Prompts live in configs/prompts as text files. Models are asked for JSON where a structure is
  needed, and a reply that is not valid JSON is handled with a fallback.

## Why

- Offline defaults mean the app and the tests run anywhere.
- Exact search is fast enough for the size of one student's notes.

## To check

- The grounding check is a word overlap test. It catches invented content but not subtle errors.
- Retrieval quality has to be measured with the 100-question dataset using the real embedder.
