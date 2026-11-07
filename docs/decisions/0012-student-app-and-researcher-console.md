# 0012: Student app and researcher console

Date: 2026-11-07
Status: Accepted

## Context

Calibrayn is a study companion for students. The first version of the interface also held model
setup and evaluation pages, and a long note about the offline model in the sidebar. A student has
no use for those, and they made the app look like a developer tool.

## Decision

- The student app has six screens: Home, Upload, Chat, Tutor, Quiz and Dashboard. It says nothing
  about models, embeddings, benchmarks or settings. When no AI model is connected it shows one
  short line saying that questions are simple.
- Model setup and evaluation are in a separate researcher console, started with its own command.
  It shows the technical details, the Hugging Face downloads and the benchmark tools.
- The command line tools stay as they are, so the console is a convenience and not the only way.
- The public Space runs the student app only, so the console controls are never exposed.
- A test lists the files in the student pages folder, so a developer page cannot slip back in.

## Why

- A product should show its users what they need and keep its own tooling out of the way.
- Keeping the console off the public Space avoids exposing downloads and token-using controls.

## To check

- The Home step about connecting a model now points to the README. A student who installs the
  app on their own computer is also the person who sets it up, so this step is a one-time task.
