# 0006: Model gateway and privacy mode

Date: 2026-11-03
Status: Accepted

## Context

The proposal requires that notes stay on the student's machine (section 3.7) and allows several
model sources. Hugging Face is added as a provider.

## Decision

- All code talks to one small interface: complete(system, prompt, max_tokens, temperature).
- Providers are mock, hf_local, hf_inference, ollama and openai. The last one means any server
  that speaks the same chat format.
- A provider that sends text off the computer is called hosted. In the default privacy mode
  (local) a hosted provider is refused when the app starts. PRIVACY_MODE=remote_allowed lifts
  this for demos.
- Servers on localhost are treated as local.
- The mock provider needs no network. It builds simple questions and marks answers by comparing
  key words. The interface shows a warning whenever it is in use.
- If the model fails while marking an answer, the grader falls back to comparing key words and
  says so on screen.
- Network and model libraries are loaded only when first used, so the app starts fast.

## Why

- One interface lets the evaluation compare models by changing one setting.
- Refusing hosted models by default makes the privacy promise something the code enforces.

## To check

- Small local models may write weaker questions. The retrieval and grading tests use the mock, so
  real model quality has to be judged during the evaluation.
