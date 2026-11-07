from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    title: str
    text: str
    done: bool


def steps(summary, model_name):
    return [
        Step(
            "Add your notes",
            "Open Upload and add a PDF, Word, PowerPoint, text or Markdown file. "
            "You can also try the sample notes.",
            summary.documents > 0,
        ),
        Step(
            "Connect a language model",
            "Open Setup to connect a real model. Until then the offline model writes simple "
            "questions from your own sentences.",
            model_name != "mock",
        ),
        Step(
            "Practise with the quiz",
            "Open Quiz and start a session. Say how sure you are before each answer.",
            summary.attempts > 0,
        ),
        Step(
            "Check your confidence",
            "After ten answers the Dashboard shows whether your confidence matches your results.",
            summary.attempts >= 10,
        ),
    ]


def next_step(summary, model_name):
    for step in steps(summary, model_name):
        if not step.done:
            return step
    if summary.due_now:
        return Step("Keep going", f"{summary.due_now} topics are due. Start a quiz.", False)
    return Step("All caught up", "Nothing is due right now. Come back later or study ahead.", True)
