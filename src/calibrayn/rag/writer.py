import json
import re
from dataclasses import dataclass

from calibrayn.models.learning import Support
from calibrayn.text import shorten

LEVEL_HINTS = {
    1: "Ask about one basic fact or definition.",
    2: "Ask about one idea and what it means.",
    3: "Ask the student to explain an idea in their own words.",
    4: "Ask the student to explain why something happens or how two ideas connect.",
    5: "Ask the student to apply the idea to a new case or to compare two ideas.",
}

STYLES = {
    Support.DIRECT_RECALL: "Ask for direct recall. Give no hints.",
    Support.WORKED_EXAMPLE: "Ask a short question that matches the idea in the worked example.",
}


@dataclass(frozen=True)
class GeneratedQuestion:
    text: str
    answer_key: str


def parse_json_reply(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?|```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        loaded = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


class Writer:
    def __init__(self, client, prompts, max_tokens=400, temperature=0.2):
        self.client = client
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _ask(self, name, **fields):
        system, user = self.prompts.render(name, **fields)
        return self.client.complete(system, user, self.max_tokens, self.temperature)

    def question(self, chunk, level, support):
        reply = self._ask(
            "question",
            level=level,
            level_hint=LEVEL_HINTS.get(level, LEVEL_HINTS[3]),
            style=STYLES[support],
            passage=chunk.text,
        )
        fallback_key = shorten(chunk.text, 400)
        parsed = parse_json_reply(reply)
        if parsed and str(parsed.get("question", "")).strip():
            key = str(parsed.get("answer", "")).strip() or fallback_key
            return GeneratedQuestion(str(parsed["question"]).strip(), key)
        lines = [line.strip() for line in reply.splitlines() if line.strip()]
        text = lines[0] if lines else f"What do your notes say about {chunk.title}?"
        return GeneratedQuestion(text, fallback_key)

    def explanation(self, chunk, related):
        listed = "\n".join(f"- {shorten(passage.chunk.text, 300)}" for passage in related) or "None"
        return self._ask("explain", passage=chunk.text, related=listed)

    def worked_example(self, chunk):
        return self._ask("example", passage=chunk.text)

    def chat_answer(self, query, passages):
        numbered = "\n".join(
            f"[{number}] {passage.chunk.text}" for number, passage in enumerate(passages, 1)
        )
        return self._ask("chat", question=query, passages=numbered)
