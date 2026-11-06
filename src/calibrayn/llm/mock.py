import json
import re

from calibrayn.llm.base import LLMError
from calibrayn.text import content_words, sentences, shorten


def block(prompt, name):
    pattern = rf"{re.escape(name)}:\n(.*?)\nEND {re.escape(name)}"
    match = re.search(pattern, prompt, re.S)
    return match.group(1).strip() if match else ""


class MockClient:
    name = "mock"
    hosted = False

    def complete(self, system, prompt, max_tokens=400, temperature=0.2):
        task = re.search(r"TASK:\s*(\w+)", system)
        if task is None:
            raise LLMError("The mock model needs a TASK line in the system prompt.")
        respond = {
            "question": self._question,
            "grade": self._grade,
            "explain": self._explain,
            "example": self._example,
            "answer": self._answer,
        }.get(task.group(1))
        if respond is None:
            raise LLMError(f"The mock model does not know the task '{task.group(1)}'.")
        return respond(prompt)

    def _question(self, prompt):
        passage = block(prompt, "PASSAGE")
        stem = " ".join(passage.split()[:5]) or "this topic"
        level = re.search(r"Difficulty level: (\d)", prompt)
        number = int(level.group(1)) if level else 3
        if number <= 2:
            text = f"What do your notes say about the part that starts with '{stem}'?"
        elif number == 3:
            text = f"Explain in your own words the part that starts with '{stem}'."
        else:
            text = f"Explain why the part that starts with '{stem}' matters, and how it connects."
        answer = " ".join(sentences(passage)[:3])
        return json.dumps({"question": text, "answer": answer})

    def _grade(self, prompt):
        key = set(content_words(block(prompt, "ANSWER KEY")))
        student = set(content_words(block(prompt, "STUDENT ANSWER")))
        if not key:
            return json.dumps({"score": 0.0, "feedback": "There is no answer key to compare."})
        overlap = len(key & student) / len(key)
        score = round(min(1.0, overlap * 1.5), 2)
        missing = sorted(key - student)[:4]
        if score >= 0.9:
            feedback = "Good. You covered the main ideas."
        elif missing:
            feedback = "Try to include: " + ", ".join(missing) + "."
        else:
            feedback = "Check your notes again."
        return json.dumps({"score": score, "feedback": feedback})

    def _explain(self, prompt):
        parts = sentences(block(prompt, "PASSAGE"))[:4]
        return "In short: " + " ".join(parts) if parts else "There is nothing to explain."

    def _example(self, prompt):
        parts = sentences(block(prompt, "PASSAGE"))[:4]
        steps = [f"Step {number}: {part}" for number, part in enumerate(parts, start=1)]
        return "Worked example from your notes.\n" + "\n".join(steps)

    def _answer(self, prompt):
        question = set(content_words(block(prompt, "QUESTION")))
        listed = re.findall(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", block(prompt, "PASSAGES"), re.S)
        best_number, best_text, best_score = None, "", 0
        for number, text in listed:
            overlap = len(question & set(content_words(text)))
            if overlap > best_score:
                best_number, best_text, best_score = number, text, overlap
        if best_number is None:
            return "I could not find this in your notes."
        return shorten(" ".join(sentences(best_text)[:2]), 400) + f" [{best_number}]"
