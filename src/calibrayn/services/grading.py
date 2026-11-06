from dataclasses import dataclass

from calibrayn.llm.base import LLMError
from calibrayn.rag.writer import parse_json_reply
from calibrayn.text import content_words


@dataclass(frozen=True)
class Verdict:
    score: float
    feedback: str
    marked_by: str


def read_score(parsed):
    if not parsed:
        return None
    value = parsed.get("score")
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if not isinstance(value, int | float) or value != value:
        return None
    return max(0.0, min(1.0, float(value)))


def keyword_verdict(answer_key, student_answer):
    key = set(content_words(answer_key))
    student = set(content_words(student_answer))
    if not key:
        return Verdict(0.0, "There was no answer key to compare with.", "keywords")
    overlap = len(key & student) / len(key)
    score = round(min(1.0, overlap * 1.5), 2)
    missing = sorted(key - student)[:4]
    if missing and score < 0.9:
        feedback = "The model was not available, so key words were compared. Try to include: "
        feedback += ", ".join(missing) + "."
    else:
        feedback = "The model was not available, so key words were compared. Looks complete."
    return Verdict(score, feedback, "keywords")


class Grader:
    def __init__(self, client, prompts, max_tokens=300):
        self.client = client
        self.prompts = prompts
        self.max_tokens = max_tokens

    def grade(self, question, answer_key, student_answer, chunk):
        if not student_answer.strip():
            return Verdict(0.0, "No answer was given.", "rule")
        system, user = self.prompts.render(
            "grading",
            question=question,
            answer_key=answer_key,
            student_answer=student_answer,
            passage=chunk.text,
        )
        try:
            reply = self.client.complete(system, user, self.max_tokens, 0.0)
        except LLMError:
            return keyword_verdict(answer_key, student_answer)
        parsed = parse_json_reply(reply)
        score = read_score(parsed)
        if score is None:
            return keyword_verdict(answer_key, student_answer)
        feedback = str(parsed.get("feedback", "")).strip() or "Marked against your notes."
        return Verdict(score, feedback, "model")
