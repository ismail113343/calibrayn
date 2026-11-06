from dataclasses import dataclass

from calibrayn.models.learning import Support
from calibrayn.orchestrator.cognitive_load import LoadLevel
from calibrayn.orchestrator.metacognition import Intervention
from calibrayn.orchestrator.zpd import Zone


@dataclass(frozen=True)
class AgentConfig:
    correct_from: float = 0.6
    delayed_recall_minutes: int = 30
    break_session_signal: float = 0.5
    guessing_grade_cap: int = 3

    def __post_init__(self) -> None:
        if not 0.4 < self.correct_from <= 0.75:
            raise ValueError("correct_from must be above 0.4 and at most 0.75")
        if self.delayed_recall_minutes < 1:
            raise ValueError("delayed_recall_minutes must be at least 1")
        if not 0.0 <= self.break_session_signal <= 1.0:
            raise ValueError("break_session_signal must be between 0 and 1")
        if not 0 <= self.guessing_grade_cap <= 5:
            raise ValueError("guessing_grade_cap must be between 0 and 5")


DEFAULT_CONFIG = AgentConfig()


@dataclass(frozen=True)
class Plan:
    level: int
    support: Support
    delayed_recall: bool
    suggest_break: bool
    reasons: tuple[str, ...]


def make_plan(*, level, zone, load, intervention, recall_due, max_level, config=DEFAULT_CONFIG):
    reasons = []
    support = Support.DIRECT_RECALL
    next_level = level

    if load.level is LoadLevel.HIGH:
        support = Support.WORKED_EXAMPLE
        reasons.append("Your recent work looks demanding, so a worked example comes first.")
    elif zone is Zone.TOO_HARD:
        support = Support.WORKED_EXAMPLE
        reasons.append("Your recent answers were hard, so a worked example comes first.")

    if intervention is Intervention.REDUCE_SUPPORT and load.level is not LoadLevel.HIGH:
        support = Support.DIRECT_RECALL
        next_level = min(level + 1, max_level)
        reasons.append(
            "You were right more often than you felt, so there is less help and a harder question."
        )

    if recall_due:
        support = Support.DIRECT_RECALL
        reasons.append("You felt sure about a topic you got wrong, so it is back now with no help.")

    session_signal = load.session_signal or 0.0
    suggest_break = load.level is LoadLevel.HIGH and session_signal >= config.break_session_signal
    if suggest_break:
        reasons.append(
            "You have worked for a while and it is getting hard. A short break may help."
        )

    if not reasons:
        reasons.append("Your results are steady, so the plan stays the same.")
    return Plan(next_level, support, recall_due, suggest_break, tuple(reasons))
