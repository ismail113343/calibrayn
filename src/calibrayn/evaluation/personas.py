from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    summary: str
    study_days: tuple[int, ...]
    reviews_per_session: int
    minutes_per_review: float
    memory_gain: float
    starting_half_life: float
    new_topic_recall: float
    confidence_bias: float
    confidence_noise: float

    def __post_init__(self) -> None:
        if not self.study_days or any(day not in range(7) for day in self.study_days):
            raise ValueError("study_days must list weekdays from 0 to 6")
        if self.reviews_per_session < 1:
            raise ValueError("reviews_per_session must be at least 1")
        if self.minutes_per_review <= 0:
            raise ValueError("minutes_per_review must be positive")
        if self.memory_gain <= 0 or self.starting_half_life <= 0:
            raise ValueError("memory_gain and starting_half_life must be positive")
        if not 0.0 <= self.new_topic_recall <= 1.0:
            raise ValueError("new_topic_recall must be between 0 and 1")
        if not -1.0 <= self.confidence_bias <= 1.0 or self.confidence_noise < 0:
            raise ValueError(
                "confidence_bias must be within -1 and 1, and noise cannot be negative"
            )


PERSONAS = {
    persona.name: persona
    for persona in (
        Persona(
            name="crammer",
            summary="Studies once a week in one long session. Judges their memory fairly.",
            study_days=(6,),
            reviews_per_session=25,
            minutes_per_review=1.2,
            memory_gain=6.0,
            starting_half_life=4.0,
            new_topic_recall=0.25,
            confidence_bias=0.0,
            confidence_noise=0.08,
        ),
        Persona(
            name="steady",
            summary="Studies every day in short sessions. Judges their memory fairly.",
            study_days=(0, 1, 2, 3, 4, 5, 6),
            reviews_per_session=10,
            minutes_per_review=1.5,
            memory_gain=6.0,
            starting_half_life=4.0,
            new_topic_recall=0.25,
            confidence_bias=0.0,
            confidence_noise=0.08,
        ),
        Persona(
            name="overconfident",
            summary="Studies every other day and feels surer than their memory deserves.",
            study_days=(0, 2, 4),
            reviews_per_session=12,
            minutes_per_review=1.5,
            memory_gain=6.0,
            starting_half_life=4.0,
            new_topic_recall=0.25,
            confidence_bias=0.35,
            confidence_noise=0.1,
        ),
        Persona(
            name="underconfident",
            summary="Studies every day and doubts memory that is actually good.",
            study_days=(0, 1, 2, 3, 4, 5, 6),
            reviews_per_session=10,
            minutes_per_review=1.5,
            memory_gain=6.0,
            starting_half_life=4.0,
            new_topic_recall=0.25,
            confidence_bias=-0.35,
            confidence_noise=0.1,
        ),
    )
}
