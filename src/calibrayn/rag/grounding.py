from dataclasses import dataclass

from calibrayn.text import content_words


@dataclass(frozen=True)
class GroundingConfig:
    min_support: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_support <= 1.0:
            raise ValueError("min_support must be between 0 and 1")


DEFAULT_CONFIG = GroundingConfig()


def support_ratio(answer, passages):
    claim = set(content_words(answer))
    if not claim:
        return 0.0
    known = set()
    for passage in passages:
        known.update(content_words(passage))
    return len(claim & known) / len(claim)


def is_supported(answer, passages, config=DEFAULT_CONFIG):
    return support_ratio(answer, passages) >= config.min_support
