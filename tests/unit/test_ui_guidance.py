import pytest

from calibrayn.services.analytics import Summary
from calibrayn.ui.components.guidance import next_step, steps
from calibrayn.ui.components.palette import (
    AMBER,
    BLUE,
    CHART_COLORS,
    GREEN,
    INK,
    MIST,
    NAVY,
    PAPER,
    RED,
    SLATE,
    TEAL,
)
from calibrayn.ui.components.samples import SAMPLES, sample_files


def summary(documents=0, attempts=0, due_now=0):
    return Summary(documents, documents, attempts and 1, due_now, attempts, None)


class TestSteps:
    def test_there_are_four_steps_in_a_fixed_order(self):
        titles = [step.title for step in steps(summary(), "mock")]
        assert titles == [
            "Add your notes",
            "Connect a language model",
            "Practise with the quiz",
            "Check your confidence",
        ]

    def test_nothing_is_done_at_the_start(self):
        assert not any(step.done for step in steps(summary(), "mock"))

    def test_notes_complete_the_first_step(self):
        assert steps(summary(documents=1), "mock")[0].done

    def test_a_real_model_completes_the_second_step(self):
        assert steps(summary(), "ollama:llama3.2")[1].done
        assert not steps(summary(), "mock")[1].done

    def test_one_answer_completes_the_quiz_step_and_ten_complete_the_last(self):
        assert steps(summary(attempts=1), "mock")[2].done
        assert not steps(summary(attempts=1), "mock")[3].done
        assert steps(summary(attempts=10), "mock")[3].done


class TestNextStep:
    def test_a_new_user_is_asked_for_notes_first(self):
        assert next_step(summary(), "mock").title == "Add your notes"

    def test_then_a_model(self):
        assert next_step(summary(documents=1), "mock").title == "Connect a language model"

    def test_then_the_quiz(self):
        assert next_step(summary(documents=1), "ollama:x").title == "Practise with the quiz"

    def test_then_the_confidence_check(self):
        step = next_step(summary(documents=1, attempts=3), "ollama:x")
        assert step.title == "Check your confidence"

    def test_when_everything_is_done_due_topics_are_mentioned(self):
        step = next_step(summary(documents=1, attempts=12, due_now=4), "ollama:x")
        assert step.title == "Keep going"
        assert "4 topics are due" in step.text

    def test_when_nothing_is_due_the_user_is_caught_up(self):
        step = next_step(summary(documents=1, attempts=12, due_now=0), "ollama:x")
        assert step.title == "All caught up"
        assert step.done


class TestPalette:
    @pytest.mark.parametrize(
        "colour", [NAVY, BLUE, TEAL, GREEN, AMBER, RED, INK, SLATE, MIST, PAPER]
    )
    def test_every_colour_is_a_six_digit_hex_code(self, colour):
        assert len(colour) == 7 and colour.startswith("#")
        int(colour[1:], 16)

    def test_the_colours_are_all_different(self):
        colours = [NAVY, BLUE, TEAL, GREEN, AMBER, RED, INK, SLATE, MIST, PAPER]
        assert len(set(colours)) == len(colours)

    def test_charts_use_the_main_palette_colours(self):
        assert CHART_COLORS == [BLUE, AMBER, GREEN, RED]

    def test_the_theme_file_uses_the_same_colours(self):
        tomllib = pytest.importorskip("tomllib")
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"
        with path.open("rb") as handle:
            theme = tomllib.load(handle)["theme"]
        assert theme["primaryColor"].upper() == BLUE
        assert theme["textColor"].upper() == INK
        assert theme["backgroundColor"].upper() == PAPER
        assert theme["secondaryBackgroundColor"].upper() == MIST
        assert theme["greenColor"].upper() == GREEN
        assert theme["orangeColor"].upper() == AMBER
        assert theme["redColor"].upper() == RED


class TestSamples:
    def test_five_documents_are_provided(self):
        assert len(SAMPLES) == 5
        assert len({name for name, _ in SAMPLES}) == 5

    def test_every_sample_is_plain_text_with_a_heading_and_body(self):
        for name, text in SAMPLES:
            assert name.endswith(".md")
            assert text.startswith("# ")
            assert len(text.split()) > 40
            text.encode("ascii")

    def test_the_files_come_back_as_bytes(self):
        files = sample_files()
        assert [name for name, _ in files] == [name for name, _ in SAMPLES]
        assert all(isinstance(content, bytes) for _, content in files)
