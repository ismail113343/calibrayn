from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from calibrayn.config import Settings
from calibrayn.models.calibration import Confidence
from calibrayn.services.workspace import Workspace

PAGES = Path(__file__).resolve().parents[2] / "src" / "calibrayn" / "ui"
NOTES = (
    "Photosynthesis turns light into chemical energy. Chlorophyll absorbs the light. "
    "The process makes glucose and oxygen."
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    settings = {
        "LLM_PROVIDER": "mock",
        "EMBEDDER": "hashing",
        "INDEX": "numpy",
        "PRIVACY_MODE": "local",
        "DATA_DIR": str(tmp_path),
        "DATABASE_PATH": str(tmp_path / "ui.db"),
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def open_page(name):
    return AppTest.from_file(str(PAGES / name), default_timeout=30).run()


def seed(answers=0):
    workspace = Workspace.open(Settings())
    workspace.add_document("biology.txt", NOTES.encode())
    for _ in range(answers):
        asked = workspace.agent.ask()
        workspace.agent.answer(asked, asked.question.answer_key, Confidence.SURE, 6.0)
    workspace.close()


def texts(elements):
    return [element.value for element in elements]


class TestHome:
    def test_an_empty_library_points_to_the_upload_page(self):
        page = open_page("Home.py")
        assert not page.exception
        assert page.title[0].value == "Calibrayn"
        assert any("Upload page" in value for value in texts(page.info))

    def test_the_numbers_reflect_the_library(self):
        seed()
        page = open_page("Home.py")
        values = {metric.label: metric.value for metric in page.metric}
        assert values["Documents"] == "1"
        assert values["Topics"] == "1"
        assert values["Due now"] == "1"

    def test_the_sidebar_warns_about_the_offline_demo_model(self):
        page = open_page("Home.py")
        assert any("offline demo model" in value for value in texts(page.sidebar.warning))
        assert any("stay on this computer" in value for value in texts(page.sidebar.caption))


class TestUpload:
    def test_an_empty_library_says_so(self):
        page = open_page("pages/1_Upload.py")
        assert not page.exception
        assert any("Nothing here yet" in value for value in texts(page.markdown))

    def test_documents_are_listed_and_can_be_removed(self):
        seed()
        page = open_page("pages/1_Upload.py")
        assert any("biology.txt" in value for value in texts(page.markdown))
        page.button(key="remove_1").click().run()
        assert not page.exception
        assert any("Nothing here yet" in value for value in texts(page.markdown))
        check = Workspace.open(Settings())
        assert check.library.documents() == []
        check.close()


class TestChat:
    def test_a_question_gets_a_cited_answer(self):
        seed()
        page = open_page("pages/2_Chat.py")
        page.chat_input[0].set_value("What does chlorophyll absorb?").run()
        assert not page.exception
        assert len(page.chat_message) == 2
        reply = page.chat_message[1]
        assert any("Chlorophyll" in value for value in texts(reply.markdown))
        assert any("biology.txt, section 1" in value for value in texts(reply.caption))

    def test_with_no_notes_it_says_it_found_nothing(self):
        page = open_page("pages/2_Chat.py")
        page.chat_input[0].set_value("anything").run()
        reply = page.chat_message[1]
        assert any("could not find" in value for value in texts(reply.markdown))

    def test_an_unrelated_question_shows_no_sources_and_no_warning(self):
        seed()
        page = open_page("pages/2_Chat.py")
        page.chat_input[0].set_value("quantum gravity").run()
        reply = page.chat_message[1]
        assert any("could not find" in value for value in texts(reply.markdown))
        assert not reply.caption
        assert not reply.warning


class TestTutor:
    def test_it_asks_for_notes_first(self):
        page = open_page("pages/3_Tutor.py")
        assert any("Add some notes" in value for value in texts(page.info))

    def test_a_topic_can_be_explained(self):
        seed()
        page = open_page("pages/3_Tutor.py")
        page.button(key="explain_button").click().run()
        assert not page.exception
        assert "Explanation" in texts(page.subheader)
        assert "Worked example" in texts(page.subheader)
        assert any("In short" in value for value in texts(page.markdown))


class TestQuiz:
    def test_with_no_notes_the_session_cannot_start(self):
        page = open_page("pages/4_Quiz.py")
        page.button(key="start_quiz").click().run()
        assert any("Add some notes" in value for value in texts(page.info))

    def test_starting_shows_a_question_and_asks_for_confidence_first(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        page.button(key="start_quiz").click().run()
        assert not page.exception
        assert page.radio[0].label == "How sure are you before you answer?"
        assert page.radio[0].value is None
        assert page.button(key="submit_answer").disabled
        page.radio[0].set_value("Sure").run()
        assert not page.button(key="submit_answer").disabled

    def test_a_right_answer_is_marked_and_scheduled(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        page.button(key="start_quiz").click().run()
        key = page.session_state["quiz_asked"].question.answer_key
        page.radio[0].set_value("Sure").run()
        page.text_area[0].set_value(key).run()
        page.button(key="submit_answer").click().run()
        assert not page.exception
        assert any("Good work" in value for value in texts(page.success))
        assert any("tomorrow" in value for value in texts(page.markdown))
        page.button(key="next_question").click().run()
        assert not page.exception
        assert page.radio[0].value is None

    def test_a_wrong_answer_comes_back_in_minutes(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        page.button(key="start_quiz").click().run()
        page.radio[0].set_value("Sure").run()
        page.text_area[0].set_value("bananas are yellow").run()
        page.button(key="submit_answer").click().run()
        assert any("Not yet" in value for value in texts(page.error))
        assert any("about 10 minutes" in value for value in texts(page.markdown))

    def test_removing_the_notes_during_a_session_resets_the_quiz(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        page.button(key="start_quiz").click().run()
        assert page.radio
        other = Workspace.open(Settings())
        other.remove_document(1)
        other.close()
        page.run()
        assert not page.exception
        assert not page.radio
        assert any("notes changed" in value for value in texts(page.info))
        assert page.button(key="start_quiz")

    def test_removing_a_document_on_the_upload_page_resets_the_quiz(self):
        seed()
        quiz = open_page("pages/4_Quiz.py")
        quiz.button(key="start_quiz").click().run()
        upload = AppTest.from_file(str(PAGES / "pages/1_Upload.py"), default_timeout=30)
        upload.session_state["quiz_stage"] = "asking"
        upload.run()
        upload.button(key="remove_1").click().run()
        assert upload.session_state["quiz_stage"] == "idle"

    def test_the_answer_is_stored(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        page.button(key="start_quiz").click().run()
        page.radio[0].set_value("Unsure").run()
        page.text_area[0].set_value("something").run()
        page.button(key="submit_answer").click().run()
        check = Workspace.open(Settings())
        stored = check.progress.attempts()
        assert len(stored) == 1
        assert stored[0].confidence is Confidence.UNSURE
        check.close()


class TestDashboard:
    def test_it_waits_for_answers(self):
        seed()
        page = open_page("pages/5_Dashboard.py")
        assert any("Answer a few quiz questions" in value for value in texts(page.info))

    def test_it_shows_progress_after_answers(self):
        seed(answers=3)
        page = open_page("pages/5_Dashboard.py")
        assert not page.exception
        headings = texts(page.subheader)
        assert "Score by day" in headings
        assert "Confidence and results" in headings
        assert "Topics to revisit" in headings
