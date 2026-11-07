import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from calibrayn.config import Settings
from calibrayn.hub.rows import HubError
from calibrayn.models.calibration import Confidence
from calibrayn.services.workspace import Workspace
from calibrayn.ui.components.samples import sample_files

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
        assert any("Add your notes" in value for value in texts(page.info))

    def test_the_numbers_reflect_the_library(self):
        seed()
        page = open_page("Home.py")
        values = {metric.label: metric.value for metric in page.metric}
        assert values["Documents"] == "1"
        assert values["Topics"] == "1"
        assert values["Due now"] == "1"

    def test_the_sidebar_warns_about_the_offline_demo_model(self):
        page = open_page("Home.py")
        assert any(
            "No language model is connected" in value for value in texts(page.sidebar.warning)
        )
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


class OllamaStandIn(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"models": [{"name": "llama3.2:latest"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"message": {"content": "Four."}}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def ollama_url():
    server = HTTPServer(("127.0.0.1", 0), OllamaStandIn)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def seed_samples(answers=0):
    workspace = Workspace.open(Settings())
    for name, content in sample_files():
        workspace.add_document(name, content)
    for _ in range(answers):
        asked = workspace.agent.ask()
        workspace.agent.answer(asked, asked.question.answer_key, Confidence.SURE, 6.0)
    workspace.close()


class TestGuidedHome:
    def test_the_checklist_marks_finished_steps(self):
        seed()
        page = open_page("Home.py")
        assert any("Done: Add your notes" in value for value in texts(page.markdown))
        assert any("To do: Connect a language model" in value for value in texts(page.markdown))

    def test_the_next_step_follows_the_state(self):
        seed()
        page = open_page("Home.py")
        assert any("Connect a language model" in value for value in texts(page.info))

    def test_the_pages_are_explained(self):
        page = open_page("Home.py")
        assert any("Evaluation" in value for value in texts(page.markdown))


class TestUploadHelp:
    def test_the_sample_notes_can_be_added_in_one_click(self):
        page = open_page("pages/1_Upload.py")
        page.button(key="sample_button").click().run()
        assert not page.exception
        check = Workspace.open(Settings())
        assert len(check.library.documents()) == 5
        assert check.library.chunk_count() == 5
        check.close()

    def test_adding_the_samples_twice_does_not_duplicate_them(self):
        page = open_page("pages/1_Upload.py")
        page.button(key="sample_button").click().run()
        page.button(key="sample_button").click().run()
        check = Workspace.open(Settings())
        assert len(check.library.documents()) == 5
        check.close()

    def test_each_document_lists_its_topics(self):
        seed_samples()
        page = open_page("pages/1_Upload.py")
        assert len(page.expander) == 5
        assert any("(photosynthesis.md, section 1)" in value for value in texts(page.markdown))


class TestChatHelp:
    def test_suggestions_come_from_the_notes(self):
        seed_samples()
        page = open_page("pages/2_Chat.py")
        assert page.button(key="suggest_0")
        assert page.button(key="suggest_2")

    def test_clicking_a_suggestion_asks_it(self):
        seed_samples()
        page = open_page("pages/2_Chat.py")
        page.button(key="suggest_0").click().run()
        assert not page.exception
        assert len(page.chat_message) == 2
        assert any(
            "Explain this from my notes" in value for value in texts(page.chat_message[0].markdown)
        )

    def test_without_notes_the_page_says_what_to_do(self):
        page = open_page("pages/2_Chat.py")
        assert any("Upload page" in value for value in texts(page.info))


class TestQuizSummary:
    def start_and_answer(self, page, good=True):
        page.button(key="start_quiz").click().run()
        key = page.session_state["quiz_asked"].question.answer_key
        page.radio[0].set_value("Sure").run()
        page.text_area[0].set_value(key if good else "bananas").run()
        page.button(key="submit_answer").click().run()

    def test_ending_a_session_shows_a_summary(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        self.start_and_answer(page)
        page.button(key="end_session").click().run()
        assert not page.exception
        assert "Session summary" in texts(page.subheader)
        values = {metric.label: metric.value for metric in page.metric}
        assert values["Answers"] == "1"
        assert values["Correct"] == "1"
        assert values["Average score"] == "100%"
        assert any("Every answer was correct" in value for value in texts(page.markdown))

    def test_missed_topics_are_listed_for_revision(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        self.start_and_answer(page, good=False)
        page.button(key="end_session").click().run()
        assert any("Topics to revisit" in value for value in texts(page.markdown))

    def test_a_new_session_starts_with_a_clean_log(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        self.start_and_answer(page)
        page.button(key="end_session").click().run()
        page.button(key="new_session").click().run()
        assert not page.exception
        assert page.session_state["quiz_log"] == []
        assert page.radio

    def test_the_sidebar_tracks_the_session(self):
        seed()
        page = open_page("pages/4_Quiz.py")
        self.start_and_answer(page)
        assert any("This session: 1 answers" in value for value in texts(page.sidebar.markdown))


class TestDashboardHelp:
    def test_it_explains_how_to_read_the_charts(self):
        seed(answers=3)
        page = open_page("pages/5_Dashboard.py")
        assert "How to read this page" in [item.label for item in page.expander]

    def test_the_confidence_charts_start_after_ten_answers(self):
        seed(answers=3)
        page = open_page("pages/5_Dashboard.py")
        assert any("start after ten answers" in value for value in texts(page.markdown))

    def test_the_confidence_charts_appear_with_enough_answers(self):
        seed_samples(answers=11)
        page = open_page("pages/5_Dashboard.py")
        assert not page.exception
        assert any("Left: 100 means" in value for value in texts(page.caption))


class TestSetupPage:
    def test_it_shows_what_is_connected(self):
        page = open_page("pages/7_Setup.py")
        assert not page.exception
        assert page.title[0].value == "Model setup"
        table = page.dataframe[0].value
        assert "mock" in table["Value"].tolist()
        assert any("stand-in" in value for value in texts(page.warning))

    def test_it_says_when_ollama_is_missing(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
        page = open_page("pages/7_Setup.py")
        assert any("Ollama was not found" in value for value in texts(page.info))

    def test_it_lists_the_models_ollama_has(self, monkeypatch, ollama_url):
        monkeypatch.setenv("OLLAMA_BASE_URL", ollama_url)
        page = open_page("pages/7_Setup.py")
        assert any("llama3.2:latest" in value for value in texts(page.success))

    def test_testing_the_offline_model_explains_itself(self):
        page = open_page("pages/7_Setup.py")
        page.button(key="test_model").click().run()
        assert any("needs no test" in value for value in texts(page.info))

    def test_a_found_model_is_used_without_any_setting(self, monkeypatch, ollama_url):
        monkeypatch.setenv("LLM_PROVIDER", "auto")
        monkeypatch.setenv("OLLAMA_BASE_URL", ollama_url)
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
        page = open_page("pages/7_Setup.py")
        assert any(
            "Model: ollama:llama3.2:latest" in value for value in texts(page.sidebar.caption)
        )
        assert not page.sidebar.warning

    def test_the_connection_test_reports_the_reply(self, monkeypatch, ollama_url):
        monkeypatch.setenv("LLM_PROVIDER", "auto")
        monkeypatch.setenv("OLLAMA_BASE_URL", ollama_url)
        page = open_page("pages/7_Setup.py")
        page.button(key="test_model").click().run()
        assert any("answered in" in value for value in texts(page.success))
        assert any("Four." in value for value in texts(page.markdown))


class TestEvaluationPage:
    def test_questions_are_written_and_search_is_checked(self):
        seed_samples()
        page = open_page("pages/6_Evaluation.py")
        page.button(key="eval_build").click().run()
        assert not page.exception
        assert any("questions are ready" in value for value in texts(page.success))
        assert any("template questions" in value for value in texts(page.warning))
        page.button(key="eval_check").click().run()
        assert not page.exception
        labels = [metric.label for metric in page.metric]
        assert "Found in top 1" in labels
        assert "Average rank score" in labels

    def test_building_questions_without_notes_reports_it(self):
        page = open_page("pages/6_Evaluation.py")
        page.button(key="eval_build").click().run()
        assert any("Add some notes" in value for value in texts(page.error))

    def test_the_simulation_compares_the_chosen_learners(self):
        page = open_page("pages/6_Evaluation.py")
        page.multiselect(key="sim_personas").set_value(["steady", "overconfident"])
        page.slider(key="sim_days").set_value(14)
        page.slider(key="sim_seeds").set_value(3)
        page.slider(key="sim_topics").set_value(10)
        page.button(key="sim_run").click().run()
        assert not page.exception
        table = page.dataframe[0].value
        assert table["Learner"].tolist() == ["steady", "overconfident"]
        assert {"Plain SM-2", "Adjusted", "Difference", "Low", "High"} <= set(table.columns)

    def test_the_simulation_cannot_run_with_no_learners(self):
        page = open_page("pages/6_Evaluation.py")
        page.multiselect(key="sim_personas").set_value([])
        page.run()
        assert page.button(key="sim_run").disabled


class TestSetupHuggingFace:
    @pytest.fixture(autouse=True)
    def nothing_downloaded(self, monkeypatch):
        monkeypatch.setattr(
            "calibrayn.hub.cache.try_to_load_from_cache", lambda *args, **kwargs: None
        )

    def statuses(self, page):
        tables = [entry.value for entry in page.dataframe]
        return next(table for table in tables if "Item" in table.columns)

    def test_the_section_shows_what_is_downloaded(self):
        page = open_page("pages/7_Setup.py")
        assert not page.exception
        table = self.statuses(page)
        assert "Not downloaded" in table["Status"].tolist()
        assert any("Hugging Face token" in item for item in table["Item"].tolist())

    def test_missing_packages_are_explained(self, monkeypatch):
        monkeypatch.setattr("calibrayn.hub.cache.find_spec", lambda name: None)
        page = open_page("pages/7_Setup.py")
        assert any("requirements-ml.txt" in value for value in texts(page.info))
        assert "Not installed" in self.statuses(page)["Status"].tolist()

    def test_a_download_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            "calibrayn.hub.models.download_model", lambda repo, token: "/saved/here"
        )
        page = open_page("pages/7_Setup.py")
        page.button(key="download_embedding").click().run()
        assert not page.exception
        assert any("Downloaded BAAI/bge-small-en-v1.5" in value for value in texts(page.success))
        assert any("/saved/here" in value for value in texts(page.caption))

    def test_the_language_model_has_its_own_button(self, monkeypatch):
        asked = []
        monkeypatch.setattr(
            "calibrayn.hub.models.download_model", lambda repo, token: asked.append(repo) or "/x"
        )
        page = open_page("pages/7_Setup.py")
        page.button(key="download_llm").click().run()
        assert asked == ["Qwen/Qwen2.5-1.5B-Instruct"]

    def test_a_failed_download_shows_the_reason(self, monkeypatch):
        def broken(repo, token):
            raise HubError("Could not download: offline")

        monkeypatch.setattr("calibrayn.hub.models.download_model", broken)
        page = open_page("pages/7_Setup.py")
        page.button(key="download_embedding").click().run()
        assert any("offline" in value for value in texts(page.error))


class TestEvaluationBenchmarks:
    ROWS = [
        {
            "question": "Who built long roads for armies?",
            "context": "The Roman empire built long roads for armies.",
        },
        {
            "question": "How do atoms share electrons?",
            "context": "Atoms bond by sharing electrons in covalent bonds.",
        },
        {
            "question": "What does chlorophyll absorb?",
            "context": "Chlorophyll absorbs light for photosynthesis.",
        },
    ]

    def test_the_own_dataset_is_compared_with_keyword_search_and_chance(self):
        seed_samples()
        page = open_page("pages/6_Evaluation.py")
        page.button(key="eval_build").click().run()
        page.button(key="eval_check").click().run()
        assert not page.exception
        assert "Compared with a keyword search and with chance" in texts(page.subheader)
        assert any("only 5 topics" in value for value in texts(page.warning))
        systems = [row for frame in page.dataframe for row in frame.value.get("system", [])]
        assert "random guessing" in systems
        assert "keyword search (BM25)" in systems

    def test_the_public_buttons_wait_for_the_needed_fields(self):
        page = open_page("pages/6_Evaluation.py")
        assert page.button(key="hub_inspect").disabled
        assert page.button(key="hub_run").disabled

    def test_the_columns_of_a_public_dataset_can_be_shown(self, monkeypatch):
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: self.ROWS)
        page = open_page("pages/6_Evaluation.py")
        page.text_input(key="hub_repo").set_value("org/data").run()
        page.button(key="hub_inspect").click().run()
        assert not page.exception
        columns = [row for frame in page.dataframe for row in frame.value.get("column", [])]
        assert columns == ["question", "context"]

    def test_a_public_dataset_is_scored_against_the_baselines(self, monkeypatch):
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: self.ROWS)
        page = open_page("pages/6_Evaluation.py")
        page.text_input(key="hub_repo").set_value("org/data").run()
        page.text_input(key="hub_question").set_value("question").run()
        page.text_input(key="hub_context").set_value("context").run()
        page.button(key="hub_run").click().run()
        assert not page.exception
        assert any("3 questions over 3 passages" in value for value in texts(page.markdown))
        systems = [row for frame in page.dataframe for row in frame.value.get("system", [])]
        assert systems[-1] == "random guessing"

    def test_a_wrong_column_name_is_explained(self, monkeypatch):
        monkeypatch.setattr("calibrayn.hub.rows.default_loader", lambda *args: self.ROWS)
        page = open_page("pages/6_Evaluation.py")
        page.text_input(key="hub_repo").set_value("org/data").run()
        page.text_input(key="hub_question").set_value("wrong").run()
        page.text_input(key="hub_context").set_value("context").run()
        page.button(key="hub_run").click().run()
        assert any("context, question" in value for value in texts(page.error))

    def test_a_dataset_that_cannot_be_read_is_explained(self, monkeypatch):
        def broken(*args):
            raise HubError("Could not read org/data: offline")

        monkeypatch.setattr("calibrayn.hub.rows.default_loader", broken)
        page = open_page("pages/6_Evaluation.py")
        page.text_input(key="hub_repo").set_value("org/data").run()
        page.button(key="hub_inspect").click().run()
        assert any("offline" in value for value in texts(page.error))
