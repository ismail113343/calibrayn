import sys
from types import SimpleNamespace

import numpy as np
import pytest

from calibrayn.ingestion.pipeline import ingest
from calibrayn.llm.mock import MockClient
from calibrayn.models.documents import Chunk, Document
from calibrayn.models.learning import Support
from calibrayn.rag.citations import cite
from calibrayn.rag.embeddings import (
    QUERY_PREFIX,
    EmbedderError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from calibrayn.rag.grounding import GroundingConfig, is_supported, support_ratio
from calibrayn.rag.index import FaissIndex, NumpyIndex
from calibrayn.rag.prompts import PromptBook
from calibrayn.rag.retriever import Passage, RetrievalConfig, Retriever
from calibrayn.rag.writer import GeneratedQuestion, Writer, parse_json_reply
from calibrayn.storage.database import Database
from calibrayn.storage.library import Library

try:
    import faiss  # noqa: F401

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

PROMPTS = "configs/prompts"


def cosine(first, second):
    return float(np.dot(first, second))


class TestHashingEmbedder:
    def test_vectors_have_the_right_shape_and_unit_length(self):
        vectors = HashingEmbedder(64).embed_documents(["cells divide", "energy from light"])
        assert vectors.shape == (2, 64)
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_the_same_text_always_gives_the_same_vector(self):
        first = HashingEmbedder(64).embed_query("cells divide by mitosis")
        second = HashingEmbedder(64).embed_query("cells divide by mitosis")
        assert np.array_equal(first, second)

    def test_a_query_matches_the_same_text_as_a_document(self):
        embedder = HashingEmbedder(64)
        assert np.allclose(
            embedder.embed_query("cell energy"), embedder.embed_documents(["cell energy"])[0]
        )

    def test_similar_text_is_closer_than_unrelated_text(self):
        embedder = HashingEmbedder()
        query = embedder.embed_query("how do plants absorb light energy")
        close = embedder.embed_documents(["Plants absorb light energy with chlorophyll."])[0]
        far = embedder.embed_documents(["The Roman empire built long roads across Europe."])[0]
        assert cosine(query, close) > cosine(query, far)

    def test_text_with_no_content_words_is_a_zero_vector(self):
        assert not HashingEmbedder(64).embed_query("the and of").any()

    def test_no_texts_give_an_empty_matrix(self):
        assert HashingEmbedder(64).embed_documents([]).shape == (0, 64)

    def test_the_name_includes_the_size(self):
        assert HashingEmbedder(128).name == "hashing-128"

    def test_tiny_sizes_are_refused(self):
        with pytest.raises(ValueError):
            HashingEmbedder(4)


class RecordingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings):
        self.calls.append((list(texts), normalize_embeddings))
        return np.ones((len(texts), 3), dtype=np.float64)


class TestSentenceTransformerEmbedder:
    def test_documents_are_encoded_normalised_as_float32(self):
        model = RecordingModel()
        vectors = SentenceTransformerEmbedder("some/model", model).embed_documents(["a", "b"])
        assert vectors.shape == (2, 3)
        assert vectors.dtype == np.float32
        assert model.calls == [(["a", "b"], True)]

    def test_bge_models_get_the_query_prefix(self):
        model = RecordingModel()
        SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", model).embed_query("question")
        assert model.calls[0][0] == [QUERY_PREFIX + "question"]

    def test_other_models_get_no_prefix(self):
        model = RecordingModel()
        SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", model).embed_query(
            "q"
        )
        assert model.calls[0][0] == ["q"]

    def test_the_name_is_the_model_name(self):
        assert SentenceTransformerEmbedder("x/y", RecordingModel()).name == "x/y"

    def test_no_documents_give_an_empty_matrix(self):
        assert (
            SentenceTransformerEmbedder("x/y", RecordingModel()).embed_documents([]).shape[0] == 0
        )

    def test_a_missing_package_gives_a_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(EmbedderError, match="requirements-ml.txt"):
            SentenceTransformerEmbedder("x/y").embed_query("q")


class TestBuildEmbedder:
    def test_hashing(self):
        assert isinstance(build_embedder(SimpleNamespace(embedder="Hashing")), HashingEmbedder)

    def test_huggingface_uses_the_configured_model(self):
        chosen = SimpleNamespace(
            embedder="huggingface", hf_embedding_model="BAAI/bge-small-en-v1.5"
        )
        embedder = build_embedder(chosen)
        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert embedder.name == "BAAI/bge-small-en-v1.5"

    def test_unknown_choice_is_refused(self):
        with pytest.raises(EmbedderError, match="hashing or huggingface"):
            build_embedder(SimpleNamespace(embedder="magic"))


INDEX_TYPES = [
    NumpyIndex,
    pytest.param(
        FaissIndex, marks=pytest.mark.skipif(not HAS_FAISS, reason="faiss is not installed")
    ),
]


@pytest.fixture(params=INDEX_TYPES)
def index(request):
    return request.param()


def axis(position, size=4):
    vector = np.zeros(size, dtype=np.float32)
    vector[position] = 1.0
    return vector


class TestIndex:
    def test_an_empty_index_finds_nothing(self, index):
        assert index.search(axis(0), 3) == []
        assert len(index) == 0

    def test_the_closest_vector_comes_first(self, index):
        index.rebuild([10, 20, 30], np.vstack([axis(0), axis(1), axis(2)]))
        hits = index.search(axis(1), 3)
        assert hits[0].chunk_id == 20
        assert hits[0].score == pytest.approx(1.0)
        assert len(hits) == 3

    def test_k_limits_the_hits(self, index):
        index.rebuild([1, 2, 3], np.vstack([axis(0), axis(1), axis(2)]))
        assert len(index.search(axis(0), 2)) == 2
        assert len(index.search(axis(0), 10)) == 3

    def test_vectors_can_be_added_later(self, index):
        index.rebuild([1], np.vstack([axis(0)]))
        index.add([2], np.vstack([axis(1)]))
        assert len(index) == 2
        assert index.search(axis(1), 1)[0].chunk_id == 2

    def test_adding_to_an_empty_index_works(self, index):
        index.add([5], np.vstack([axis(3)]))
        assert index.search(axis(3), 1)[0].chunk_id == 5

    def test_adding_nothing_changes_nothing(self, index):
        index.add([], np.zeros((0, 4)))
        assert len(index) == 0

    def test_vectors_can_be_removed(self, index):
        index.rebuild([1, 2, 3], np.vstack([axis(0), axis(1), axis(2)]))
        index.remove([2])
        assert len(index) == 2
        assert 2 not in [hit.chunk_id for hit in index.search(axis(1), 3)]

    def test_removing_everything_empties_the_index(self, index):
        index.rebuild([1, 2], np.vstack([axis(0), axis(1)]))
        index.remove([1, 2])
        assert index.search(axis(0), 3) == []

    def test_rebuilding_with_nothing_empties_the_index(self, index):
        index.rebuild([1], np.vstack([axis(0)]))
        index.rebuild([], np.zeros((0, 0)))
        assert len(index) == 0

    def test_mismatched_sizes_are_refused(self, index):
        index.rebuild([1], np.vstack([axis(0)]))
        with pytest.raises(ValueError):
            index.add([2], np.vstack([axis(0, size=6)]))
        with pytest.raises(ValueError):
            index.search(axis(0, size=6), 1)


@pytest.fixture
def library():
    database = Database(":memory:")
    yield Library(database)
    database.close()


@pytest.fixture
def retriever(library):
    return Retriever(library, HashingEmbedder(), NumpyIndex())


BIOLOGY = b"Photosynthesis turns light into chemical energy. Chlorophyll absorbs the light."
HISTORY = b"The Roman empire built long roads. Roads helped armies and trade move quickly."
CHEMISTRY = b"Atoms bond by sharing electrons. Covalent bonds hold molecules together."


class TestRetriever:
    def test_config_must_ask_for_at_least_one_passage(self):
        with pytest.raises(ValueError):
            RetrievalConfig(top_k=0)

    def test_the_most_relevant_passage_comes_first(self, library, retriever):
        for name, text in (("bio.txt", BIOLOGY), ("hist.txt", HISTORY), ("chem.txt", CHEMISTRY)):
            ingest(library, retriever, name, text)
        found = retriever.retrieve("which pigment absorbs light for photosynthesis")
        assert isinstance(found[0], Passage)
        assert "Photosynthesis" in found[0].chunk.text
        assert found[0].score >= found[-1].score

    def test_k_limits_the_passages(self, library, retriever):
        for name, text in (("bio.txt", BIOLOGY), ("hist.txt", HISTORY), ("chem.txt", CHEMISTRY)):
            ingest(library, retriever, name, text)
        assert len(retriever.retrieve("energy light roads", k=2)) == 2

    def test_skipped_chunks_are_left_out_but_the_count_is_kept(self, library, retriever):
        for name, text in (("bio.txt", BIOLOGY), ("hist.txt", HISTORY), ("chem.txt", CHEMISTRY)):
            ingest(library, retriever, name, text)
        first = retriever.retrieve("energy light roads atoms", k=1)[0].chunk.id
        others = retriever.retrieve("energy light roads atoms", k=2, skip=(first,))
        assert len(others) == 2
        assert first not in [item.chunk.id for item in others]

    def test_a_blank_query_or_an_empty_library_finds_nothing(self, library, retriever):
        assert retriever.retrieve("anything") == []
        ingest(library, retriever, "bio.txt", BIOLOGY)
        assert retriever.retrieve("   ") == []

    def test_refresh_rebuilds_the_index_from_storage(self, library, retriever):
        ingest(library, retriever, "bio.txt", BIOLOGY)
        fresh = Retriever(library, HashingEmbedder(), NumpyIndex())
        assert len(fresh.index) == 0
        fresh.refresh()
        assert len(fresh.index) == 1
        assert fresh.retrieve("chlorophyll")[0].chunk.document_id == 1

    def test_a_new_embedder_triggers_re_embedding(self, library, retriever):
        retriever.refresh()
        ingest(library, retriever, "bio.txt", BIOLOGY)
        smaller = Retriever(library, HashingEmbedder(64), NumpyIndex())
        smaller.refresh()
        _, matrix = library.vectors()
        assert matrix.shape[1] == 64
        assert library.embedder_name() == "hashing-64"
        assert smaller.retrieve("chlorophyll light")

    def test_removed_chunks_stop_appearing(self, library, retriever):
        ingest(library, retriever, "bio.txt", BIOLOGY)
        _, chunks = ingest(library, retriever, "hist.txt", HISTORY)
        retriever.remove([chunks[0].id])
        found = retriever.retrieve("roman roads armies", k=5)
        assert chunks[0].id not in [item.chunk.id for item in found]


class TestPromptBook:
    def test_a_prompt_has_a_system_and_a_user_part(self, tmp_path):
        (tmp_path / "hello.txt").write_text("SYSTEM $name\n---\nUSER $name", encoding="utf-8")
        assert PromptBook(tmp_path).render("hello", name="Sam") == ("SYSTEM Sam", "USER Sam")

    def test_dollar_signs_in_values_are_left_alone(self, tmp_path):
        (tmp_path / "p.txt").write_text("S\n---\n$body", encoding="utf-8")
        assert (
            PromptBook(tmp_path).render("p", body="costs $5 and $name")[1] == "costs $5 and $name"
        )

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PromptBook(tmp_path).render("nothing")

    def test_a_file_without_a_divider_is_reported(self, tmp_path):
        (tmp_path / "flat.txt").write_text("just one part", encoding="utf-8")
        with pytest.raises(ValueError, match="---"):
            PromptBook(tmp_path).render("flat")

    def test_a_missing_field_is_reported(self, tmp_path):
        (tmp_path / "p.txt").write_text("S\n---\n$needed", encoding="utf-8")
        with pytest.raises(KeyError):
            PromptBook(tmp_path).render("p")

    @pytest.mark.parametrize(
        ("name", "fields"),
        [
            ("question", {"level": 3, "level_hint": "h", "style": "s", "passage": "p"}),
            (
                "grading",
                {"question": "q", "answer_key": "a", "student_answer": "s", "passage": "p"},
            ),
            ("explain", {"passage": "p", "related": "r"}),
            ("example", {"passage": "p"}),
            ("chat", {"question": "q", "passages": "[1] p"}),
        ],
    )
    def test_the_shipped_prompts_render_completely(self, name, fields):
        system, user = PromptBook(PROMPTS).render(name, **fields)
        assert system.startswith("TASK:")
        assert "$" not in system + user


def chunk(text="Photosynthesis turns light into chemical energy. Chlorophyll absorbs light."):
    return Chunk(1, 1, 0, 3, "Photosynthesis turns light into chemical energy.", text)


class CapturingClient:
    name = "capture"
    hosted = False

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, system, prompt, max_tokens=400, temperature=0.2):
        self.prompts.append((system, prompt))
        return self.reply


def writer_with(client):
    return Writer(client, PromptBook(PROMPTS))


class TestParseJsonReply:
    def test_plain_json(self):
        assert parse_json_reply('{"a": 1}') == {"a": 1}

    def test_json_inside_a_code_fence(self):
        assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_chatter_around_it(self):
        assert parse_json_reply('Sure! {"a": 1} Hope that helps.') == {"a": 1}

    @pytest.mark.parametrize("text", ["no json here", "{broken", "[1, 2]", ""])
    def test_anything_else_is_none(self, text):
        assert parse_json_reply(text) is None


class TestWriter:
    def test_a_question_comes_from_the_passage(self):
        made = writer_with(MockClient()).question(chunk(), 3, Support.DIRECT_RECALL)
        assert isinstance(made, GeneratedQuestion)
        assert made.text.startswith("Explain in your own words")
        assert made.answer_key.startswith("Photosynthesis turns light")

    def test_the_level_and_style_reach_the_prompt(self):
        client = CapturingClient('{"question": "Q?", "answer": "A"}')
        writer_with(client).question(chunk(), 5, Support.WORKED_EXAMPLE)
        prompt = client.prompts[0][1]
        assert "Difficulty level: 5 out of 5" in prompt
        assert "apply the idea" in prompt
        assert "worked example" in prompt
        assert "Photosynthesis turns light" in prompt

    def test_an_unknown_level_falls_back_to_the_middle_hint(self):
        client = CapturingClient('{"question": "Q?", "answer": "A"}')
        writer_with(client).question(chunk(), 9, Support.DIRECT_RECALL)
        assert "in their own words" in client.prompts[0][1]

    def test_json_in_a_fence_is_understood(self):
        client = CapturingClient('```json\n{"question": "Why?", "answer": "Because."}\n```')
        made = writer_with(client).question(chunk(), 2, Support.DIRECT_RECALL)
        assert (made.text, made.answer_key) == ("Why?", "Because.")

    def test_a_missing_answer_falls_back_to_the_passage(self):
        client = CapturingClient('{"question": "Why?"}')
        made = writer_with(client).question(chunk(), 2, Support.DIRECT_RECALL)
        assert made.text == "Why?"
        assert made.answer_key.startswith("Photosynthesis")

    def test_a_plain_text_reply_becomes_the_question(self):
        client = CapturingClient("What is photosynthesis?\nExtra line.")
        made = writer_with(client).question(chunk(), 2, Support.DIRECT_RECALL)
        assert made.text == "What is photosynthesis?"
        assert made.answer_key.startswith("Photosynthesis")

    def test_an_empty_reply_gets_a_default_question(self):
        made = writer_with(CapturingClient("")).question(chunk(), 2, Support.DIRECT_RECALL)
        assert made.text.startswith("What do your notes say about Photosynthesis")

    def test_an_explanation_lists_the_related_notes(self):
        client = CapturingClient("An explanation.")
        related = [Passage(chunk("Chloroplasts hold the pigment."), 0.5)]
        assert writer_with(client).explanation(chunk(), related) == "An explanation."
        assert "- Chloroplasts hold the pigment." in client.prompts[0][1]

    def test_an_explanation_without_related_notes_says_none(self):
        client = CapturingClient("x")
        writer_with(client).explanation(chunk(), [])
        assert "RELATED:\nNone\nEND RELATED" in client.prompts[0][1]

    def test_a_worked_example_uses_the_example_prompt(self):
        client = CapturingClient("steps")
        assert writer_with(client).worked_example(chunk()) == "steps"
        assert client.prompts[0][0].startswith("TASK: example")

    def test_a_chat_answer_numbers_the_passages(self):
        client = CapturingClient("answer")
        passages = [Passage(chunk("First text."), 0.9), Passage(chunk("Second text."), 0.8)]
        writer_with(client).chat_answer("What?", passages)
        prompt = client.prompts[0][1]
        assert "[1] First text." in prompt
        assert "[2] Second text." in prompt
        assert "QUESTION:\nWhat?" in prompt


class TestGrounding:
    def test_an_answer_made_of_note_words_is_fully_supported(self):
        assert support_ratio("Chlorophyll absorbs light", ["Chlorophyll absorbs the light."]) == 1.0

    def test_an_answer_with_invented_words_has_low_support(self):
        ratio = support_ratio("Dragons guard golden treasure", ["Chlorophyll absorbs light."])
        assert ratio == 0.0

    def test_support_is_the_share_of_known_words(self):
        ratio = support_ratio(
            "chlorophyll absorbs dragons treasure", ["Chlorophyll absorbs light."]
        )
        assert ratio == pytest.approx(0.5)

    def test_an_answer_with_nothing_to_check_has_no_support(self):
        assert support_ratio("the and of", ["anything at all"]) == 0.0
        assert support_ratio("", ["anything at all"]) == 0.0

    def test_several_passages_are_pooled(self):
        assert support_ratio("alpha beta", ["alpha here", "beta here"]) == 1.0

    def test_the_threshold_decides(self):
        passages = ["chlorophyll absorbs light"]
        assert is_supported("chlorophyll absorbs dragons treasure", passages, GroundingConfig(0.5))
        assert not is_supported(
            "chlorophyll absorbs dragons treasure", passages, GroundingConfig(0.6)
        )

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_the_threshold_must_be_between_zero_and_one(self, value):
        with pytest.raises(ValueError):
            GroundingConfig(value)


def test_a_citation_names_the_file_and_the_place():
    from datetime import datetime

    document = Document(1, "lecture.pdf", "page", datetime(2026, 1, 1))
    assert cite(document, chunk()) == "lecture.pdf, page 3"
