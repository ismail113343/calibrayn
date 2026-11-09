import pytest

from calibrayn.ingestion.chunker import ChunkConfig, make_chunks
from calibrayn.ingestion.loaders import (
    NoTextFound,
    UnreadableFile,
    UnsupportedFile,
    clean_markdown,
    in_sections,
    load_file,
)
from calibrayn.ingestion.pipeline import ingest
from calibrayn.models.documents import Page
from calibrayn.rag.embeddings import HashingEmbedder
from calibrayn.rag.index import NumpyIndex
from calibrayn.rag.retriever import Retriever
from calibrayn.storage.database import Database
from calibrayn.storage.library import Library


def paragraph(word, count=100):
    return " ".join([word] * count) + "."


class TestLoaders:
    def test_plain_text_becomes_one_section(self):
        loaded = load_file("notes.txt", b"Cells are small.\n\nThey divide.")
        assert loaded.unit == "section"
        assert len(loaded.pages) == 1
        assert "Cells are small." in loaded.pages[0].text

    def test_markdown_is_read_like_text_and_suffix_case_does_not_matter(self):
        assert load_file("NOTES.MD", b"# Title\n\nBody text here.").pages[0].number == 1

    def test_long_text_is_split_into_numbered_sections(self):
        text = "\n\n".join(paragraph("word") for _ in range(10)).encode()
        loaded = load_file("long.txt", text)
        assert [page.number for page in loaded.pages] == [1, 2, 3]

    def test_bytes_that_are_not_utf8_are_replaced_not_fatal(self):
        loaded = load_file("odd.txt", b"caf\xe9 notes are here")
        assert "notes are here" in loaded.pages[0].text

    def test_empty_text_has_no_text(self):
        with pytest.raises(NoTextFound):
            load_file("empty.txt", b"  \n\n  ")

    def test_in_sections_of_nothing_is_empty(self):
        assert in_sections("").pages == ()

    def test_pdf_pages_keep_their_numbers(self, pdf_bytes):
        loaded = load_file("lecture.pdf", pdf_bytes(["First page text", "Second page text"]))
        assert loaded.unit == "page"
        assert [page.number for page in loaded.pages] == [1, 2]
        assert "First page text" in loaded.pages[0].text
        assert "Second page text" in loaded.pages[1].text

    def test_a_pdf_without_text_is_reported(self, pdf_bytes):
        with pytest.raises(NoTextFound, match="Scanned"):
            load_file("scan.pdf", pdf_bytes([""]))

    def test_a_broken_pdf_is_reported(self):
        with pytest.raises(UnreadableFile):
            load_file("bad.pdf", b"this is not a pdf")

    def test_word_files_are_read_paragraph_by_paragraph(self, docx_bytes):
        loaded = load_file("essay.docx", docx_bytes(["First idea.", "Second idea."]))
        assert loaded.unit == "section"
        assert "First idea." in loaded.pages[0].text
        assert "Second idea." in loaded.pages[0].text

    def test_a_broken_word_file_is_reported(self):
        with pytest.raises(UnreadableFile):
            load_file("bad.docx", b"not a zip")

    def test_a_zip_without_a_word_document_is_reported(self, pptx_bytes):
        with pytest.raises(UnreadableFile):
            load_file("wrong.docx", pptx_bytes({1: ["x"]}))

    def test_slides_are_ordered_by_their_number(self, pptx_bytes):
        loaded = load_file("deck.pptx", pptx_bytes({10: ["Tenth"], 2: ["Second"]}))
        assert loaded.unit == "slide"
        assert [page.number for page in loaded.pages] == [2, 10]
        assert loaded.pages[0].text == "Second"

    def test_a_broken_deck_is_reported(self):
        with pytest.raises(UnreadableFile):
            load_file("bad.pptx", b"not a zip")

    def test_unsupported_types_are_refused_with_the_allowed_list(self):
        with pytest.raises(UnsupportedFile, match="PDF, DOCX, PPTX, TXT or MD"):
            load_file("program.exe", b"x")

    def test_a_file_without_a_suffix_is_refused(self):
        with pytest.raises(UnsupportedFile):
            load_file("README", b"x")


class TestChunkConfig:
    @pytest.mark.parametrize(
        "bad_settings",
        [
            {"target_words": 5},
            {"overlap_words": -1},
            {"target_words": 50, "overlap_words": 50},
        ],
    )
    def test_bad_settings_are_rejected(self, bad_settings):
        with pytest.raises(ValueError):
            ChunkConfig(**bad_settings)


def sentence_pages(count, words_each=5, page=1):
    text = " ".join(
        " ".join(f"w{index}x{place}" for place in range(words_each - 1)) + f" e{index}."
        for index in range(count)
    )
    return [Page(page, text)]


class TestChunker:
    def test_no_pages_give_no_chunks(self):
        assert make_chunks([]) == []
        assert make_chunks([Page(1, "   ")]) == []

    def test_a_short_page_is_one_chunk_with_a_title_and_page(self):
        drafts = make_chunks([Page(7, "Cells divide by mitosis. They copy their DNA first.")])
        assert len(drafts) == 1
        assert drafts[0].position == 0
        assert drafts[0].page == 7
        assert drafts[0].title == "Cells divide by mitosis."
        assert "DNA first" in drafts[0].text

    def test_no_chunk_is_longer_than_the_target(self):
        config = ChunkConfig(target_words=40, overlap_words=10)
        drafts = make_chunks(sentence_pages(30), config)
        assert len(drafts) > 3
        assert all(len(draft.text.split()) <= 40 for draft in drafts)

    def test_positions_run_in_order(self):
        config = ChunkConfig(target_words=30, overlap_words=0)
        drafts = make_chunks(sentence_pages(20), config)
        assert [draft.position for draft in drafts] == list(range(len(drafts)))

    def test_every_sentence_is_kept(self):
        config = ChunkConfig(target_words=30, overlap_words=10)
        drafts = make_chunks(sentence_pages(20), config)
        joined = " ".join(draft.text for draft in drafts)
        for index in range(20):
            assert f"w{index}x0" in joined

    def test_the_end_of_one_chunk_opens_the_next(self):
        config = ChunkConfig(target_words=30, overlap_words=10)
        drafts = make_chunks(sentence_pages(20), config)
        previous_tail = " ".join(drafts[0].text.split()[-10:])
        assert drafts[1].text.startswith(previous_tail)

    def test_no_overlap_means_no_shared_text(self):
        config = ChunkConfig(target_words=30, overlap_words=0)
        drafts = make_chunks(sentence_pages(20), config)
        first_words = set(drafts[0].text.split())
        assert not first_words & set(drafts[1].text.split())

    def test_a_chunk_starts_on_the_page_where_its_new_text_begins(self):
        pages = sentence_pages(6, page=1) + sentence_pages(6, page=2)
        config = ChunkConfig(target_words=30, overlap_words=0)
        drafts = make_chunks(pages, config)
        assert drafts[0].page == 1
        assert drafts[-1].page == 2

    def test_one_huge_sentence_is_cut_to_the_target(self):
        huge = " ".join(f"word{index}" for index in range(200))
        drafts = make_chunks([Page(1, huge)], ChunkConfig(target_words=50, overlap_words=0))
        assert len(drafts) == 4
        assert all(len(draft.text.split()) <= 50 for draft in drafts)

    def test_long_titles_are_shortened(self):
        long_line = "word " * 40
        drafts = make_chunks([Page(1, long_line)])
        assert len(drafts[0].title) <= 63
        assert drafts[0].title.endswith("...")

    def test_chunking_is_repeatable(self):
        config = ChunkConfig(target_words=30, overlap_words=10)
        assert make_chunks(sentence_pages(20), config) == make_chunks(sentence_pages(20), config)


@pytest.fixture
def library():
    database = Database(":memory:")
    yield Library(database)
    database.close()


@pytest.fixture
def retriever(library):
    return Retriever(library, HashingEmbedder(), NumpyIndex())


NOTES = (
    "Photosynthesis turns light into chemical energy. Chlorophyll absorbs the light. "
    "Mitochondria release energy from glucose. Cells divide by mitosis to grow."
)


class TestPipeline:
    def test_a_file_is_stored_and_searchable(self, library, retriever):
        document, chunks = ingest(library, retriever, "biology.txt", NOTES.encode())
        assert document.name == "biology.txt"
        assert len(chunks) == 1
        assert library.chunk_count() == 1
        assert len(retriever.index) == 1
        found = retriever.retrieve("chlorophyll light")
        assert found[0].chunk.id == chunks[0].id

    def test_a_pdf_keeps_its_page_numbers(self, library, retriever, pdf_bytes):
        content = pdf_bytes(["Page one is about cells.", "Page two is about energy."])
        _, chunks = ingest(library, retriever, "book.pdf", content)
        assert chunks[0].page == 1

    def test_a_refused_file_stores_nothing(self, library, retriever):
        with pytest.raises(UnsupportedFile):
            ingest(library, retriever, "tool.exe", b"x")
        assert library.documents() == []
        assert len(retriever.index) == 0

    def test_a_file_without_text_stores_nothing(self, library, retriever):
        with pytest.raises(NoTextFound):
            ingest(library, retriever, "blank.txt", b"   ")
        assert library.documents() == []

    def test_two_files_can_be_added(self, library, retriever):
        ingest(library, retriever, "a.txt", NOTES.encode())
        ingest(library, retriever, "b.txt", b"The Roman empire built long roads.")
        assert len(library.documents()) == 2
        assert len(retriever.index) == 2


class TestMarkdown:
    def test_a_heading_becomes_a_sentence_of_its_own(self):
        assert clean_markdown("# Cells\n\nCells divide.") == "Cells.\n\nCells divide."

    def test_every_heading_level_and_closing_hashes_are_removed(self):
        assert clean_markdown("### Why it matters ###") == "Why it matters."
        assert clean_markdown("## Why? ##") == "Why?"

    def test_a_heading_keeps_its_own_punctuation(self):
        assert clean_markdown("# Note:") == "Note:"

    def test_emphasis_markers_are_removed(self):
        assert clean_markdown("This is **very** and __truly__ so, *really* and _quite_.") == (
            "This is very and truly so, really and quite."
        )

    def test_underscores_inside_words_are_left_alone(self):
        assert clean_markdown("Use snake_case_name and 2*3*4 here.") == (
            "Use snake_case_name and 2*3*4 here."
        )

    def test_links_images_and_code_keep_only_their_text(self):
        text = "See [the docs](http://x.org) and ![a diagram](a.png) for `value`."
        assert clean_markdown(text) == "See the docs and a diagram for value."

    def test_list_items_become_separate_sentences(self):
        assert clean_markdown("- first\n* second.\n1. third\n2) fourth") == (
            "first.\nsecond.\nthird.\nfourth."
        )

    def test_quotes_lose_their_marker(self):
        assert clean_markdown("> A quoted line\n> and more") == "A quoted line\nand more"

    def test_code_fences_are_dropped_but_the_code_stays(self):
        assert clean_markdown("```python\nx = 1\n```").strip() == "x = 1"

    def test_plain_text_is_unchanged(self):
        text = "Plain sentence one. Plain sentence two."
        assert clean_markdown(text) == text

    def test_markdown_files_are_cleaned_when_loaded(self):
        loaded = load_file("notes.md", b"# Cells\n\n**Cells** divide.")
        assert loaded.pages[0].text == "Cells.\n\nCells divide."

    def test_text_files_are_not_treated_as_markdown(self):
        assert load_file("notes.txt", b"# not a heading").pages[0].text == "# not a heading"

    def test_a_markdown_file_gives_a_clean_topic_title(self, library, retriever):
        _, chunks = ingest(
            library, retriever, "bio.md", b"# Photosynthesis\n\nPhotosynthesis uses light."
        )
        assert chunks[0].title == "Photosynthesis."
        assert "#" not in chunks[0].text
