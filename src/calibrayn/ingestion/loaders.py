import io
import re
import zipfile
from pathlib import PurePath
from xml.etree import ElementTree

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from calibrayn.models.documents import LoadedFile, Page

SECTION_WORDS = 400
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
SLIDE_NAME = re.compile(r"ppt/slides/slide(\d+)\.xml")


class UnsupportedFile(ValueError):
    pass


class UnreadableFile(ValueError):
    pass


class NoTextFound(ValueError):
    pass


def in_sections(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]
    pages = []
    current = []
    count = 0
    for paragraph in paragraphs:
        current.append(paragraph)
        count += len(paragraph.split())
        if count >= SECTION_WORDS:
            pages.append(Page(len(pages) + 1, "\n\n".join(current)))
            current = []
            count = 0
    if current:
        pages.append(Page(len(pages) + 1, "\n\n".join(current)))
    return LoadedFile("section", tuple(pages))


def read_plain(content):
    return in_sections(content.decode("utf-8", errors="replace"))


def read_pdf(content):
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = tuple(
            Page(number, (page.extract_text() or "").strip())
            for number, page in enumerate(reader.pages, start=1)
        )
    except (PyPdfError, ValueError, OSError, KeyError, TypeError) as error:
        raise UnreadableFile("This PDF could not be read.") from error
    return LoadedFile("page", pages)


def read_docx(content):
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise UnreadableFile("This Word file could not be read.") from error
    paragraphs = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t")).strip()
        if text:
            paragraphs.append(text)
    return in_sections("\n\n".join(paragraphs))


def read_pptx(content):
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            slides = sorted(
                (int(match.group(1)), name)
                for name in archive.namelist()
                if (match := SLIDE_NAME.fullmatch(name))
            )
            pages = []
            for number, name in slides:
                root = ElementTree.fromstring(archive.read(name))
                lines = [(node.text or "").strip() for node in root.iter(f"{DRAWING_NS}t")]
                pages.append(Page(number, "\n".join(line for line in lines if line)))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise UnreadableFile("This PowerPoint file could not be read.") from error
    return LoadedFile("slide", tuple(pages))


READERS = {
    ".pdf": read_pdf,
    ".txt": read_plain,
    ".md": read_plain,
    ".docx": read_docx,
    ".pptx": read_pptx,
}


def load_file(name, content):
    suffix = PurePath(name).suffix.lower()
    read = READERS.get(suffix)
    if read is None:
        raise UnsupportedFile(
            f"{suffix or 'This kind of file'} is not supported. Use PDF, DOCX, PPTX, TXT or MD."
        )
    loaded = read(content)
    if not any(page.text.strip() for page in loaded.pages):
        raise NoTextFound("No text was found in this file. Scanned pages are not supported yet.")
    return loaded
