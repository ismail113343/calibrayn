import io
import zipfile
from xml.sax.saxutils import escape

import pytest


def build_pdf(page_texts):
    count = len(page_texts)
    font_id = 3 + 2 * count
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(count))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>",
    ]
    for index, text in enumerate(page_texts):
        content_id = 4 + 2 * index
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    body = b"%PDF-1.4\n"
    offsets = []
    for number, text in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{text}\nendobj\n".encode("latin-1")
    table = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        body += f"{offset:010d} 00000 n \n".encode("latin-1")
    body += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{table}\n%%EOF".encode(
            "latin-1"
        )
    )
    return body


def build_docx(paragraphs):
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    runs = "".join(f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>" for text in paragraphs)
    document = f'<w:document xmlns:w="{namespace}"><w:body>{runs}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def build_pptx(slides):
    presentation = "http://schemas.openxmlformats.org/presentationml/2006/main"
    drawing = "http://schemas.openxmlformats.org/drawingml/2006/main"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for number, lines in slides.items():
            runs = "".join(f"<a:p><a:r><a:t>{escape(line)}</a:t></a:r></a:p>" for line in lines)
            xml = (
                f'<p:sld xmlns:p="{presentation}" xmlns:a="{drawing}"><p:cSld><p:spTree><p:sp>'
                f"<p:txBody>{runs}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
            )
            archive.writestr(f"ppt/slides/slide{number}.xml", xml)
    return buffer.getvalue()


@pytest.fixture
def pdf_bytes():
    return build_pdf


@pytest.fixture
def docx_bytes():
    return build_docx


@pytest.fixture
def pptx_bytes():
    return build_pptx
