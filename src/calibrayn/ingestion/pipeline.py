from calibrayn.ingestion.chunker import DEFAULT_CONFIG, make_chunks
from calibrayn.ingestion.loaders import NoTextFound, load_file


def ingest(library, retriever, name, content, config=DEFAULT_CONFIG):
    loaded = load_file(name, content)
    drafts = make_chunks(loaded.pages, config)
    if not drafts:
        raise NoTextFound("No text was found in this file.")
    vectors = retriever.embedder.embed_documents([draft.text for draft in drafts])
    document, chunks = library.add_document(name, loaded.unit, drafts, vectors)
    retriever.add([chunk.id for chunk in chunks], vectors)
    return document, chunks
