from dataclasses import dataclass
from datetime import datetime

from calibrayn.config import Settings, load_tuning
from calibrayn.ingestion.pipeline import ingest
from calibrayn.llm.factory import build_client
from calibrayn.orchestrator.agent import StudyAgent
from calibrayn.rag.citations import cite
from calibrayn.rag.embeddings import build_embedder
from calibrayn.rag.grounding import is_supported, support_ratio
from calibrayn.rag.index import build_index
from calibrayn.rag.prompts import PromptBook
from calibrayn.rag.retriever import Retriever
from calibrayn.rag.writer import Writer
from calibrayn.services.analytics import Analytics
from calibrayn.services.grading import Grader
from calibrayn.storage.database import Database
from calibrayn.storage.library import Library
from calibrayn.storage.progress import Progress

NOT_FOUND_PHRASE = "could not find"


@dataclass(frozen=True)
class ChatReply:
    text: str
    citations: tuple[str, ...]
    support: float
    grounded: bool


@dataclass(frozen=True)
class Lesson:
    chunk_id: int
    title: str
    citation: str
    explanation: str
    worked_example: str


class Workspace:
    def __init__(self, settings, tuning, database, client, embedder, index, clock=datetime.now):
        self.settings = settings
        self.tuning = tuning
        self.database = database
        self.client = client
        self.library = Library(database)
        self.progress = Progress(database)
        self.retriever = Retriever(self.library, embedder, index, tuning.retrieval)
        self.retriever.refresh()
        prompts = PromptBook(settings.prompts_dir)
        self.writer = Writer(client, prompts)
        self.grader = Grader(client, prompts)
        self.agent = StudyAgent(
            self.library, self.progress, self.writer, self.grader, tuning, clock=clock
        )
        self.analytics = Analytics(self.library, self.progress, tuning)
        self.clock = clock

    @classmethod
    def open(cls, settings=None, tuning=None, client=None, embedder=None, index=None, clock=None):
        settings = settings or Settings()
        return cls(
            settings,
            tuning or load_tuning(settings.settings_file),
            Database(settings.db_path),
            client or build_client(settings),
            embedder or build_embedder(settings),
            index or build_index(settings),
            clock or datetime.now,
        )

    def add_document(self, name, content):
        return ingest(self.library, self.retriever, name, content, self.tuning.chunking)

    def remove_document(self, document_id):
        removed = self.library.delete_document(document_id)
        self.retriever.remove(removed)
        self.agent.reload()
        return len(removed)

    def citation(self, chunk):
        document = self.library.document(chunk.document_id)
        return cite(document, chunk) if document else f"page {chunk.page}"

    def chat(self, query):
        passages = self.retriever.retrieve(query)
        if not passages:
            return ChatReply("I could not find anything about that in your notes.", (), 0.0, False)
        text = self.writer.chat_answer(query, passages)
        if NOT_FOUND_PHRASE in text.lower():
            return ChatReply(text, (), 0.0, False)
        sources = [passage.chunk.text for passage in passages]
        return ChatReply(
            text=text,
            citations=tuple(self.citation(passage.chunk) for passage in passages),
            support=support_ratio(text, sources),
            grounded=is_supported(text, sources, self.tuning.grounding),
        )

    def lesson(self, chunk_id):
        chunk = self.library.chunk(chunk_id)
        if chunk is None:
            raise KeyError(f"There is no topic with id {chunk_id}.")
        related = self.retriever.retrieve(chunk.text, k=3, skip=(chunk.id,))
        return Lesson(
            chunk_id=chunk.id,
            title=chunk.title,
            citation=self.citation(chunk),
            explanation=self.writer.explanation(chunk, related),
            worked_example=self.writer.worked_example(chunk),
        )

    def close(self):
        self.database.close()
