from __future__ import annotations

from pathlib import Path

from .knowledge_base import DocumentParser, VectorStore


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        text_llm,
        parser: DocumentParser | None = None,
    ):
        self.vector_store = vector_store
        self.text_llm = text_llm
        self.parser = parser or DocumentParser()

    def import_document(self, file_path: str | Path) -> int:
        documents = self.parser.parse(file_path)
        return self.vector_store.add_documents(documents)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        return self.vector_store.search(query, k)

    @staticmethod
    def build_context(results: list[dict]) -> str:
        sections = []
        for index, item in enumerate(results, start=1):
            page = f"，第 {item['page']} 页" if item.get("page") else ""
            sections.append(
                f"资料 {index}（{item.get('source', '未知来源')}{page}）\n{item['text']}"
            )
        return "\n\n".join(sections)

    def answer(self, query: str, k: int = 3) -> str:
        results = self.retrieve(query, k)
        return self.text_llm.generate(
            query,
            kb_context=self.build_context(results),
            kb_mode=True,
        )

    def stream_answer(self, query: str, k: int = 3):
        results = self.retrieve(query, k)
        yield from self.text_llm.stream_generate(
            query,
            kb_context=self.build_context(results),
            kb_mode=True,
        )
