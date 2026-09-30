from src.knowledge_base import DocumentParser
from src.knowledge_base import VectorStore
from src.rag import RAGPipeline
from types import SimpleNamespace
import pytest


class FakeEmbedding:
    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        import numpy as np

        vectors = np.zeros((len(texts), 384), dtype="float32")
        vectors[:, 0] = 1
        return vectors


def test_txt_parsing_and_overlap(tmp_path):
    path = tmp_path / "knowledge.txt"
    path.write_text("甲" * 1200, encoding="utf-8")
    parser = DocumentParser()
    chunks = parser.parse_txt(path)
    assert len(chunks) == 3
    assert chunks[0]["source"] == "knowledge.txt"
    assert chunks[0]["text"][-50:] == chunks[1]["text"][:50]


def test_parser_rejects_unknown_format(tmp_path):
    path = tmp_path / "x.doc"
    path.write_text("x")
    try:
        DocumentParser().parse(path)
    except ValueError as exc:
        assert "不支持" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_missing_index_does_not_load_embedding_backend(tmp_path, monkeypatch):
    store = VectorStore()
    monkeypatch.setattr(store, "_ensure_backend", lambda: pytest.fail("unnecessary model loading"))
    assert store.load(tmp_path / "missing") == 0


def test_import_persists_index_and_reload_can_retrieve(tmp_path):
    faiss = pytest.importorskip("faiss")
    store = VectorStore(embedding_model=FakeEmbedding(), faiss_module=faiss)
    index_dir = tmp_path / "中文索引"
    path = tmp_path / "课程.txt"
    path.write_text("每周三下午两点开课。", encoding="utf-8")
    assert RAGPipeline(store, None, index_dir=index_dir).import_document(path) == 1
    reloaded = VectorStore(embedding_model=FakeEmbedding(), faiss_module=faiss)
    assert reloaded.load(index_dir) == 1
    assert reloaded.search("何时开课")[0]["source"] == "课程.txt"
    # A corrupt load must leave the previously working knowledge base intact.
    (index_dir / "documents.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="数量不一致"):
        reloaded.load(index_dir)
    assert reloaded.search("何时开课")[0]["text"] == "每周三下午两点开课。"


def test_empty_document_import_explains_problem_without_saving(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n  ", encoding="utf-8")
    rag = RAGPipeline(SimpleNamespace(), None, index_dir=tmp_path / "index")
    with pytest.raises(ValueError, match="没有可提取的文字"):
        rag.import_document(path)
    assert not (tmp_path / "index").exists()
