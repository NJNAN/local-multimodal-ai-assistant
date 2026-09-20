from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT

from config import MODEL_CONFIG, PATHS
from src.knowledge_base import VectorStore
from src.rag import RAGPipeline
from src.text_llm import TextLLM


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库 RAG 测试")
    parser.add_argument("documents", nargs="*", help="PDF/TXT/MD 文档")
    parser.add_argument("--query", default="大学养老知识库包含哪些主要内容？")
    args = parser.parse_args()
    documents = [*args.documents]
    if not documents:
        documents = [str(path) for path in PATHS["knowledge"].glob("**/*") if path.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}]
    if not documents:
        raise RuntimeError(f"没有文档。请把 PDF/TXT/MD/DOCX 放入 {PATHS['knowledge']}")
    llm = TextLLM(MODEL_CONFIG["text"])
    store = VectorStore(MODEL_CONFIG["embedding"])
    rag = RAGPipeline(store, llm)
    imported = sum(rag.import_document(path) for path in documents)
    results = rag.retrieve(args.query, 3)
    baseline = llm.generate(args.query, max_new_tokens=200)
    enhanced = rag.answer(args.query, 3)
    output = [
        f"导入文档: {len(documents)}，知识块: {imported}",
        f"问题: {args.query}",
        "\nTop-3 检索:",
        *[f"{i}. {x['source']} 距离={x['distance']:.4f}\n{x['text'][:240]}" for i, x in enumerate(results, 1)],
        f"\n无知识库回答:\n{baseline}",
        f"\nRAG 回答:\n{enhanced}",
    ]
    text = "\n".join(output)
    print(text)
    (PATHS["outputs"] / "kb_test_log.txt").write_text(text, encoding="utf-8")
    store.save(PATHS["knowledge"] / ".index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
