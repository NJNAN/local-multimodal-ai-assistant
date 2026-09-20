from src.knowledge_base import DocumentParser


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
