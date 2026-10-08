import pytest
from src.asr import ASREngine


def test_saved_custom_terms_correct_real_asr_output(tmp_path):
    from src.hotwords import HotwordStore
    store = HotwordStore(tmp_path / 'terms.json')
    store.update([{'term': '通义千问', 'aliases': ['同义千问']},
                  {'term': 'FAISS', 'aliases': ['费斯', 'faiss']}], enabled=True)
    restored = HotwordStore(store.path)
    class Model:
        def generate(self, **kwargs):
            return [{'text': '<|zh|>同义千问用费斯检索，faiss不等于fair。'}]
    audio = tmp_path / 'a.wav'
    audio.write_bytes(b'test')
    engine = ASREngine(device='cpu', model=Model(), hotwords=restored)
    assert engine.transcribe(audio) == '通义千问用FAISS检索，FAISS不等于fair。'
    restored.update(restored.entries, enabled=False)
    assert engine.transcribe(audio) == '同义千问用费斯检索，faiss不等于fair。'


def test_longest_match_single_pass_and_english_word_boundaries(tmp_path):
    from src.hotwords import HotwordStore
    store = HotwordStore(tmp_path / 'terms.json')
    store.update([{'term': '千问X', 'aliases': ['千问模型']},
                  {'term': 'B', 'aliases': ['千问']},
                  {'term': 'RAG', 'aliases': ['rag']}])
    assert store.correct('千问模型和千问，fragment rag。') == '千问X和B，fragment RAG。'


def test_invalid_rules_do_not_replace_previous_saved_settings(tmp_path):
    from src.hotwords import HotwordStore
    store = HotwordStore(tmp_path / 'terms.json')
    store.update([{'term': '课程术语', 'aliases': ['课程数语']}])
    with pytest.raises(ValueError):
        store.update([{'term': '', 'aliases': ['错词']}])
    assert HotwordStore(store.path).correct('课程数语') == '课程术语'
