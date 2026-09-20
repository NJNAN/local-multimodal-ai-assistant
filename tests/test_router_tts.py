from src.modal_router import ModalRouter
from src.tts import SentenceBuffer, TTSEngine


def test_router_auto_and_manual_modes():
    router = ModalRouter()
    assert router.route("你好") == "text"
    assert router.route("看完这集再说") == "text"      # 「看」的误匹配不再触发视觉
    assert router.route("帮我看看这个") == "vision"
    assert router.route("看看摄像头画面") == "vision"
    assert router.route("/text 继续聊天") == "text"
    assert router.route("看看这里") == "text"
    router.enable_auto()
    assert router.route("look at the camera") == "vision"


def test_sentence_buffer_handles_stream_chunks():
    buffer = SentenceBuffer()
    assert buffer.feed("你好，我是") == []
    assert buffer.feed("助手。第二") == ["你好，我是助手。"]
    assert buffer.flush() == ["第二"]


def test_tts_language_and_split(tmp_path):
    engine = TTSEngine(tmp_path)
    assert engine.detect_language("你好世界") == "zh"
    assert engine.detect_language("hello world") == "en"
    assert engine.split_sentences("你好。How are you? 好！") == ["你好。", "How are you?", "好！"]
