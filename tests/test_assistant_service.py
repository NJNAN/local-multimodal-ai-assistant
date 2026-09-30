import threading
from types import SimpleNamespace

from src.assistant_service import AssistantService
from src.modal_router import ModalRouter
from src.rag import RAGPipeline


class FakeText:
    def __init__(self):
        self.calls = []
        self.closed = False

    def stream_generate(self, query, **kwargs):
        self.calls.append((query, kwargs))
        try:
            yield "第一段"
            yield "第二段"
        finally:
            self.closed = True


class FakeVision:
    def __init__(self):
        self.calls = []
        self.cancel = None

    def analyze_frames(self, frames, query, **kwargs):
        self.calls.append((frames, query, kwargs))
        if self.cancel:
            self.cancel()
        return "图片回答"


def make_service(sources=None):
    text = FakeText()
    vision = FakeVision()
    rag = SimpleNamespace(retrieve=lambda query, k: sources or [], build_context=RAGPipeline.build_context)
    video = SimpleNamespace(recent_frames=lambda count: ["camera1", "camera2"])
    return AssistantService(text, rag, vision, ModalRouter(), video)


def test_response_details_preserve_real_token_counts_and_vision_does_not_claim_stream_ttft():
    service = make_service()
    details = []
    client = SimpleNamespace(last_timings={"predicted_n": 5, "predicted_per_second": 42}, last_usage={})
    service.text_llm.client = client
    service.process("问题", mode="text", on_details=details.append)
    assert details[-1].inference["output_tokens"] == 5
    assert details[-1].inference["server_decode_tokens_s"] == 42
    assert details[-1].inference["ttft_seconds"] is not None
    service.vl_model.client = client
    service.process("图片", mode="vision", frames=["test"], on_details=details.append)
    assert details[-1].inference["output_tokens"] == 5
    assert details[-1].inference["ttft_seconds"] is None
    assert details[-1].inference["tpot_ms"] is None


def test_precancelled_request_cannot_reuse_previous_inference_metrics():
    service = make_service()
    service.text_llm.client = SimpleNamespace(last_timings={"predicted_n": 999}, last_usage={})
    event = threading.Event()
    event.set()
    details = []
    assert service.process("问题", cancel_event=event, on_details=details.append) == ""
    assert details[-1].cancelled
    assert details[-1].inference == {}


def test_text_stream_reports_sources_and_keeps_history():
    source = {"text": "课程资料", "source": "课程.pdf", "page": 2}
    service = make_service([source])
    chunks, details = [], []
    assert service.process("问题", chunks.append, on_details=details.append) == "第一段第二段"
    assert chunks == ["第一段", "第二段"]
    assert service.text_llm.calls[0][1]["kb_mode"] is True
    assert "课程.pdf" in service.text_llm.calls[0][1]["kb_context"]
    assert details[0].sources == [source]
    assert details[0].first_chunk_seconds <= details[0].elapsed_seconds
    assert len(service.history) == 2
    assert service.text_llm.closed


def test_stop_preserves_partial_answer_but_not_history_and_closes_stream():
    service = make_service()
    details, chunks = [], []

    def stop_after_chunk(chunk):
        chunks.append(chunk)
        service.cancel()

    assert service.process("停止测试", stop_after_chunk, on_details=details.append) == "第一段"
    assert chunks == ["第一段"]
    assert service.history == []
    assert details[0].cancelled
    assert service.text_llm.closed
    # Cancellation belongs to the old request and must not poison the next one.
    assert service.process("下一轮") == "第一段第二段"


def test_pre_cancelled_request_does_not_call_any_model():
    service = make_service()
    event = threading.Event()
    event.set()
    details = []
    assert service.process("不要运行", cancel_event=event, on_details=details.append) == ""
    assert not service.text_llm.calls
    assert not service.vl_model.calls
    assert details[0].cancelled


def test_uploaded_images_work_without_camera_and_use_multiturn_history():
    service = make_service()
    service.video_capture.recent_frames = lambda count: (_ for _ in ()).throw(AssertionError("camera must not be used"))
    assert service.process("是什么颜色", frames=["upload"]) == "图片回答"
    assert service.process("刚才说了什么", frames=["upload"]) == "图片回答"
    frames, _, kwargs = service.vl_model.calls[1]
    assert frames == ["upload"]
    assert kwargs["history"] == [
        {"role": "user", "content": "是什么颜色"},
        {"role": "assistant", "content": "图片回答"},
    ]


def test_cancelled_vision_answer_is_not_emitted_or_remembered():
    service = make_service()
    service.vl_model.cancel = service.cancel
    chunks, details = [], []
    assert service.process("/vision 画面", chunks.append, on_details=details.append) == ""
    assert chunks == []
    assert service.history == []
    assert details[0].cancelled


def test_command_prefix_cleanup_keeps_literal_commands_in_body():
    service = make_service()
    service.process("/TEXT 解释 /vision 命令")
    assert service.text_llm.calls[0][0] == "解释 /vision 命令"
    for i in range(8):
        service.process(f"问题{i}", mode="text")
    assert len(service.history) == 12
    service.reset_history()
    assert not service.history
