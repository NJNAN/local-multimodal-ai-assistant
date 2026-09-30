from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class ResponseDetails:
    mode: str
    elapsed_seconds: float
    first_chunk_seconds: float | None
    cancelled: bool = False
    sources: list[dict] = field(default_factory=list)
    inference: dict = field(default_factory=dict)


class AssistantService:
    def __init__(self, text_llm, rag, vl_model, router, video_capture):
        self.text_llm = text_llm
        self.rag = rag
        self.vl_model = vl_model
        self.router = router
        self.video_capture = video_capture
        self.history: list[dict] = []
        self._cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._process_lock = threading.Lock()

    def cancel(self) -> None:
        with self._state_lock:
            self._cancel_event.set()

    def reset_history(self) -> None:
        with self._process_lock:
            self.history.clear()

    @staticmethod
    def _clean_query(query: str) -> str:
        # Preserve literal commands in the middle of a user's message.
        for command in ("/vision", "/text"):
            if query.lower().startswith(command):
                return query[len(command):].strip()
        return query.strip()

    def process(
        self, query: str, on_chunk=None, *, frames=None, cancel_event=None,
        mode: str | None = None, on_details=None,
    ) -> str:
        """Serialize requests; keep cancelled answers out of conversation history.

        ``frames`` supplies local images without requiring a camera. A request-owned
        event also honours a stop received before processing starts. The original
        process(query, on_chunk) interface remains supported.
        """
        with self._process_lock:
            event = cancel_event if cancel_event is not None else threading.Event()
            with self._state_lock:
                self._cancel_event = event
            started = time.perf_counter()
            first_chunk = None
            sources = []
            selected_mode = mode or ("vision" if frames is not None else self.router.route(query))
            if selected_mode not in {"text", "vision"}:
                raise ValueError("mode 必须是 text 或 vision")
            clean_query = self._clean_query(query)
            answer = ""

            def emit(chunk):
                nonlocal first_chunk
                if event.is_set() or not chunk:
                    return
                if first_chunk is None:
                    first_chunk = time.perf_counter() - started
                if on_chunk:
                    on_chunk(chunk)

            if not event.is_set() and selected_mode == "vision":
                selected_frames = frames if frames is not None else self.video_capture.recent_frames(2)
                answer = self.vl_model.analyze_frames(
                    selected_frames, clean_query or "描述画面内容", history=list(self.history),
                )
                if event.is_set():
                    answer = ""
                else:
                    emit(answer)
            elif not event.is_set():
                sources = self.rag.retrieve(clean_query, k=3)
                context = self.rag.build_context(sources)
                chunks = []
                if not event.is_set():
                    stream = self.text_llm.stream_generate(
                        clean_query, kb_context=context, kb_mode=bool(sources),
                        history=list(self.history),
                    )
                    try:
                        for chunk in stream:
                            if event.is_set():
                                break
                            chunks.append(chunk)
                            emit(chunk)
                            if event.is_set():
                                break
                    finally:
                        close = getattr(stream, "close", None)
                        if close:
                            close()
                answer = "".join(chunks).strip()
            if answer and not event.is_set():
                self.history.extend([
                    {"role": "user", "content": clean_query},
                    {"role": "assistant", "content": answer},
                ])
                self.history = self.history[-12:]
            if on_details:
                client = getattr(self.vl_model if selected_mode == "vision" else self.text_llm, "client", None)
                from .inference_metrics import response_measurements
                inference = response_measurements(getattr(client, "last_timings", {}), getattr(client, "last_usage", {}),
                                                  time.perf_counter() - started, first_chunk) if client and answer and not event.is_set() else {}
                if selected_mode == "vision" and inference:
                    # The vision adapter returns a complete answer rather than token SSE.
                    inference["ttft_seconds"] = inference["tpot_ms"] = None
                on_details(ResponseDetails(
                    mode=selected_mode,
                    elapsed_seconds=time.perf_counter() - started,
                    first_chunk_seconds=first_chunk,
                    cancelled=event.is_set(),
                    sources=[item.copy() for item in sources],
                    inference=inference,
                ))
            return answer
