from __future__ import annotations

import threading


class AssistantService:
    def __init__(self, text_llm, rag, vl_model, router, video_capture):
        self.text_llm = text_llm
        self.rag = rag
        self.vl_model = vl_model
        self.router = router
        self.video_capture = video_capture
        self.history: list[dict] = []
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def process(self, query: str, on_chunk=None) -> str:
        self._cancel_event.clear()
        mode = self.router.route(query)
        clean_query = query.replace("/vision", "", 1).replace("/text", "", 1).strip()
        if mode == "vision":
            answer = self.vl_model.analyze_frames(
                self.video_capture.recent_frames(2), clean_query or "描述画面内容"
            )
            if on_chunk:
                on_chunk(answer)
        else:
            results = self.rag.retrieve(clean_query, k=3)
            context = self.rag.build_context(results)
            chunks = []
            for chunk in self.text_llm.stream_generate(
                clean_query,
                kb_context=context,
                kb_mode=bool(results),
                history=self.history,
            ):
                if self._cancel_event.is_set():
                    break
                chunks.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
            answer = "".join(chunks).strip()
        if answer:
            self.history.extend(
                [
                    {"role": "user", "content": clean_query},
                    {"role": "assistant", "content": answer},
                ]
            )
            self.history = self.history[-12:]
        return answer
