"""Real local models: text -> uploaded image -> visual follow-up -> session export.

No camera, microphone, TTS service or external document is required. This check
loads the configured GGUF models and saves reproducible evidence under outputs/.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from _bootstrap import PROJECT_ROOT
from config import MODEL_CONFIG, PATHS
from src.assistant_service import AssistantService
from src.conversation import export_conversation
from src.image_utils import write_image
from src.knowledge_base import VectorStore
from src.llama_backend import create_text_client, create_vision_client, shutdown
from src.modal_router import ModalRouter
from src.rag import RAGPipeline
from src.text_llm import TextLLM
from src.vl_model import VLModel


class NoCamera:
    def recent_frames(self, count=2):
        raise AssertionError("Uploaded-image requests must not access a camera")


def main() -> int:
    import cv2
    import numpy as np

    image = np.full((360, 540, 3), 255, dtype="uint8")
    cv2.rectangle(image, (30, 100), (160, 230), (255, 0, 0), -1)
    cv2.circle(image, (270, 165), 65, (0, 0, 255), -1)
    triangle = np.array([[370, 230], [505, 230], [438, 100]], dtype="int32")
    cv2.fillPoly(image, [triangle], (0, 180, 0))
    write_image(PATHS["outputs"] / "optimization_shapes.jpg", image)

    text = TextLLM(MODEL_CONFIG["text"], client=create_text_client())
    rag = RAGPipeline(VectorStore(MODEL_CONFIG["embedding"]), text)
    vision = VLModel(MODEL_CONFIG["vision"], client=create_vision_client())
    service = AssistantService(text, rag, vision, ModalRouter(), NoCamera())
    records = []
    checks = [
        ("请用一句简短的中文介绍你自己。", None),
        ("图中有哪些几何图形？用一句中文回答。", [image]),
        ("刚才提到的这些图形分别是什么颜色？用一句中文回答。", [image]),
    ]
    try:
        for query, frames in checks:
            details, chunks = [], []
            answer = service.process(query, chunks.append, frames=frames, on_details=details.append)
            assert answer and chunks, "Model returned no answer"
            assert len(details) == 1 and not details[0].cancelled
            records.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query": query, "answer": answer, "status": "completed",
                "details": asdict(details[0]),
            })
            print(f"[{details[0].mode} {details[0].elapsed_seconds:.2f}s] {answer}", flush=True)
        assert len(service.history) == 6
        assert any(word in records[1]["answer"] for word in ("圆", "方", "三角")), "Shape recognition did not match the test scene"
        assert any(word in records[2]["answer"] for word in ("红", "蓝", "绿")), "Color follow-up did not match the test scene"
        for suffix in (".json", ".md"):
            export_conversation(records, PATHS["outputs"] / f"optimization_model_check{suffix}")
        print("Text, uploaded-image, visual follow-up and export checks passed", flush=True)
        return 0
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
