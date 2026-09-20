from __future__ import annotations

import base64
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class VLModel:
    """Local llama.cpp (llama-server) vision adapter for Qwen3-VL.

    The model and its mmproj projector are loaded together by llama-server
    on first use. Exactly the newest two frames are sent for temporal
    context, matching the course requirement.
    """

    MAX_FRAMES = 2

    def __init__(
        self,
        model_name: str = "qwen3-vl-4b-instruct",
        base_url: str = "http://127.0.0.1:8081",
        timeout: float = 300,
        keep_alive: str = "5m",
        client=None,
    ):
        self.model_name = model_name
        self.keep_alive = keep_alive
        if client is None:
            from .llama_backend import create_vision_client

            client = create_vision_client()
        self.client = client

    def health(self) -> dict:
        models = self.client.list_models()
        if self.model_name not in models:
            return {"ok": False, "reason": f"模型不可用 {self.model_name}", "models": models}
        details = self.client.show_model(self.model_name)
        capabilities = details.get("capabilities", [])
        has_vision = "vision" in capabilities
        return {
            "ok": has_vision,
            "reason": "" if has_vision else "视觉模型缺少图像输入能力",
            "capabilities": capabilities,
        }

    def create_messages(self, video_paths: list[str], text_prompt: str) -> list[dict]:
        selected = video_paths[-self.MAX_FRAMES :]
        return [
            {
                "role": "system",
                "content": "你是多模态AI助手的视觉模块。只根据提供的画面回答，无法确认时明确说明。",
            },
            {
                "role": "user",
                "content": text_prompt,
                "images": [self._encode_path(path) for path in selected],
            },
        ]

    def generate_response(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        response = self.client.chat(
            {
                "model": self.model_name,
                "messages": messages,
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0.2, "num_predict": max_new_tokens},
            }
        )
        return str(response.get("message", {}).get("content", "")).strip()

    def analyze_frames(
        self,
        frames: list,
        text_prompt: str,
        max_new_tokens: int = 256,
        history: list[dict] | None = None,
    ) -> str:
        selected = frames[-self.MAX_FRAMES :]
        if not selected:
            raise ValueError("没有可供视觉理解的摄像头帧")
        images = [self._encode_frame(frame) for frame in selected]
        messages = [
            {
                "role": "system",
                "content": "你是多模态AI助手的视觉模块。只根据提供的画面回答，无法确认时明确说明。",
            }
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": text_prompt, "images": images})
        return self.generate_response(messages, max_new_tokens)

    @staticmethod
    def _encode_path(path: str | Path) -> str:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")

    @staticmethod
    def _encode_frame(frame) -> str:
        if isinstance(frame, (str, Path)):
            return VLModel._encode_path(frame)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("缺少 opencv-python") from exc
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
        )
        if not ok:
            raise ValueError("摄像头帧 JPEG 编码失败")
        return base64.b64encode(encoded.tobytes()).decode("ascii")
