from __future__ import annotations

import logging
import re
import tempfile
import wave
from pathlib import Path

LOGGER = logging.getLogger(__name__)
TAG_PATTERN = re.compile(r"<\|[^|>]+\|>")


class ASREngine:
    """Lazy FunASR/SenseVoiceSmall adapter."""

    def __init__(
        self,
        model_path: str = "iic/SenseVoiceSmall",
        device: str = "auto",
        language: str = "zh",
        use_itn: bool = True,
        model=None,
    ):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.language = language
        self.use_itn = use_itn
        self.model = model

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda":
            return "cuda:0"
        if device != "auto":
            return device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self) -> "ASREngine":
        if self.model is not None:
            return self
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "缺少 FunASR。请运行项目安装脚本或 pip install funasr modelscope。"
            ) from exc
        LOGGER.info("正在加载 SenseVoice: %s (%s)", self.model_path, self.device)
        kwargs = {
            "model": self.model_path,
            "device": self.device,
            "disable_update": True,
            "trust_remote_code": True,
        }
        try:
            self.model = AutoModel(**kwargs)
        except Exception:
            if self.device.startswith("cuda"):
                LOGGER.exception("GPU 加载 ASR 失败，回退到 CPU")
                self.device = "cpu"
                kwargs["device"] = "cpu"
                self.model = AutoModel(**kwargs)
            else:
                raise
        return self

    def warmup(self, sample_rate: int = 16_000) -> None:
        """提前加载模型并跑一次静音热身，消除首次语音的等待。"""
        self.load()
        silence = b"\x00\x00" * int(sample_rate * 0.4)
        self.transcribe(silence, sample_rate=sample_rate)

    def transcribe(
        self,
        audio_path_or_bytes: str | Path | bytes,
        *,
        sample_rate: int = 16_000,
    ) -> str:
        self.load()
        temporary_path: Path | None = None
        if isinstance(audio_path_or_bytes, bytes):
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            temporary_path = Path(handle.name)
            with wave.open(str(temporary_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_path_or_bytes)
            audio_path = temporary_path
        else:
            audio_path = Path(audio_path_or_bytes)
            if not audio_path.exists():
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        try:
            result = self.model.generate(
                input=str(audio_path),
                cache={},
                language=self.language,
                use_itn=self.use_itn,
                batch_size_s=60,
            )
            raw_text = self._extract_text(result)
            return self._clean_result(raw_text)
        finally:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _extract_text(result) -> str:
        value = result
        while isinstance(value, (list, tuple)):
            if not value:
                return ""
            value = value[0]
        if isinstance(value, dict):
            for key in ("text", "sentence", "value"):
                if key in value:
                    return str(value[key])
            return ""
        return str(value or "")

    def _clean_result(self, raw_text: str) -> str:
        text = TAG_PATTERN.sub("", raw_text)
        text = re.sub(r"\s+([，。！？,.!?])", r"\1", text)
        return text.strip()
