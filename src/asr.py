"""语音识别引擎：FunASR SenseVoiceSmall 的惰性加载适配器。

首次调用 transcribe() 时才真正加载模型（GPU 失败自动回退 CPU）；
支持直接吃 16kHz WAV 路径或原始 PCM 字节；输出会清洗掉语言/情感标记。
"""
from __future__ import annotations

import logging
import re
import tempfile
import threading
import wave
from pathlib import Path

LOGGER = logging.getLogger(__name__)
TAG_PATTERN = re.compile(r"<\|[^|>]+\|>")  # 清洗 <|zh|>/<|NEUTRAL|> 之类标记


class ASREngine:
    """SenseVoiceSmall 适配器：惰性加载 + GPU/CPU 自动回退。"""

    def __init__(
        self,
        model_path: str = "iic/SenseVoiceSmall",
        device: str = "auto",
        language: str = "zh",
        use_itn: bool = True,
        model=None,
        hotwords=None,
    ):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.language = language
        self.use_itn = use_itn
        self.model = model
        self.hotwords = hotwords
        self._model_lock = threading.RLock()
        self.ready = threading.Event()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """把配置中的设备名解析为 FunASR 需要的具体设备（auto → cuda:0/cpu）。"""
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
        with self._model_lock:
            return self._load_model()

    def _load_model(self) -> "ASREngine":
        """确保模型已加载（幂等）：缺依赖给可读报错，GPU 失败自动回退 CPU。"""
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
                # 显存不足/驱动异常时不中断，退回 CPU 再试一次
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
        # 用 0.4 秒静音走一遍完整识别链路，把初始化开销挪到启动阶段
        silence = b"\x00\x00" * int(sample_rate * 0.4)
        self.transcribe(silence, sample_rate=sample_rate)
        self.ready.set()

    def transcribe(
        self,
        audio_path_or_bytes: str | Path | bytes,
        *,
        sample_rate: int = 16_000,
    ) -> str:
        """识别一段音频并返回清洗后的文本。

        参数可以是 WAV 文件路径，也可以是 16-bit 单声道 PCM 字节
        （字节会先转存为临时 WAV，识别完自动删除）。
        """
        self.load()
        temporary_path: Path | None = None
        if isinstance(audio_path_or_bytes, bytes):
            # 字节输入：落成临时 WAV 再喂给 FunASR
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
            with self._model_lock:
                result = self.model.generate(
                    input=str(audio_path), cache={}, language=self.language,
                    use_itn=self.use_itn, batch_size_s=60, disable_pbar=True,
                )
            raw_text = self._extract_text(result)
            text = self._clean_result(raw_text)
            corrected = self.hotwords.correct(text) if self.hotwords else text
            if corrected != text:
                LOGGER.info('热词纠正: %s -> %s', text, corrected)
            return corrected
        finally:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _extract_text(result) -> str:
        """从 FunASR 各种返回结构里挖出文本（list/tuple 逐层取第一个元素）。"""
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
        """去掉 <|...|> 标记，并清理标点前的多余空白。"""
        text = TAG_PATTERN.sub("", raw_text)
        text = re.sub(r"\s+([，。！？,.!?])", r"\1", text)
        return text.strip()
