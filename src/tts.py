from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path


class TTSEngine:
    def __init__(
        self,
        output_dir: str | Path,
        zh_voice: str = "zh-CN-XiaoxiaoNeural",
        en_voice: str = "en-US-JennyNeural",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zh_voice = zh_voice
        self.en_voice = en_voice

    def detect_language(self, text: str) -> str:
        chinese_chars = len(re.findall(r"[\u3400-\u9fff]", text))
        latin_chars = len(re.findall(r"[A-Za-z]", text))
        if chinese_chars or latin_chars:
            return "zh" if chinese_chars >= latin_chars * 0.25 else "en"
        try:
            import langid

            language, _ = langid.classify(text)
            return "zh" if language.startswith("zh") else "en"
        except ImportError:
            return "zh"

    async def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
    ) -> Path:
        if not text.strip():
            raise ValueError("TTS 文本不能为空")
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("缺少 edge-tts") from exc
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected_voice = voice or (
            self.zh_voice if self.detect_language(text) == "zh" else self.en_voice
        )
        communicate = edge_tts.Communicate(text.strip(), selected_voice)
        await communicate.save(str(output))
        return output

    def synthesize_sync(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
        retries: int = 2,
    ) -> Path:
        """网络抖动时自动重试（edge-tts 走微软在线服务，偶发连接超时）。"""
        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                return self._synthesize_sync_once(text, output_path, voice)
            except BaseException as exc:  # noqa: BLE001 – 需要覆盖 asyncio.TimeoutError
                last_error = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _synthesize_sync_once(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
    ) -> Path:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.synthesize(text, output_path, voice))
        result: list[Path] = []
        error: list[BaseException] = []

        def runner():
            try:
                result.append(asyncio.run(self.synthesize(text, output_path, voice)))
            except BaseException as exc:
                error.append(exc)

        import threading

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return result[0]

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text)
        return [part.strip() for part in parts if part.strip()]


class SentenceBuffer:
    """Turns arbitrary LLM stream chunks into complete TTS sentences."""

    def __init__(self):
        self.pending = ""

    def feed(self, chunk: str) -> list[str]:
        self.pending += chunk
        matches = list(re.finditer(r"[。！？!?；;\n]", self.pending))
        if not matches:
            return []
        end = matches[-1].end()
        complete, self.pending = self.pending[:end], self.pending[end:]
        return TTSEngine.split_sentences(complete)

    def flush(self) -> list[str]:
        remaining = self.pending.strip()
        self.pending = ""
        return [remaining] if remaining else []
