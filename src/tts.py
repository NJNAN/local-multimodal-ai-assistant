"""语音合成（TTS）引擎：edge-tts 中英文合成 + 流式分句工具。

TTSEngine 负责「文本 → mp3 文件」的合成（中英文自动选音色、网络抖动重试）；
SentenceBuffer 把 LLM 的流式输出攒成完整句子，供边生成边播放的播放器使用。
"""
from __future__ import annotations

import asyncio
import re
import os
import logging
import threading
import time
from pathlib import Path


class SpeechCancelled(Exception):
    """本轮朗读已取消。"""


class TTSEngine:
    """edge-tts 封装：语言检测、异步合成与同步包装。"""

    def __init__(
        self,
        output_dir: str | Path,
        zh_voice: str = "zh-CN-XiaoxiaoNeural",
        en_voice: str = "en-US-JennyNeural",
        timeout_seconds: float = 10.0,
        backend: str = 'online',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zh_voice = zh_voice
        self.en_voice = en_voice
        self.timeout_seconds = timeout_seconds
        self.backend = backend
        self._native = None
        self._native_unavailable = False
        self._native_lock = threading.RLock()

    def set_backend(self, backend):
        if backend not in {'auto', 'windows', 'online'}:
            raise ValueError('未知语音合成方式')
        self.backend = backend

    @property
    def audio_suffix(self):
        return '.wav' if self.backend in {'auto', 'windows'} and os.name == 'nt' and not self._native_unavailable else '.mp3'

    def warmup(self):
        if self.backend in {'auto', 'windows'} and os.name == 'nt':
            from .windows_tts import WindowsSpeechBackend, WindowsVoicesUnavailable
            with self._native_lock:
                if self._native is None:
                    self._native = WindowsSpeechBackend()
                try:
                    self._native.start()
                except WindowsVoicesUnavailable:
                    self._native_unavailable = True
                    if self.backend == 'windows':
                        raise
                    logging.getLogger(__name__).warning('中文离线音色不可用，回退在线语音')

    def close(self):
        if self._native:
            self._native.close()

    def detect_language(self, text: str) -> str:
        """粗判语言：按汉字/拉丁字母数量决定中英文音色（langid 仅作兜底）。"""
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
        """异步合成一段文本到 mp3 文件（未指定音色时自动选择中/英文音色）。"""
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
        communicate = edge_tts.Communicate(text.strip(), selected_voice, connect_timeout=3, receive_timeout=5)
        await communicate.save(str(output))
        return output

    def synthesize_sync(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
        retries: int = 0,
        cancel_event=None,
    ) -> Path:
        """网络抖动时自动重试（edge-tts 走微软在线服务，偶发连接超时）。"""
        if not text.strip():
            raise ValueError('TTS 文本不能为空')
        if self.backend in {'auto', 'windows'} and not self._native_unavailable:
            self.warmup()
            if self._native and not self._native_unavailable:
                from .windows_tts import WindowsVoicesUnavailable
                try:
                    return self._native.synthesize(text, output_path, self.detect_language(text), self.timeout_seconds, cancel_event)
                except WindowsVoicesUnavailable:
                    if self.backend == 'windows':
                        raise
                    output_path = Path(output_path).with_suffix('.mp3')
        last_error: BaseException | None = None
        for attempt in range(retries + 1):
            try:
                return self._synthesize_sync_once(text, output_path, voice, cancel_event)
            except SpeechCancelled:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    delay = .3 * (attempt + 1)
                    if cancel_event:
                        if cancel_event.wait(delay):
                            raise SpeechCancelled()
                    else:
                        time.sleep(delay)
        assert last_error is not None
        raise last_error

    def _synthesize_sync_once(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
        cancel_event=None,
    ) -> Path:
        """同步版单次合成：若当前线程已有事件循环，就放到子线程里跑 asyncio.run。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._bounded_synthesize(text, output_path, voice, cancel_event))
        result: list[Path] = []
        error: list[BaseException] = []

        def runner():
            try:
                result.append(asyncio.run(self._bounded_synthesize(text, output_path, voice, cancel_event)))
            except BaseException as exc:
                error.append(exc)

        import threading

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return result[0]

    async def _bounded_synthesize(self, text, output_path, voice, cancel_event):
        async def run():
            task = asyncio.create_task(self.synthesize(text, output_path, voice))
            try:
                while not task.done():
                    if cancel_event and cancel_event.is_set():
                        raise SpeechCancelled()
                    await asyncio.wait({task}, timeout=.05)
                return await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        try:
            return await asyncio.wait_for(run(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f'语音合成超时（单次上限 {self.timeout_seconds:g} 秒），请检查网络') from exc

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """按中英文标点/换行切句，用于流式播放（让首句尽快出声）。"""
        parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text)
        return [part.strip() for part in parts if part.strip()]


class SentenceBuffer:
    """把 LLM 流的任意片段攒成完整句子，供 TTS 分句播放。"""

    def __init__(self, max_chars: int = 80):
        self.pending = ""
        self.max_chars = max_chars

    def feed(self, chunk: str) -> list[str]:
        """喂入一个新片段；只交付「最后一个完整句末」之前的部分。"""
        self.pending += chunk
        matches = list(re.finditer(r"[。！？!?；;\n]", self.pending))
        result = []
        if matches:
            end = matches[-1].end()
            complete, self.pending = self.pending[:end], self.pending[end:]
            result.extend(TTSEngine.split_sentences(complete))
        while len(self.pending) >= self.max_chars:
            soft = list(re.finditer(r'[，,、：: ]', self.pending[:self.max_chars]))
            end = soft[-1].end() if soft and soft[-1].end() >= self.max_chars // 2 else self.max_chars
            result.append(self.pending[:end].strip())
            self.pending = self.pending[end:]
        return [sentence for sentence in result if sentence]

    def flush(self) -> list[str]:
        """流结束：把剩余未成句的尾巴也交付出去。"""
        remaining = self.pending.strip()
        self.pending = ""
        return [remaining] if remaining else []
