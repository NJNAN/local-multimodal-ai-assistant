"""用系统已安装的中文/英文音色离线合成；复用一个隐藏的 SAPI 进程。"""
from __future__ import annotations
import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class WindowsVoicesUnavailable(RuntimeError):
    pass


class WindowsSpeechBackend:
    def __init__(self):
        self._process = None
        self._replies = queue.Queue()
        self._lock = threading.RLock()
        self.voices = {}

    def start(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                return
            if os.name != 'nt':
                raise WindowsVoicesUnavailable('当前系统没有 Windows 离线音色')
            self._replies = queue.Queue()
            self._process = subprocess.Popen(
                ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                 '-File', str(Path(__file__).with_suffix('.ps1'))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                encoding='utf-8', text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            process, replies = self._process, self._replies
            def reader():
                for line in process.stdout:
                    try:
                        replies.put(json.loads(line))
                    except ValueError:
                        LOGGER.warning('离线语音返回无效响应')
                replies.put({'ok': False, 'error': 'Local speech worker exited'})
            threading.Thread(target=reader, daemon=True, name='windows-speech-replies').start()
            try:
                self.voices = replies.get(timeout=3)
            except queue.Empty as exc:
                self.close()
                raise WindowsVoicesUnavailable('Windows 离线音色未能初始化') from exc
            if not self.voices.get('ready') or not self.voices.get('zh'):
                self.close()
                raise WindowsVoicesUnavailable('未找到已安装的中文离线音色')

    def synthesize(self, text, output_path, language, timeout=10, cancel_event=None):
        from .tts import SpeechCancelled
        with self._lock:
            self.start()
            if not self.voices.get(language):
                raise WindowsVoicesUnavailable(f'未安装 {language} 离线音色')
            output = Path(output_path).with_suffix('.wav').resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            request = {'text': text, 'path': str(output), 'language': language}
            self._process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
            self._process.stdin.flush()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if cancel_event and cancel_event.is_set():
                    self.close()
                    raise SpeechCancelled()
                try:
                    reply = self._replies.get(timeout=.05)
                except queue.Empty:
                    continue
                if not reply.get('ok'):
                    raise RuntimeError(reply.get('error', '离线语音合成失败'))
                LOGGER.info('Windows TTS %.3fs, voice=%s', reply['seconds'], reply['voice'])
                return output
            self.close()
            raise TimeoutError('离线语音合成超时')

    def close(self):
        with self._lock:
            if self._process:
                process, self._process = self._process, None
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=.5)
                if process.stdin:
                    process.stdin.close()
