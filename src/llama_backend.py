"""llama.cpp (llama-server) backend for the text and vision models.

The project replaced its Ollama backend with llama.cpp because both models
now ship as GGUF:

* text   : Qwen3.5-4B IQ4_XS      (bartowski GGUF)
* vision : Qwen3-VL-4B-Instruct   (Q4_K_M + Q8_0 mmproj, Qwen official)

Upper layers (TextLLM / VLModel / GUI / RAG / TTS) stay untouched: this
module provides

* :class:`LlamaServer`        – one ``llama-server`` process.
* :class:`LlamaServerManager` – lifecycle for both processes. By default
  only one heavy model is resident at a time, protecting the 8 GB GPU.
* :class:`LlamaCppClient`     – talks to llama-server over local HTTP but
  exposes the same small protocol the old ``OllamaClient`` had
  (``list_models`` / ``show_model`` / ``chat`` / ``stream_chat``), so
  TextLLM and VLModel need no protocol changes.

llama-server API surface used: ``/health``, ``/props``, ``/v1/models``,
``/v1/chat/completions`` (streaming SSE; the vision model additionally
accepts ``image_url`` data-URI content parts backed by the mmproj).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_DEVICE_CACHE: dict[str, str] = {}
_MANAGER: "LlamaServerManager | None" = None
_MANAGER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _config() -> dict:
    """Late import so helper functions stay importable without full config."""
    from config import LLAMA_CONFIG

    return LLAMA_CONFIG


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    # Localhost model traffic must never be routed through a system proxy.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _tail(path: str | Path | None, lines: int = 15) -> str:
    if not path:
        return ""
    log_path = Path(path)
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def detect_auto_device(server_path: str) -> str:
    """Return a ``--device`` hint when several inference devices exist.

    This laptop exposes both an Intel iGPU and the NVIDIA dGPU through the
    Vulkan backend, and llama.cpp would otherwise be free to pick the iGPU.
    When exactly one device is visible (e.g. a CUDA build) no hint is
    needed, so an empty string is returned.
    """
    if not server_path:
        return ""
    if server_path in _DEVICE_CACHE:
        return _DEVICE_CACHE[server_path]
    device = ""
    try:
        result = subprocess.run(
            [server_path, "--list-devices"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        lines = [
            line.strip()
            for line in (result.stdout or "").splitlines()
            if line.strip().startswith(("Vulkan", "CUDA", "ROCm", "SYCL", "Metal"))
        ]
        if len(lines) > 1:
            for line in lines:
                if "NVIDIA" in line.upper():
                    device = line.split(":", 1)[0].strip()
                    break
    except (OSError, subprocess.SubprocessError):
        device = ""
    _DEVICE_CACHE[server_path] = device
    return device


# ---------------------------------------------------------------------------
# one llama-server process
# ---------------------------------------------------------------------------


class LlamaServer:
    """A single llama-server process serving one GGUF model."""

    def __init__(
        self,
        name: str,
        server_path: str,
        model_path: str | Path,
        *,
        mmproj_path: str | Path | None = None,
        port: int = 8080,
        n_ctx: int = 8192,
        alias: str = "",
        n_gpu_layers: str = "auto",
        device: str = "",
        fit: bool = True,
        fit_target_mb: int = 1024,
        keep_alive_seconds: int = 0,
        log_path: str | Path | None = None,
    ):
        self.name = name
        self.server_path = server_path or ""
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path) if mmproj_path else None
        self.port = int(port)
        self.n_ctx = int(n_ctx)
        self.alias = alias or self.model_path.stem
        self.n_gpu_layers = str(n_gpu_layers)
        self.device = device
        self.fit = fit
        self.fit_target_mb = int(fit_target_mb)
        self.keep_alive_seconds = int(keep_alive_seconds)
        self.log_path = Path(log_path) if log_path else None
        self.proc: subprocess.Popen | None = None
        self.load_seconds: float | None = None
        self._started_at: float | None = None
        self._log_handle = None
        self._lock = threading.RLock()

    # -- introspection -------------------------------------------------------

    def missing_files(self) -> list[str]:
        missing: list[str] = []
        if not self.server_path or not Path(self.server_path).is_file():
            missing.append(f"llama-server 可执行文件未找到 ({self.server_path or '未配置'})")
        if not self.model_path.is_file():
            missing.append(f"模型文件缺失: {self.model_path}")
        if self.mmproj_path and not self.mmproj_path.is_file():
            missing.append(f"mmproj 文件缺失: {self.mmproj_path}")
        return missing

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def build_command(self) -> list[str]:
        cmd = [
            self.server_path,
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(self.n_ctx),
            "--alias",
            self.alias,
            "--metrics",
            "--no-webui",
            "-ngl",
            self.n_gpu_layers,
        ]
        if self.fit:
            cmd += ["--fit", "on", "--fit-target", str(self.fit_target_mb)]
        if self.device:
            cmd += ["--device", self.device]
        if self.mmproj_path:
            cmd += ["--mmproj", str(self.mmproj_path)]
        return cmd

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        with self._lock:
            if self.is_running():
                return False
            missing = self.missing_files()
            if missing:
                raise FileNotFoundError("；".join(missing))
            cmd = self.build_command()
            log_handle = None
            log_target: Any = subprocess.DEVNULL
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = open(self.log_path, "ab")
                log_target = log_handle
            LOGGER.info("[llama.cpp] 启动 %s 模型: %s", self.name, " ".join(cmd))
            self._started_at = time.perf_counter()
            self.proc = subprocess.Popen(
                cmd,
                stdout=log_target,
                stderr=subprocess.STDOUT,
                cwd=str(Path(self.server_path).parent) if self.server_path else None,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._log_handle = log_handle
            return True

    def wait_ready(self, timeout_s: float = 300, poll_s: float = 0.5) -> float:
        """Block until ``/health`` reports ok; returns the load time in seconds."""
        deadline = time.monotonic() + timeout_s
        opener = _no_proxy_opener()
        while time.monotonic() < deadline:
            if self.proc is None:
                raise RuntimeError(f"{self.name}: llama-server 未启动")
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name}: llama-server 提前退出 (exit={self.proc.returncode})。"
                    f"日志({self.log_path}):\n{_tail(self.log_path)}"
                )
            try:
                with opener.open(f"{self.base_url}/health", timeout=3) as response:
                    body = json.load(response)
                if isinstance(body, dict) and body.get("status") == "ok":
                    self.load_seconds = time.monotonic() - (self._started_at or time.monotonic())
                    return self.load_seconds
            except urllib.error.HTTPError:
                pass  # 503 while the model is still loading → keep polling
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(poll_s)
        raise TimeoutError(
            f"{self.name}: llama-server 在 {timeout_s:.0f}s 内未就绪。"
            f"日志({self.log_path}):\n{_tail(self.log_path)}"
        )

    def stop(self, timeout_s: float = 15) -> None:
        with self._lock:
            proc = self.proc
            if proc is not None and proc.poll() is None:
                LOGGER.info("[llama.cpp] 停止 %s 模型 (pid=%s)", self.name, proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    LOGGER.warning("[llama.cpp] %s 未在 %ss 内退出，强制结束", self.name, timeout_s)
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            self.proc = None
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except OSError:
                    pass
                self._log_handle = None

    def props(self) -> dict:
        """Model metadata reported by llama-server (build, ftype, vision...)."""
        if not self.is_running():
            return {}
        try:
            return _http_json(f"{self.base_url}/props", timeout=5)
        except (urllib.error.URLError, OSError, ValueError):
            return {}


def _http_json(url: str, payload: dict | None = None, timeout: float = 60) -> Any:
    opener = _no_proxy_opener()
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


# ---------------------------------------------------------------------------
# lifecycle manager (text + vision share one manager)
# ---------------------------------------------------------------------------


class LlamaServerManager:
    """Lifecycle management for the text and vision llama-server processes."""

    def __init__(
        self,
        servers: dict[str, LlamaServer],
        *,
        exclusive: bool = True,
        start_timeout: float = 300,
    ):
        self.servers = servers
        self.exclusive = exclusive
        self.start_timeout = float(start_timeout)
        self._lock = threading.RLock()
        self._last_use: dict[str, float] = {}
        self._inflight: dict[str, int] = {mode: 0 for mode in servers}
        self._shutdown = threading.Event()
        self._closing = False
        self.on_status = None
        self._idle_thread = threading.Thread(
            target=self._idle_loop, name="llama-idle-monitor", daemon=True
        )
        self._idle_thread.start()
        atexit.register(self.stop_all)

    # -- request bookkeeping -------------------------------------------------

    def begin_request(self, mode: str) -> None:
        with self._lock:
            self._inflight[mode] = self._inflight.get(mode, 0) + 1

    def end_request(self, mode: str) -> None:
        with self._lock:
            self._inflight[mode] = max(0, self._inflight.get(mode, 0) - 1)
            self._last_use[mode] = time.time()

    # -- loading / eviction --------------------------------------------------

    def ensure(self, mode: str) -> LlamaServer:
        """Ensure ``mode`` is ready to serve; evict the other model if needed."""
        server = self.servers[mode]
        with self._lock:
            if self._closing:
                raise RuntimeError("llama.cpp 服务正在关闭，本次请求已取消")
            if self.exclusive:
                for other_mode, other in self.servers.items():
                    if (
                        other_mode != mode
                        and other.is_running()
                        and not server.is_running()
                    ):
                        LOGGER.info("[llama.cpp] 切换模型：先停止 %s 释放显存", other_mode)
                        self._notify(f"切换模型：正在加载{'文本' if mode == 'text' else '视觉'}模型（约 5~8 秒）…")
                        other.stop()
                        # give the driver a moment to reclaim VRAM
                        time.sleep(0.8)
            if not server.is_running():
                label = "文本" if mode == "text" else "视觉"
                self._notify(f"正在加载{label}模型（首次约 5 秒）…")
                self._reclaim_port(server)
                server.start()
                load_seconds = server.wait_ready(self.start_timeout)
                LOGGER.info("[llama.cpp] %s 模型加载完成，耗时 %.1fs", mode, load_seconds)
                self._notify(f"{label}模型就绪，正在生成…")
            self._last_use[mode] = time.time()
        return server

    def stop_all(self) -> None:
        self._shutdown.set()
        with self._lock:
            for server in self.servers.values():
                try:
                    server.stop()
                except Exception:  # noqa: BLE001 – shutdown must never raise
                    LOGGER.exception("[llama.cpp] 停止 %s 时出错", server.name)
        try:
            self._idle_thread.join(timeout=2)
        except RuntimeError:
            pass

    def describe(self) -> dict:
        """Current state of both servers, used by health checks and scripts."""
        return {
            mode: {
                "alias": server.alias,
                "model": str(server.model_path),
                "mmproj": str(server.mmproj_path) if server.mmproj_path else None,
                "server": server.server_path,
                "port": server.port,
                "running": server.is_running(),
                "missing": server.missing_files(),
                "load_seconds": server.load_seconds,
            }
            for mode, server in self.servers.items()
        }

    # -- status / shutdown helpers -------------------------------------------

    def any_running(self) -> bool:
        with self._lock:
            return any(server.is_running() for server in self.servers.values())

    def begin_shutdown(self) -> None:
        """Refuse new model requests while the app is exiting."""
        with self._lock:
            self._closing = True

    def _notify(self, message: str) -> None:
        callback = self.on_status
        if callback:
            try:
                callback(message)
            except Exception:  # noqa: BLE001 – UI 回调不允许影响推理
                pass

    def _port_in_use(self, server: LlamaServer) -> bool:
        try:
            _http_json(f"{server.base_url}/health", timeout=2)
            return True
        except urllib.error.HTTPError:
            return True  # 503（模型加载中）等 HTTP 响应也说明端口被占用
        except Exception:
            return False

    @staticmethod
    def _listening_pid(port: int) -> str | None:
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                timeout=15,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return None
        suffix = f":{port}"
        for line in (result.stdout or "").splitlines():
            parts = line.split()
            if (
                len(parts) >= 5
                and parts[0].upper() == "TCP"
                and parts[1].endswith(suffix)
                and parts[3].upper() == "LISTENING"
            ):
                return parts[4]
        return None

    def _reclaim_port(self, server: LlamaServer) -> None:
        """端口被上次异常退出的 llama-server 占用时先回收（仅限本项目端口）。"""
        if not self._port_in_use(server):
            return
        pid = self._listening_pid(server.port)
        if not pid:
            return
        try:
            info = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=15,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return
        if "llama-server" not in (info.stdout or "").lower():
            return
        LOGGER.warning(
            "[llama.cpp] 端口 %s 被残留的 llama-server (pid=%s) 占用，先回收",
            server.port,
            pid,
        )
        try:
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass
        time.sleep(0.5)

    # -- idle keep-alive -----------------------------------------------------

    def _idle_loop(self) -> None:
        while not self._shutdown.wait(5):
            with self._lock:
                for mode, server in self.servers.items():
                    keep_alive = server.keep_alive_seconds
                    if keep_alive <= 0 or not server.is_running():
                        continue
                    if self._inflight.get(mode, 0) > 0:
                        continue
                    last = self._last_use.get(mode)
                    if last and (time.time() - last) > keep_alive:
                        LOGGER.info(
                            "[llama.cpp] %s 模型空闲超过 %ss，自动卸载释放显存", mode, keep_alive
                        )
                        server.stop()


# ---------------------------------------------------------------------------
# client — same protocol the old OllamaClient exposed
# ---------------------------------------------------------------------------


class LlamaCppClient:
    """Client protocol compatible with the previous ``OllamaClient``.

    ``TextLLM`` and ``VLModel`` build Ollama-style payloads; this client
    translates them onto llama-server's OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        manager: LlamaServerManager,
        mode: str,
        *,
        timeout: float | None = None,
        config: dict | None = None,
    ):
        config = config or _config()
        cfg = config[mode]
        self.manager = manager
        self.mode = mode
        self.server = manager.servers[mode]
        self.model_name = cfg.get("alias", mode)
        self._disable_thinking = bool(cfg.get("disable_thinking", False))
        self.timeout = float(timeout or config.get("request_timeout", 300))
        self._opener = _no_proxy_opener()

    # -- availability --------------------------------------------------------

    def _availability(self) -> tuple[bool, str]:
        missing = self.server.missing_files()
        if missing:
            return False, "；".join(missing)
        return True, ""

    def list_models(self) -> list[str]:
        ok, _ = self._availability()
        return [self.model_name] if ok else []

    def show_model(self, name: str | None = None) -> dict:
        ok, reason = self._availability()
        capabilities = ["completion"] + (["vision"] if self.mode == "vision" else [])
        payload: dict[str, Any] = {
            "model": name or self.model_name,
            "capabilities": capabilities if ok else [],
            "reason": reason,
            "modalities": {"vision": self.mode == "vision"},
        }
        if ok and self.server.is_running():
            props = self.server.props()
            if props:
                payload["build_info"] = props.get("build_info")
                payload["model_ftype"] = props.get("model_ftype")
        return payload

    # -- chat ----------------------------------------------------------------

    def chat(self, payload: dict) -> dict:
        self.manager.begin_request(self.mode)
        try:
            self.manager.ensure(self.mode)
            body = self._build_body(payload, stream=False)
            data = self._post("/v1/chat/completions", body)
            content = ""
            choices = data.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content") or ""
            return {
                "message": {"content": content},
                "timings": data.get("timings"),
                "usage": data.get("usage"),
            }
        finally:
            self.manager.end_request(self.mode)

    def stream_chat(self, payload: dict):
        self.manager.begin_request(self.mode)
        try:
            self.manager.ensure(self.mode)
            body = self._build_body(payload, stream=True)
            for event in self._post_stream("/v1/chat/completions", body):
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    yield {"message": {"content": content}, "done": False}
                timings = event.get("timings")
                if timings:
                    # final chunk carries timings; surface them for metrics
                    yield {
                        "message": {"content": ""},
                        "done": choice.get("finish_reason") is not None,
                        "timings": timings,
                    }
        finally:
            self.manager.end_request(self.mode)

    # -- request translation -------------------------------------------------

    def _build_body(self, payload: dict, *, stream: bool) -> dict:
        options = payload.get("options") or {}
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._convert_messages(payload.get("messages") or []),
            "stream": stream,
            "cache_prompt": True,
        }
        temperature = options.get("temperature")
        if temperature is not None:
            body["temperature"] = float(temperature)
        num_predict = options.get("num_predict")
        if num_predict:
            body["max_tokens"] = int(num_predict)
        if self._disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list[dict]:
        """Ollama-style messages → OpenAI-style messages (images as data URIs)."""
        converted: list[dict] = []
        for message in messages:
            role = message.get("role", "user")
            images = message.get("images") or []
            content = message.get("content", "")
            if not images:
                converted.append(
                    {"role": role, "content": content if content is not None else ""}
                )
                continue
            parts: list[dict] = []
            if isinstance(content, str) and content.strip():
                parts.append({"type": "text", "text": content})
            for image in images:
                data = str(image)
                if not data.startswith("data:"):
                    data = f"data:image/jpeg;base64,{data}"
                parts.append({"type": "image_url", "image_url": {"url": data}})
            converted.append({"role": role, "content": parts})
        return converted

    # -- HTTP ----------------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.server.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except OSError:
                pass
            raise RuntimeError(f"llama-server 返回 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"无法连接 llama-server ({self.server.base_url})：{exc.reason}；"
                f"日志: {self.server.log_path}"
            ) from exc

    def _post_stream(self, path: str, body: dict):
        request = urllib.request.Request(
            f"{self.server.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        yield json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except OSError:
                pass
            raise RuntimeError(f"llama-server 返回 HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"无法连接 llama-server ({self.server.base_url})：{exc.reason}；"
                f"日志: {self.server.log_path}"
            ) from exc


# ---------------------------------------------------------------------------
# factories (shared singleton manager for text + vision)
# ---------------------------------------------------------------------------


def _build_servers(config: dict) -> dict[str, LlamaServer]:
    servers: dict[str, LlamaServer] = {}
    for mode in ("text", "vision"):
        cfg = config[mode]
        device = cfg.get("device") or detect_auto_device(config.get("server") or "")
        servers[mode] = LlamaServer(
            name=mode,
            server_path=config.get("server") or "",
            model_path=cfg["model"],
            mmproj_path=cfg.get("mmproj"),
            port=cfg["port"],
            n_ctx=cfg["n_ctx"],
            alias=cfg["alias"],
            n_gpu_layers=cfg["n_gpu_layers"],
            device=device,
            fit=config.get("fit", True),
            fit_target_mb=config.get("fit_target_mb", 1024),
            keep_alive_seconds=cfg.get("keep_alive_seconds", 0),
            log_path=Path(config.get("log_dir") or Path.cwd()) / f"llama_server_{mode}.log",
        )
    return servers


def get_manager(config: dict | None = None) -> LlamaServerManager:
    """Return the process-wide manager (created on first use)."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            config = config or _config()
            _MANAGER = LlamaServerManager(
                _build_servers(config),
                exclusive=config.get("exclusive", True),
                start_timeout=config.get("start_timeout", 300),
            )
    return _MANAGER


def create_text_client(config: dict | None = None) -> LlamaCppClient:
    config = config or _config()
    return LlamaCppClient(
        get_manager(config), "text", timeout=config.get("request_timeout"), config=config
    )


def create_vision_client(config: dict | None = None) -> LlamaCppClient:
    config = config or _config()
    return LlamaCppClient(
        get_manager(config), "vision", timeout=config.get("request_timeout"), config=config
    )


def shutdown() -> None:
    """Stop every llama-server managed by this process (safe to call twice)."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            _MANAGER.begin_shutdown()
            _MANAGER.stop_all()


def _reset_manager_for_tests() -> None:
    """Testing helper: drop the singleton without leaking processes."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            _MANAGER.stop_all()
        _MANAGER = None
