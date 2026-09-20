from __future__ import annotations

from types import SimpleNamespace

from src.llama_backend import LlamaCppClient, LlamaServer, LlamaServerManager


class FakeServer:
    """In-memory stand-in for LlamaServer (no processes spawned)."""

    def __init__(self, name: str):
        self.name = name
        self.alias = f"{name}-alias"
        self.port = 18099
        self.server_path = "llama-server"
        self.model_path = f"C:/tmp/{name}.gguf"
        self.mmproj_path = None
        self.keep_alive_seconds = 0
        self.load_seconds = None
        self.device = ""
        self.running = False
        self.started = 0
        self.stopped = 0

    def is_running(self) -> bool:
        return self.running

    def missing_files(self) -> list[str]:
        return []

    def start(self) -> bool:
        self.running = True
        self.started += 1
        return True

    def wait_ready(self, timeout_s: float = 300, poll_s: float = 0.5) -> float:
        self.load_seconds = 0.05
        return self.load_seconds

    def stop(self, timeout_s: float = 15) -> None:
        self.running = False
        self.stopped += 1

    def props(self) -> dict:
        return {}


def _make_client(mode: str = "text"):
    server = FakeServer(mode)
    manager = SimpleNamespace(
        servers={mode: server},
        begin_request=lambda m: None,
        end_request=lambda m: None,
        ensure=lambda m: server,
    )
    config = {
        mode: {"alias": server.alias, "disable_thinking": mode == "text"},
        "request_timeout": 5,
    }
    return LlamaCppClient(manager, mode, config=config), server


def test_convert_messages_keeps_text_only_messages():
    client, _ = _make_client()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "你好"},
    ]
    assert client._convert_messages(messages) == messages


def test_convert_messages_turns_images_into_data_uri_parts():
    client, _ = _make_client("vision")
    messages = [
        {"role": "user", "content": "看图", "images": ["AAAA"]},
    ]
    converted = client._convert_messages(messages)
    parts = converted[0]["content"]
    assert parts[0] == {"type": "text", "text": "看图"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"


def test_build_body_maps_options_and_thinking_flag():
    client, _ = _make_client("text")
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "think": False,
        "options": {"temperature": 0.4, "num_predict": 64},
    }
    body = client._build_body(payload, stream=False)
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.4
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["stream"] is False


def test_vision_client_does_not_force_thinking_off_by_default():
    client, _ = _make_client("vision")
    body = client._build_body({"messages": []}, stream=True)
    assert "chat_template_kwargs" not in body
    assert body["stream"] is True


def test_llama_server_build_command_includes_mmproj_fit_and_device():
    server = LlamaServer(
        "vision",
        "C:/llama/llama-server.exe",
        "C:/models/model.gguf",
        mmproj_path="C:/models/mmproj.gguf",
        port=8081,
        n_ctx=4096,
        alias="vl",
        device="Vulkan1",
        fit=True,
        fit_target_mb=1024,
    )
    cmd = server.build_command()
    joined = " ".join(cmd)
    assert "--mmproj" in cmd and "mmproj.gguf" in joined
    assert "--fit" in cmd and "on" in cmd
    assert "--device" in cmd and "Vulkan1" in cmd
    assert cmd[cmd.index("-c") + 1] == "4096"
    assert cmd[cmd.index("--alias") + 1] == "vl"


def test_llama_server_missing_files_reported():
    server = LlamaServer("text", "Z:/missing/llama-server.exe", "Z:/missing/model.gguf")
    missing = server.missing_files()
    assert len(missing) == 2
    assert any("llama-server" in item for item in missing)
    assert any("模型文件" in item for item in missing)


def test_manager_exclusive_eviction_stops_other_model():
    text = FakeServer("text")
    vision = FakeServer("vision")
    vision.running = True
    manager = LlamaServerManager({"text": text, "vision": vision}, exclusive=True, start_timeout=5)
    try:
        manager.ensure("text")
        assert vision.stopped == 1
        assert text.started == 1
        assert text.running
    finally:
        manager.stop_all()
