from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Localhost model traffic must never be routed through a system proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"无法连接 Ollama ({self.base_url})，请确认 Ollama 已启动"
            ) from exc

    def list_models(self) -> list[str]:
        with self._request("/api/tags") as response:
            payload = json.load(response)
        return [item.get("name", "") for item in payload.get("models", [])]

    def show_model(self, name: str) -> dict:
        with self._request("/api/show", {"model": name}) as response:
            return json.load(response)

    def chat(self, payload: dict) -> dict:
        payload = {**payload, "stream": False}
        with self._request("/api/chat", payload) as response:
            return json.load(response)

    def stream_chat(self, payload: dict):
        payload = {**payload, "stream": True}
        with self._request("/api/chat", payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)
