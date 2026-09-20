from __future__ import annotations

SYSTEM_PROMPT = """你是多模态AI助手。你通过语音、文本、摄像头画面和本地知识库帮助用户。
回答应友好、专业、准确，优先使用简洁自然的中文。不要声称看到了未提供的画面，也不要编造知识库内容。
不确定时明确说明不确定，并给出可验证的下一步。"""


class TextLLM:
    """Local llama.cpp (llama-server) text model with token streaming.

    The public API is unchanged from the previous Ollama-backed version.
    ``base_url`` / ``keep_alive`` / ``num_ctx`` remain accepted for
    backward compatibility, but the runtime now comes from
    ``config.LLAMA_CONFIG`` (Qwen3.5-4B IQ4_XS served by llama-server) or
    from an explicitly injected ``client``.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 300,
        keep_alive: str = "10m",
        num_ctx: int = 8192,
        client=None,
    ):
        self.model_name = model_name
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        if client is None:
            from .llama_backend import create_text_client

            client = create_text_client()
        self.client = client

    def health(self) -> dict:
        models = self.client.list_models()
        return {
            "ok": self.model_name in models,
            "model": self.model_name,
            "available_models": models,
        }

    def _build_messages(
        self,
        user_text: str,
        kb_context: str = "",
        kb_mode: bool = False,
        history: list[dict] | None = None,
    ) -> list[dict]:
        system = SYSTEM_PROMPT
        if kb_mode:
            context = kb_context.strip() or "（当前没有检索到相关资料）"
            system += (
                "\n\n回答时优先依据下面的本地参考资料。资料不足时先说明资料不足，"
                "再明确区分地使用通用知识补充。\n【参考资料】\n" + context
            )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-12:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _build_payload(
        self,
        user_text: str,
        kb_context: str = "",
        kb_mode: bool = False,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(user_text, kb_context, kb_mode, history),
            "keep_alive": self.keep_alive,
            "think": False,
            "options": {"temperature": 0.4, "num_ctx": self.num_ctx},
        }
        if max_new_tokens:
            payload["options"]["num_predict"] = max_new_tokens
        return payload

    def generate(
        self,
        user_text: str,
        kb_context: str = "",
        kb_mode: bool = False,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ) -> str:
        payload = self._build_payload(user_text, kb_context, kb_mode, max_new_tokens, history)
        response = self.client.chat(payload)
        return str(response.get("message", {}).get("content", "")).strip()

    def stream_generate(
        self,
        user_text: str,
        kb_context: str = "",
        kb_mode: bool = False,
        max_new_tokens: int | None = None,
        history: list[dict] | None = None,
    ):
        payload = self._build_payload(user_text, kb_context, kb_mode, max_new_tokens, history)
        for event in self.client.stream_chat(payload):
            content = event.get("message", {}).get("content", "")
            if content:
                yield content
