from src.text_llm import TextLLM


class FakeClient:
    def list_models(self):
        return ["qwen:test"]

    def chat(self, payload):
        return {"message": {"content": "测试回答"}}

    def stream_chat(self, payload):
        yield {"message": {"content": "测"}}
        yield {"message": {"content": "试"}, "done": True}


def test_generate_and_stream():
    llm = TextLLM("qwen:test", client=FakeClient())
    assert llm.health()["ok"]
    assert llm.generate("你好") == "测试回答"
    assert "".join(llm.stream_generate("你好")) == "测试"


def test_kb_prompt_contains_context():
    llm = TextLLM("qwen:test", client=FakeClient())
    messages = llm._build_messages("问题", "校内资料", True)
    assert "校内资料" in messages[0]["content"]
