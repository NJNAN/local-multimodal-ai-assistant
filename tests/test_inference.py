from types import SimpleNamespace

import pytest

from src.inference_options import InferenceOptions
from src.inference_metrics import parse_prometheus, counter_delta, response_measurements
from src.benchmarking import check_answer, percentile, aggregate, render_results, render_negative, render_resume
from src.llama_backend import LlamaServer, LlamaServerManager


def test_default_options_preserve_runtime_and_opt_in_options_are_explicit():
    assert InferenceOptions().arguments(vision=False) == []
    options = InferenceOptions(cache_type_k="q8_0", cache_type_v="q4_0", spec_type="ngram-cache", parallel=1)
    args = options.arguments(vision=False)
    assert args[args.index("--spec-type") + 1] == "ngram-cache"
    assert "--lookup-cache-dynamic" not in args
    assert args[args.index("--cache-type-v") + 1] == "q4_0"


@pytest.mark.parametrize("values", [{"cache_type_k": "bad"}, {"spec_type": "draft-simple"}, {"flash_attn": "bad"},
                                   {"parallel": 0}, {"image_max_tokens": -1}, {"cache_reuse": -1},
                                   {"image_min_tokens": 1024, "image_max_tokens": 256}])
def test_invalid_options_fail_instead_of_silently_falling_back(values):
    with pytest.raises(ValueError):
        InferenceOptions(**values)


def test_image_budget_requires_vision_and_environment_is_validated(monkeypatch):
    options = InferenceOptions(image_min_tokens=64, image_max_tokens=256)
    with pytest.raises(ValueError):
        options.arguments(vision=False)
    assert "--image-max-tokens" in options.arguments(vision=True)
    monkeypatch.setenv("MMAI_TEST_CTK", "q8_0")
    monkeypatch.setenv("MMAI_TEST_PARALLEL", "2")
    assert InferenceOptions.from_environment("MMAI_TEST")["parallel"] == 2
    assert InferenceOptions.from_environment("MMAI_TEST")["cache_type_k"] == "q8_0"


def test_metrics_labels_counter_delta_and_reset_remain_unknown():
    before = {"available": True, "samples": parse_prometheus('''# ignored
llamacpp:spec_decode_num_draft_tokens_total 100
llamacpp:spec_decode_num_accepted_tokens_total 30
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 20
invalid NaN
''')}
    after = {"available": True, "samples": parse_prometheus('''llamacpp:spec_decode_num_draft_tokens_total 140
llamacpp:spec_decode_num_accepted_tokens_total 50
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 35''')}
    delta = counter_delta(before, after)
    assert delta["acceptance_rate"] == .5
    assert delta["accepted_per_position"][0]["delta"] == 15
    assert counter_delta(after, before)["acceptance_rate"] is None
    assert counter_delta({}, after)["draft_tokens"] is None
    assert counter_delta({}, {})["acceptance_rate"] is None


def test_timings_count_tokens_not_characters_and_do_not_guess_missing_fields():
    result = response_measurements({"predicted_n": 5, "cache_n": 20}, {}, 2, .4)
    assert result["output_tokens"] == 5
    assert result["tpot_ms"] == 400
    assert result["cached_tokens"] == 20
    assert result["throughput_tokens_s"] == 2.5
    assert response_measurements({}, {}, 1, .2)["tpot_ms"] is None
    assert response_measurements({}, {"completion_tokens": 1}, 1, .2)["tpot_ms"] is None


@pytest.mark.parametrize("answer,expected", [('```json\n{"count":2}\n```', True), ('{"count":2}', True),
                                           ('{"count":3}', False), ('说明：{"count":2}', False),
                                           ('{"count":2,"extra":1}', False)])
def test_quality_gate_is_exact_and_cannot_pass_on_answer_substring(answer, expected):
    assert check_answer(answer, {"expected": {"count": 2}}) is expected


def test_engine_usage_only_sse_chunk_is_retained(monkeypatch):
    from test_llama_backend import _make_client
    client, _ = _make_client()
    monkeypatch.setattr(client, "_post_stream", lambda *a: iter([
        {"choices": [{"delta": {"content": "answer"}}]},
        {"choices": [], "usage": {"completion_tokens": 7}, "timings": {"predicted_n": 7}}]))
    assert list(client.stream_chat({"messages": []}))[0]["message"]["content"] == "answer"
    assert client.last_usage["completion_tokens"] == 7
    assert client.last_timings["predicted_n"] == 7


def test_slot_basename_validation_and_inflight_guard(monkeypatch):
    server = LlamaServer("text", "engine.exe", "m.gguf", inference={"parallel": 1, "slot_save_path": "cache"})
    monkeypatch.setattr(server, "is_running", lambda: True)
    calls = []
    monkeypatch.setattr("src.llama_backend._http_json", lambda url, payload: calls.append((url, payload)) or {"n_saved": 10})
    for filename in ("../other.bin", "C:\\other.bin", "..", "dir/file.bin", ""):
        with pytest.raises(ValueError):
            server.slot_action("restore", filename)
    assert not calls
    with pytest.raises(ValueError):
        server.slot_action("save", "valid.bin", 1)
    manager = LlamaServerManager({"text": server})
    try:
        manager.begin_request("text")
        with pytest.raises(RuntimeError):
            manager.save_slot("text", "valid.bin")
        assert not calls
        manager.end_request("text")
        assert manager.save_slot("text", "valid.bin")["n_saved"] == 10
        assert calls[0][1] == {"filename": "valid.bin"}
    finally:
        manager.stop_all()


def test_unloaded_metrics_do_not_start_process(monkeypatch):
    server = LlamaServer("text", "engine.exe", "m.gguf")
    monkeypatch.setattr(server, "start", lambda: pytest.fail("monitor started a model"))
    assert server.metrics() == {"available": False, "reason": "model_unloaded"}


def test_percentiles_and_empty_profiles_do_not_invent_success():
    assert percentile([1, 3, None, 5], .5) == 3
    assert aggregate({"requests": []})["accuracy"] is None


def test_evidence_checker_detects_report_and_case_drift(tmp_path):
    import importlib.util
    from pathlib import Path
    from src.benchmarking import sha256
    checker_path = Path(__file__).parents[1] / "scripts/16_check_benchmarks.py"
    import sys
    monkey_path = str(checker_path.parent)
    sys.path.insert(0, monkey_path)
    try:
        spec = importlib.util.spec_from_file_location("bench_checker", checker_path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
    finally:
        sys.path.remove(monkey_path)
    cases = tmp_path / "benchmarks/cases.json"
    cases.parent.mkdir()
    cases.write_text("{}")
    data = {"created_at": "test", "repeats": 1, "profiles": [],
            "environment": {"gpu": "test", "server_version": "test", "cases_sha256": sha256(cases)}}
    (tmp_path / "RESULTS.md").write_text(render_results(data), encoding="utf-8")
    (tmp_path / "NEGATIVE-RESULTS.md").write_text(render_negative(data), encoding="utf-8")
    (tmp_path / "RESUME.md").write_text(render_resume(data), encoding="utf-8")
    assert checker.validate(data, tmp_path) == []
    (tmp_path / "RESULTS.md").write_text("invented result", encoding="utf-8")
    assert "report drift" in checker.validate(data, tmp_path)[0]
    cases.write_text("changed")
    assert any("hash differs" in e for e in checker.validate(data, tmp_path))


@pytest.mark.parametrize("tamper", ["correct", "throughput_tokens_s", "drop_request"])
def test_checker_recomputes_quality_and_timing_even_when_reports_match_tampered_json(tmp_path, tamper):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    root = Path(__file__).parents[1]
    scripts = str(root / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("bench_checker", root / "scripts/16_check_benchmarks.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
    finally:
        sys.path.remove(scripts)
    data = json.loads((root / "benchmarks/results/latest.json").read_text(encoding="utf-8"))
    for name in data["environment"].get("source_sha256", {}):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / name).read_bytes())
    cases = tmp_path / "benchmarks/cases.json"
    cases.parent.mkdir()
    cases.write_bytes((root / "benchmarks/cases.json").read_bytes())
    if tamper == "drop_request":
        data["profiles"][0]["requests"].pop()
    elif tamper == "correct":
        data["profiles"][0]["requests"][0]["correct"] = False
    else:
        data["profiles"][0]["requests"][0]["throughput_tokens_s"] += 10
    for name, render in (("RESULTS.md", render_results), ("NEGATIVE-RESULTS.md", render_negative), ("RESUME.md", render_resume)):
        (tmp_path / name).write_text(render(data), encoding="utf-8")
    errors = checker.validate(data, tmp_path)
    assert errors
    assert not any("report drift" in error for error in errors)
    assert not any("source drift" in error for error in errors)
    expected = {"correct": "quality score differs", "throughput_tokens_s": "differs from raw", "drop_request": "incomplete case coverage"}
    assert any(expected[tamper] in error for error in errors)
