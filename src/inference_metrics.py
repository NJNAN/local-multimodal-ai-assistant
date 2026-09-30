"""Prometheus counters and reproducible inference measurements (no GUI deps)."""
from __future__ import annotations

import json
import math
import re


def parse_prometheus(text: str) -> list[dict]:
    samples = []
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Za-z_:][\w:]*)(?:\{(.*?)\})?\s+([^\s]+)(?:\s+\d+)?', line.strip())
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = {key: json.loads('"' + escaped + '"') for key, escaped in
                  re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', raw_labels or "")}
        samples.append({"name": name, "labels": labels, "value": value})
    return samples


def counter_delta(before: dict, after: dict) -> dict:
    """Missing/reset counters remain unknown, never fabricated as zero."""
    def keyed(data):
        return {(p["name"], json.dumps(p["labels"], sort_keys=True)): p for p in data.get("samples", [])}
    previous, current = keyed(before), keyed(after)
    values = []
    for key, sample in current.items():
        old = previous.get(key)
        delta = sample["value"] - old["value"] if old else None
        values.append({**sample, "delta": delta if delta is not None and delta >= 0 else None})
    def get(name):
        found = [p["delta"] for p in values if p["name"] == name and not p["labels"]]
        return found[0] if found else None
    draft = get("llamacpp:spec_decode_num_draft_tokens_total")
    accepted = get("llamacpp:spec_decode_num_accepted_tokens_total")
    return {"draft_tokens": draft, "accepted_tokens": accepted,
            "acceptance_rate": accepted / draft if draft and accepted is not None else None,
            "accepted_per_position": [p for p in values if p["name"] == "llamacpp:spec_decode_num_accepted_tokens_per_pos_total"],
            "available": bool(before.get("available") and after.get("available"))}


def response_measurements(timings: dict, usage: dict, elapsed: float, first: float | None) -> dict:
    tokens = timings.get("predicted_n", usage.get("completion_tokens"))
    return {"e2e_seconds": elapsed, "ttft_seconds": first,
            "output_tokens": tokens,
            "throughput_tokens_s": tokens / elapsed if tokens is not None and elapsed > 0 else None,
            "tpot_ms": (elapsed - first) * 1000 / (tokens - 1) if first is not None and tokens and tokens > 1 else None,
            "server_decode_tokens_s": timings.get("predicted_per_second"),
            "prefill_ms": timings.get("prompt_ms"), "cached_tokens": timings.get("cache_n"),
            "timings": timings, "usage": usage}
