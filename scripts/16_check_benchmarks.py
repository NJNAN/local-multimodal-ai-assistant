from pathlib import Path
import argparse
import json
from _bootstrap import PROJECT_ROOT
from src.benchmarking import render_results, render_negative, render_resume, sha256, check_answer
from src.inference_metrics import response_measurements


def validate(data, root):
    errors = []
    for filename, render in (("RESULTS.md", render_results), ("NEGATIVE-RESULTS.md", render_negative), ("RESUME.md", render_resume)):
        path = root / filename
        if not path.exists() or path.read_text(encoding="utf-8") != render(data):
            errors.append(f"{filename}: report drift; regenerate from raw data")
    cases = root / "benchmarks/cases.json"
    if sha256(cases) != data["environment"]["cases_sha256"]:
        errors.append("case suite hash differs from measured suite")
        return errors
    try:
        case_data = json.loads(cases.read_text(encoding="utf-8"))
    except ValueError:
        return errors + ["case suite is not valid JSON"]
    measured_files = {"src/llama_backend.py", "src/inference_options.py", "src/inference_metrics.py", "src/benchmarking.py", "scripts/14_inference_bench.py"}
    for name, digest in data["environment"].get("source_sha256", {}).items():
        path = root / name
        if name not in measured_files or not path.is_file() or sha256(path) != digest:
            errors.append(f"{name}: measured source drift; rerun experiments")
    ids = set()
    for p in data["profiles"]:
        if p["id"] in ids:
            errors.append("duplicate profile id")
        ids.add(p["id"])
        rows = p.get("requests", [])
        if p["status"] == "ok" and not rows:
            errors.append(f"{p['id']}: empty successful profile")
        suite = {case["id"]: case for case in case_data.get(p.get("mode"), [])}
        if p["status"] == "ok" and len(rows) != len(suite) * data["repeats"]:
            errors.append(f"{p['id']}: incomplete case coverage")
        pairs = set()
        for r in rows:
            pair = (r["case"], r["repeat"])
            if pair in pairs or r["repeat"] not in range(data["repeats"]):
                errors.append(f"{p['id']}: duplicate or invalid repeat")
            pairs.add(pair)
            if r["case"] not in suite or r["correct"] != check_answer(r["answer"], suite[r["case"]]):
                errors.append(f"{p['id']}: quality score differs from actual answer")
            if r.get("output_tokens") is None or r["output_tokens"] < 0:
                errors.append(f"{p['id']}: missing token counts")
            if r["ttft_seconds"] is None or not 0 <= r["ttft_seconds"] <= r["e2e_seconds"]:
                errors.append(f"{p['id']}: invalid stream timing")
            derived = response_measurements(r.get("timings", {}), r.get("usage", {}), r["e2e_seconds"], r["ttft_seconds"])
            for key in ("output_tokens", "tpot_ms", "throughput_tokens_s", "cached_tokens"):
                if r.get(key) != derived[key]:
                    errors.append(f"{p['id']}: {key} differs from raw timing/usage")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Fail on published benchmark/data drift")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "benchmarks/results/latest.json")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    errors = validate(data, PROJECT_ROOT)
    for error in errors:
        print(error)
    if not errors:
        print("Evidence checker passed: reports, source hashes, answer scores, coverage and stream timings")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
