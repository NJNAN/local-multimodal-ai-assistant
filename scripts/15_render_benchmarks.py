from pathlib import Path
import argparse
import json
from _bootstrap import PROJECT_ROOT
from src.benchmarking import render_results, render_negative, render_resume, write_visuals


def main():
    parser = argparse.ArgumentParser(description="Generate evidence from raw benchmark JSON")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "benchmarks/results/latest.json")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    (PROJECT_ROOT / "RESULTS.md").write_text(render_results(data), encoding="utf-8")
    (PROJECT_ROOT / "NEGATIVE-RESULTS.md").write_text(render_negative(data), encoding="utf-8")
    (PROJECT_ROOT / "RESUME.md").write_text(render_resume(data), encoding="utf-8")
    write_visuals(data, args.input.parent)
    print("Generated RESULTS.md, NEGATIVE-RESULTS.md, overview.svg, dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
