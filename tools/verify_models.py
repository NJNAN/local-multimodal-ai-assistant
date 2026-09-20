"""校验三个模型文件的 SHA256 是否与官方仓库一致（下载完成后运行）。

用法:

    D:\\mm_ai_env\\Scripts\\python.exe tools\\verify_models.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECT = {
    "models/qwen3.5-4b/Qwen_Qwen3.5-4B-IQ4_XS.gguf": (
        "628124c61a806897d3f28dcd416be6da",
        2671339488,
    ),
    "models/qwen3-vl-4b/Qwen3VL-4B-Instruct-Q4_K_M.gguf": (
        "66358cb18bb6b3b1b6675aa412c7a88e",
        2497281664,
    ),
    "models/qwen3-vl-4b/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf": (
        "30ba2c7dd3127a4561b6cba9d13d0f71",
        453974304,
    ),
}


def main() -> int:
    ok = True
    for relative, (prefix, size) in EXPECT.items():
        path = ROOT / relative
        name = Path(relative).name
        if not path.exists():
            print(f"[缺失]   {name}（尚未下载）")
            ok = False
            continue
        actual_size = path.stat().st_size
        if actual_size != size:
            print(f"[未完成] {name}: {actual_size}/{size} 字节（继续续传补齐）")
            ok = False
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        if value.startswith(prefix):
            print(f"[通过]   {name} (sha256 {value[:16]}…)")
        else:
            print(f"[校验失败] {name}: {value[:32]} != {prefix}（继续续传补齐）")
            ok = False
    print("全部通过 ✓" if ok else "存在未完成/失败项 ✗（重新运行下载脚本续传）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
