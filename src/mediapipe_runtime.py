"""Make legacy MediaPipe graph/model resources readable from CJK Windows paths."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)
_RESOURCE_LOCK = threading.RLock()


def _copy_resources(package_dir: Path, cache_root: Path) -> None:
    source_dir = package_dir / "modules"
    if not source_dir.is_dir():
        raise RuntimeError("MediaPipe 模型资源缺失，请安装 requirements.txt 中指定的版本")
    for source in source_dir.rglob("*"):
        if not source.is_file() or source.suffix not in {".binarypb", ".tflite", ".txt", ".pbtxt"}:
            continue
        destination = cache_root / "mediapipe" / source.relative_to(package_dir)
        source_stat = source.stat()
        if destination.is_file():
            dest_stat = destination.stat()
            if dest_stat.st_size == source_stat.st_size and dest_stat.st_mtime_ns == source_stat.st_mtime_ns:
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
            staging = Path(handle.name)
        try:
            shutil.copy2(source, staging)
            staging.replace(destination)
        finally:
            staging.unlink(missing_ok=True)


@contextmanager
def compatible_resources(mediapipe_module):
    package_dir = Path(mediapipe_module.__file__).resolve().parent
    if os.name != "nt" or str(package_dir).isascii():
        yield
        return
    from mediapipe.python import solution_base

    key = hashlib.sha256(str(package_dir).encode("utf-8")).hexdigest()[:12]
    default_root = Path(tempfile.gettempdir()) / "mmai_mediapipe" / key
    cache_root = Path(os.environ.get("MMAI_MEDIAPIPE_CACHE", str(default_root))).resolve()
    if not str(cache_root).isascii():
        raise RuntimeError("请将 MMAI_MEDIAPIPE_CACHE 设置为不含中文的可写目录")
    with _RESOURCE_LOCK:
        _copy_resources(package_dir, cache_root)
        # SolutionBase resets the native resource directory on every constructor,
        # deriving it from this module attribute. Redirect just that derivation
        # while constructing graphs; do not edit any installed package files.
        original_file = solution_base.__file__
        solution_base.__file__ = str(cache_root / "mediapipe" / "python" / "solution_base.py")
        try:
            LOGGER.info("MediaPipe 使用兼容资源目录: %s", cache_root)
            yield
        finally:
            solution_base.__file__ = original_file
