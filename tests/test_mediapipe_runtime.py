from src.mediapipe_runtime import _copy_resources


def test_cache_contains_graph_models_and_handedness_labels(tmp_path):
    package = tmp_path / "中文环境" / "mediapipe"
    modules = package / "modules" / "hand_landmark"
    modules.mkdir(parents=True)
    files = {"graph.binarypb": b"graph", "model.tflite": b"model", "handedness.txt": b"Left\nRight"}
    for name, data in files.items():
        (modules / name).write_bytes(data)
    (modules / "__init__.py").write_text("", encoding="utf-8")
    cache = tmp_path / "cache"
    _copy_resources(package, cache)
    for name, data in files.items():
        destination = cache / "mediapipe" / "modules" / "hand_landmark" / name
        assert destination.read_bytes() == data
    assert not (cache / "mediapipe/modules/hand_landmark/__init__.py").exists()
    # An existing cache is reused, while a changed bundled resource is refreshed.
    (modules / "handedness.txt").write_bytes(b"updated labels")
    _copy_resources(package, cache)
    assert (cache / "mediapipe/modules/hand_landmark/handedness.txt").read_bytes() == b"updated labels"


def test_real_detector_initializes_from_unicode_installation_and_restores_module_path():
    import pytest
    import numpy as np

    mp = pytest.importorskip("mediapipe")
    from mediapipe.python import solution_base
    from src.human_detector import HumanDetector

    original_file = solution_base.__file__
    detector = HumanDetector(model_complexity=1)
    try:
        assert solution_base.__file__ == original_file
        result = detector.process_frame(np.zeros((240, 320, 3), dtype="uint8"))
        assert set(result) == {"hands", "face", "pose"}
        assert not result["hands"].multi_hand_landmarks
    finally:
        detector.close()
