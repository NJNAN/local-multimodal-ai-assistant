from types import SimpleNamespace

from src.gesture import GestureRecognizer
from src.gesture_actions import GestureActionMapper


def palm_landmarks():
    points = [SimpleNamespace(x=0.5, y=0.8, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.9, z=0)
    points[2] = SimpleNamespace(x=0.45, y=0.72, z=0)
    points[3] = SimpleNamespace(x=0.35, y=0.62, z=0)
    points[4] = SimpleNamespace(x=0.2, y=0.52, z=0)
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        points[pip] = SimpleNamespace(x=0.5, y=0.55, z=0)
        points[tip] = SimpleNamespace(x=0.5, y=0.3, z=0)
    return points


def test_open_palm_and_cooldown():
    now = [10.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    assert recognizer.recognize(palm_landmarks()) == "open_palm"
    assert recognizer.recognize(palm_landmarks()) is None
    now[0] += 2.1
    assert recognizer.recognize(palm_landmarks()) == "open_palm"


def test_all_gesture_actions_mapped():
    mapper = GestureActionMapper()
    assert all(mapper.map(name) != "unknown" for name in GestureRecognizer.GESTURES)
