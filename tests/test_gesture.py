import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.gesture import GestureRecognizer
from src.gesture_actions import GestureActionMapper


def point(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def transform(points, degrees=0, scale=1, mirror=False, aspect_ratio=1):
    angle = math.radians(degrees)
    result = []
    for p in points:
        x, y = p.x - 0.5, p.y - 0.5
        if mirror:
            x = -x
        result.append(point(0.5 + scale * (x * math.cos(angle) - y * math.sin(angle)),
                            (0.5 + scale * (x * math.sin(angle) + y * math.cos(angle))) * aspect_ratio,
                            p.z * scale))
    return result


def hand_for_gesture(gesture):
    if gesture in {'thumbs_down', 'index_down'}:
        return transform(hand_for_gesture(gesture.replace('down', 'up')), degrees=180)
    points = [point(0.5, 0.8) for _ in range(21)]
    points[0] = point(0.5, 0.9)
    points[1:5] = [point(0.42, 0.8), point(0.34, 0.73), point(0.41, 0.68), point(0.45, 0.74)]
    extended = {'open_palm': {5, 9, 13, 17}, 'victory': {5, 9}, 'index_up': {5}}.get(gesture, set())
    for mcp, x in ((5, 0.38), (9, 0.47), (13, 0.56), (17, 0.64)):
        points[mcp] = point(x, 0.70)
        heights = (0.56, 0.45, 0.34) if mcp in extended else (0.60, 0.67, 0.74)
        points[mcp + 1:mcp + 4] = [point(x, y) for y in heights]
    if gesture == 'open_palm':
        points[2:5] = [point(0.34, 0.73), point(0.25, 0.66), point(0.16, 0.59)]
    elif gesture == 'thumbs_up':
        points[2:5] = [point(0.29, 0.70), point(0.29, 0.54), point(0.29, 0.38)]
    return points


def palm_landmarks():
    return hand_for_gesture('open_palm')


def observe(recognizer, now, gesture, frames=5, step=0.1):
    events = []
    for _ in range(frames):
        now[0] += step
        event = recognizer.recognize(hand_for_gesture(gesture) if gesture else None)
        if event:
            events.append(event)
    return events


def test_all_gesture_actions_mapped():
    mapper = GestureActionMapper()
    assert all(mapper.map(name) != 'unknown' for name in GestureRecognizer.GESTURES)


@pytest.mark.parametrize('gesture', GestureRecognizer.GESTURES)
def test_six_gestures_and_analysis_does_not_consume_action_cooldown(gesture):
    now = [0.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    points = hand_for_gesture(gesture)
    assert recognizer.classify(points) == gesture
    assert recognizer.classify(points) == gesture
    assert recognizer.last_gesture is None
    assert recognizer.recognize(points) is None
    assert observe(recognizer, now, gesture) == [gesture]
    assert observe(recognizer, now, gesture, frames=50) == []


@pytest.mark.parametrize('gesture', GestureRecognizer.GESTURES)
@pytest.mark.parametrize('aspect_ratio,mirror,scale', [(4/3, False, 1), (16/9, True, 0.4), (3/4, True, 1.2)])
def test_geometry_accounts_for_frame_dimensions_mirroring_and_hand_size(gesture, aspect_ratio, mirror, scale):
    points = transform(hand_for_gesture(gesture), mirror=mirror, scale=scale, aspect_ratio=aspect_ratio)
    assert GestureRecognizer().classify(points, aspect_ratio=aspect_ratio) == gesture


@pytest.mark.parametrize('gesture', ['victory', 'open_palm'])
@pytest.mark.parametrize('degrees', [45, 90, 180, 270])
def test_victory_and_palm_can_rotate_without_becoming_directional_gestures(gesture, degrees):
    assert GestureRecognizer().classify(transform(hand_for_gesture(gesture), degrees=degrees)) == gesture


@pytest.mark.parametrize('gesture', ['thumbs_up', 'thumbs_down', 'index_up', 'index_down'])
def test_horizontal_pointing_does_not_execute_vertical_actions(gesture):
    assert GestureRecognizer().classify(transform(hand_for_gesture(gesture), degrees=90)) is None


def test_requires_multiple_frames_and_elapsed_time():
    now = [0.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    assert observe(recognizer, now, 'thumbs_down', frames=20, step=0.01) == []
    assert observe(recognizer, now, 'thumbs_down', frames=3) == ['thumbs_down']


def test_brief_alternating_labels_never_trigger_any_action():
    now = [0.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    for _ in range(10):
        for gesture in ('thumbs_down', 'victory', 'index_up', 'thumbs_up', 'open_palm', None):
            assert observe(recognizer, now, gesture, frames=1) == []
    assert recognizer.last_gesture is None


def test_held_gesture_cannot_launch_other_actions_after_cooldown():
    now = [0.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    assert observe(recognizer, now, 'thumbs_down') == ['thumbs_down']
    for gesture in ('victory', 'index_up', 'thumbs_up', 'open_palm', 'thumbs_down'):
        assert observe(recognizer, now, gesture, frames=10) == []
    assert recognizer.confirmed_gesture == 'thumbs_down'
    assert observe(recognizer, now, None, frames=1) == []
    assert observe(recognizer, now, 'thumbs_down', frames=10) == []
    assert observe(recognizer, now, None, frames=8) == []
    assert recognizer.confirmed_gesture is None
    assert observe(recognizer, now, 'victory') == ['victory']


def test_cooldown_applies_to_all_actions_even_after_release():
    now = [0.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    assert observe(recognizer, now, 'thumbs_down') == ['thumbs_down']
    observe(recognizer, now, None, frames=8)
    assert observe(recognizer, now, 'victory', frames=5) == []
    assert observe(recognizer, now, 'victory', frames=10) == ['victory']


def test_tracking_gap_requires_new_confirmation_and_cannot_fake_release():
    now = [0.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    assert observe(recognizer, now, 'thumbs_down', frames=3) == []
    now[0] += 2
    assert observe(recognizer, now, 'thumbs_down', frames=3) == []
    assert observe(recognizer, now, 'thumbs_down', frames=2) == ['thumbs_down']
    assert observe(recognizer, now, None, frames=1) == []
    now[0] += 2
    assert observe(recognizer, now, None, frames=1) == []
    assert observe(recognizer, now, 'victory', frames=10) == []


def test_reset_discards_partial_confirmation():
    now = [0.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    observe(recognizer, now, 'victory', frames=3)
    recognizer.reset()
    assert observe(recognizer, now, 'victory', frames=3) == []
    assert observe(recognizer, now, 'victory', frames=2) == ['victory']


def test_folded_fist_is_not_index_down_and_ambiguous_fingers_are_not_victory():
    recognizer = GestureRecognizer()
    assert recognizer.classify(hand_for_gesture('fist')) is None
    points = hand_for_gesture('victory')
    # Sideways curled fingertips can sit above the PIP in camera coordinates.
    points[6], points[7], points[8] = point(0.35, 0.60), point(0.42, 0.58), point(0.44, 0.53)
    points[10], points[11], points[12] = point(0.46, 0.60), point(0.53, 0.58), point(0.55, 0.53)
    assert recognizer.classify(points) is None


@pytest.mark.parametrize('value', [None, [], [point(0, 0)] * 21, [point(float('nan'), 0)] * 21])
def test_invalid_or_degenerate_landmarks_do_not_execute_actions(value):
    assert GestureRecognizer().classify(value) is None


REGRESSION_SAMPLES = json.loads((Path(__file__).parent / 'fixtures/gesture_regression.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('sample', REGRESSION_SAMPLES, ids=[f"{p['expected']}_{i}" for i, p in enumerate(REGRESSION_SAMPLES)])
def test_landmarks_from_real_false_trigger_photos(sample):
    points = [point(*p) for p in sample['landmarks']]
    assert GestureRecognizer().classify(points, aspect_ratio=sample['aspect_ratio']) == sample['expected']


def test_real_thumbs_down_sequence_never_triggers_photo_or_microphone():
    now = [0.0]
    recognizer = GestureRecognizer(clock=lambda: now[0])
    events = []
    down = [p for p in REGRESSION_SAMPLES if p['expected'] == 'thumbs_down']
    for _ in range(20):
        for sample in down:
            now[0] += 0.1
            event = recognizer.recognize([point(*p) for p in sample['landmarks']], aspect_ratio=sample['aspect_ratio'])
            if event:
                events.append(event)
    assert events == ['thumbs_down']
