"""回放本地录音检查断句上限，测量离线/在线合成与可选 ASR；不播放声音。"""
from _bootstrap import PROJECT_ROOT
import argparse
import json
import time
import wave
from pathlib import Path
from config import AUDIO_CONFIG, DEVICE, MODEL_CONFIG, PATHS
from src.vad import VADProcessor
from src.tts import TTSEngine


def replay(path, silence_ms, limit_ms):
    with wave.open(str(path)) as wav:
        pcm, rate = wav.readframes(wav.getnframes()), wav.getframerate()
    vad = VADProcessor(sample_rate=rate, silence_duration_ms=silence_ms,
                       max_segment_duration_ms=limit_ms)
    endpoints, segments = [], []
    for index in range(0, len(pcm) - vad.frame_bytes + 1, vad.frame_bytes):
        if vad.process_frame(pcm[index:index + vad.frame_bytes]):
            endpoints.append(round((index + vad.frame_bytes) / (rate * 2), 3))
            segment = vad.get_segment()
            if segment:
                segments.append(segment)
    if vad.flush():
        segment = vad.get_segment()
        if segment:
            segments.append(segment)
    return {'audio_seconds': round(len(pcm) / (rate * 2), 3), 'first_endpoint_seconds': endpoints[0] if endpoints else None,
            'segment_count': len(segments), 'max_segment_seconds': round(max((len(p) for p in segments), default=0) / (rate * 2), 3)}, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-model', action='store_true')
    parser.add_argument('--with-online', action='store_true')
    args = parser.parse_args()
    directory = PATHS['outputs'] / 'voice_latency_check'
    directory.mkdir(parents=True, exist_ok=True)
    report = {'endpoint_replay': [], 'tts': [], 'asr': []}
    segments = []
    for path in sorted(PATHS['outputs'].glob('test_asr_*.wav'))[-3:]:
        before, _ = replay(path, 600, 10_000_000)
        after, pieces = replay(path, AUDIO_CONFIG['silence_duration_ms'], AUDIO_CONFIG['max_segment_duration_ms'])
        segments.extend(pieces[:1])
        report['endpoint_replay'].append({'file': path.name, 'before': before, 'after': after})
    for backend in ['auto'] + (['online'] if args.with_online else []):
        engine = TTSEngine(directory, backend=backend)
        started = time.perf_counter()
        engine.warmup()
        startup = time.perf_counter() - started
        try:
            for index, text in enumerate(['你好，语音测试。', '首句生成后就可以开始朗读，不用再等整段回答。']):
                started = time.perf_counter()
                try:
                    path = engine.synthesize_sync(text, directory / f'{backend}_{index}{engine.audio_suffix}')
                    item = {'backend': backend, 'chars': len(text), 'seconds': round(time.perf_counter() - started, 4),
                            'bytes': path.stat().st_size, 'warmup_seconds': round(startup, 4)}
                except Exception as exc:
                    item = {'backend': backend, 'error': str(exc), 'seconds': round(time.perf_counter() - started, 4)}
                report['tts'].append(item)
        finally:
            engine.close()
    if args.with_model:
        from src.asr import ASREngine
        asr = ASREngine(MODEL_CONFIG['asr'], DEVICE)
        started = time.perf_counter()
        asr.warmup()
        report['asr_warmup_seconds'] = round(time.perf_counter() - started, 4)
        for pcm in segments:
            started = time.perf_counter()
            text = asr.transcribe(pcm)
            report['asr'].append({'audio_seconds': round(len(pcm) / 32000, 3),
                                  'recognition_seconds': round(time.perf_counter() - started, 4), 'text_chars': len(text)})
    (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
