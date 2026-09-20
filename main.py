from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

from config import (
    AUDIO_CONFIG,
    DEVICE,
    LLAMA_CONFIG,
    LOG_CONFIG,
    MODEL_CONFIG,
    PATHS,
    TTS_CONFIG,
    VIDEO_CONFIG,
    set_seed,
)
from src.asr import ASREngine
from src.assistant_service import AssistantService
from src.audio_capture import AudioCapture
from src.audio_thread import AudioStreamThread
from src.bridge import Bridge
from src.gesture import GestureRecognizer
from src.gesture_actions import GestureActionMapper
from src.knowledge_base import VectorStore
from src.llama_backend import (
    create_text_client,
    create_vision_client,
    get_manager,
    shutdown,
)
from src.logging_utils import configure_logging
from src.modal_router import ModalRouter
from src.rag import RAGPipeline
from src.text_llm import TextLLM
from src.tts import TTSEngine
from src.tts_player import StreamingTTSPlayer
from src.vad import VADProcessor
from src.video_capture import VideoCapture, VideoThread
from src.vl_model import VLModel


def build_components():
    set_seed()
    configure_logging(LOG_CONFIG["file"], LOG_CONFIG["level"])
    bridge = Bridge()
    manager = get_manager()
    manager.on_status = bridge.status.emit
    text_llm = TextLLM(MODEL_CONFIG["text"], client=create_text_client())
    vector_store = VectorStore(MODEL_CONFIG["embedding"])
    rag = RAGPipeline(vector_store, text_llm)
    index_dir = PATHS["knowledge"] / ".index"
    try:
        loaded = vector_store.load(index_dir)
        if loaded:
            logging.getLogger(__name__).info("已加载知识库索引: %d 个知识块", loaded)
    except Exception:
        logging.getLogger(__name__).exception("知识库索引加载失败，可在界面中重新导入")
    video_capture = VideoCapture(**VIDEO_CONFIG)
    vl_model = VLModel(MODEL_CONFIG["vision"], client=create_vision_client())
    service = AssistantService(text_llm, rag, vl_model, ModalRouter(), video_capture)
    tts = TTSEngine(
        PATHS["temp"], TTS_CONFIG["zh_voice"], TTS_CONFIG["en_voice"]
    )
    player = StreamingTTSPlayer(tts, bridge.status.emit, playback=TTS_CONFIG["enabled"])
    return bridge, service, video_capture, player


def health_check() -> int:
    """检查 llama.cpp 运行时、两个模型文件，并做一次文本模型装载测试。"""
    manager = get_manager()
    report = manager.describe()
    labels = {"text": "文本模型 Qwen3.5-4B (IQ4_XS)", "vision": "视觉模型 Qwen3-VL-4B (Q4_K_M)"}
    ok = True
    for mode in ("text", "vision"):
        info = report[mode]
        if info["missing"]:
            ok = False
            print(f"[失败] {labels[mode]}")
            for item in info["missing"]:
                print(f"       {item}")
        else:
            extra = f"，mmproj {info['mmproj']}" if info["mmproj"] else ""
            print(f"[通过] {labels[mode]}: {info['model']}{extra}")
    server = LLAMA_CONFIG["server"]
    print(f"llama-server: {server or '未找到（请安装 llama.cpp 或设置 MMAI_LLAMA_SERVER）'}")
    print(f"计算设备: {DEVICE}")
    if not server or not ok:
        return 1
    try:
        loaded = manager.ensure("text")
        print(f"[通过] 文本模型装载测试: {loaded.load_seconds:.1f}s（端口 {loaded.port}）")
    except Exception as exc:
        print(f"[失败] 文本模型装载测试: {exc}")
        return 1
    finally:
        shutdown()
    return 0


def _prewarm_asr(asr: ASREngine) -> None:
    """后台加载 ASR 并热身，让第一次说话的识别几乎无等待。"""
    try:
        asr.warmup()
        logging.getLogger(__name__).info("ASR 预热完成")
    except Exception:
        logging.getLogger(__name__).exception("ASR 预热失败（不影响使用）")


def _prewarm_text_model() -> None:
    """启动几秒后后台预载文本模型；若有模型已在使用中则跳过。"""
    try:
        time.sleep(5.0)
        manager = get_manager()
        if manager.any_running():
            return
        logging.getLogger(__name__).info("后台预热文本模型…")
        manager.ensure("text")
    except Exception:
        logging.getLogger(__name__).exception("文本模型预热失败（不影响按需加载）")


def main() -> int:
    parser = argparse.ArgumentParser(description="本地多模态AI助手")
    parser.add_argument("--health-check", action="store_true", help="只检查核心模型")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="无窗口后端启动并自动退出")
    args = parser.parse_args()
    if args.health_check:
        return health_check()

    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        args.no_audio = True
        args.no_video = True

    from PyQt5.QtWidgets import QApplication

    from src.gui import MainWindow

    app = QApplication(sys.argv)
    bridge, service, video_capture, player = build_components()
    video_thread = None
    audio_thread = None
    detector = None
    startup_errors: list[str] = []

    if not args.no_video:
        try:
            video_capture.start()
            video_thread = VideoThread(video_capture, bridge.frame_ready.emit)
        except Exception as exc:
            startup_errors.append(f"摄像头不可用: {exc}")

    if not args.no_audio:
        try:
            audio_capture = AudioCapture(
                rate=AUDIO_CONFIG["rate"],
                chunk=AUDIO_CONFIG["chunk"],
                channels=AUDIO_CONFIG["channels"],
                input_device_index=AUDIO_CONFIG["input_device_index"],
            ).start()
            vad = VADProcessor(
                sample_rate=AUDIO_CONFIG["rate"],
                silence_duration_ms=AUDIO_CONFIG["silence_duration_ms"],
                min_speech_duration_ms=AUDIO_CONFIG["min_speech_duration_ms"],
                aggressiveness=AUDIO_CONFIG["vad_aggressiveness"],
            )
            asr = ASREngine(MODEL_CONFIG["asr"], DEVICE)
            threading.Thread(
                target=_prewarm_asr, args=(asr,), daemon=True, name="asr-prewarm"
            ).start()
            audio_thread = AudioStreamThread(
                audio_capture,
                vad,
                asr,
                bridge.asr_result.emit,
                lambda exc: bridge.error.emit(str(exc)),
                PATHS["outputs"],
            )
        except Exception as exc:
            startup_errors.append(f"麦克风不可用: {exc}")

    try:
        from src.human_detector import HumanDetector

        detector = HumanDetector(model_complexity=1)  # 复杂度1检测约46ms/帧（复杂度2约94ms），配合采集线程更流畅
    except Exception as exc:
        startup_errors.append(f"人体检测不可用: {exc}")

    window = MainWindow(
        service,
        video_capture,
        audio_thread=audio_thread,
        video_thread=video_thread,
        detector=detector,
        gesture_recognizer=GestureRecognizer(),
        action_mapper=GestureActionMapper(),
        tts_player=player,
        output_dir=PATHS["outputs"],
        bridge=bridge,
    )
    window.show()
    for message in startup_errors:
        bridge.error.emit(message)
    player.start()
    if video_thread:
        video_thread.start()
    if audio_thread:
        audio_thread.start()
    if not args.smoke_test:
        threading.Thread(
            target=_prewarm_text_model, daemon=True, name="llama-prewarm"
        ).start()
    if args.smoke_test:
        from PyQt5.QtCore import QTimer

        QTimer.singleShot(1500, window.close)
    try:
        code = app.exec_()
    finally:
        # 保证退出时不留任何 llama-server 子进程
        shutdown()
    if args.smoke_test:
        print("GUI smoke test passed")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
