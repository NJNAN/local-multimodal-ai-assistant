"""本地多模态 AI 助手的主入口：装配组件、启动界面与后台线程。

启动流程：
  1. set_seed(42) + 日志初始化；
  2. 装配组件：文本模型 / 向量知识库（RAG）/ 视觉模型 / 模态路由 / TTS 播放器；
  3. 启动摄像头、麦克风（采集→VAD→ASR）与 MediaPipe 人体手势检测——
     任一硬件缺失只降级并提示，不影响其余功能（异常隔离）；
  4. 创建主窗口、连接信号、启动后台线程，并在几秒后后台预热文本模型；
  5. 退出时统一 shutdown()，保证不留 llama-server 子进程。

常用命令行：
  python main.py                        # 完整界面
  python main.py --health-check         # 只做模型健康检查（不打开界面）
  python main.py --smoke-test           # 无窗口自检（1.5 秒后自动退出）
  python main.py --no-audio --no-video  # 无麦克风/摄像头环境
"""
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
from src.hotwords import HotwordStore
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
    """按固定顺序装配全部组件，返回 (bridge, service, video_capture, player)。

    先建信号桥与模型客户端；知识库索引本地存在时尝试加载，
    失败只记日志、不阻止启动（可在界面里重新导入）。
    """
    set_seed()
    configure_logging(LOG_CONFIG["file"], LOG_CONFIG["level"])
    bridge = Bridge()
    manager = get_manager()
    # 把模型管理器（按需加载/切换/卸载）的状态变化转成界面信号
    manager.on_status = bridge.status.emit
    text_llm = TextLLM(MODEL_CONFIG["text"], client=create_text_client())
    vector_store = VectorStore(MODEL_CONFIG["embedding"])
    index_dir = PATHS["knowledge"] / ".index"
    rag = RAGPipeline(vector_store, text_llm, index_dir=index_dir)
    try:
        loaded = vector_store.load(index_dir)
        if loaded:
            logging.getLogger(__name__).info("已加载知识库索引: %d 个知识块", loaded)
    except Exception:
        # 索引损坏/不兼容时降级启动：记日志即可，界面里可重新导入生成
        logging.getLogger(__name__).exception("知识库索引加载失败，可在界面中重新导入")
    video_capture = VideoCapture(**VIDEO_CONFIG)
    vl_model = VLModel(MODEL_CONFIG["vision"], client=create_vision_client())
    service = AssistantService(text_llm, rag, vl_model, ModalRouter(), video_capture)
    tts = TTSEngine(
        PATHS["temp"], TTS_CONFIG["zh_voice"], TTS_CONFIG["en_voice"],
        timeout_seconds=TTS_CONFIG['timeout_seconds'],
        backend=TTS_CONFIG['backend'],
    )
    player = StreamingTTSPlayer(tts, bridge.status.emit, playback=TTS_CONFIG["enabled"])
    return bridge, service, video_capture, player


def health_check() -> int:
    """检查 llama.cpp 运行时、两个模型文件，并做一次文本模型装载测试。

    不打开界面（供 --health-check 与部署前自检使用）；装载测试结束后
    立即 shutdown() 释放子进程。返回 0 = 全部通过。
    """
    manager = get_manager()
    report = manager.describe()
    labels = {"text": "文本模型 Qwen3.5-4B (IQ4_XS)", "vision": "视觉模型 Qwen3-VL-4B (Q4_K_M)"}
    ok = True
    # 先做「文件存在性」检查：缺哪个文件就报哪个，不启动任何进程
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
        # 真正启动一次文本模型，验证「能装载起来」而不只是文件存在
        loaded = manager.ensure("text")
        print(f"[通过] 文本模型装载测试: {loaded.load_seconds:.1f}s（端口 {loaded.port}）")
    except Exception as exc:
        print(f"[失败] 文本模型装载测试: {exc}")
        return 1
    finally:
        shutdown()
    return 0


def _prewarm_asr(asr: ASREngine, bridge=None) -> None:
    """后台加载 ASR 并热身，让第一次说话的识别几乎无等待。"""
    try:
        if bridge:
            bridge.audio_status.emit('正在准备语音识别…')
        asr.warmup()
        logging.getLogger(__name__).info("ASR 预热完成")
        if bridge:
            bridge.audio_status.emit('语音监听已开启')
    except Exception:
        logging.getLogger(__name__).exception("ASR 预热失败（不影响使用）")
        if bridge:
            bridge.audio_status.emit('语音准备失败，可继续打字')


def _prewarm_text_model() -> None:
    """启动几秒后后台预载文本模型；若有模型已在使用中则跳过。"""
    try:
        time.sleep(5.0)
        manager = get_manager()
        # 用户可能已经开始切到视觉模式：避免此时再抢显存，直接跳过
        if manager.any_running():
            return
        logging.getLogger(__name__).info("后台预热文本模型…")
        manager.ensure("text")
    except Exception:
        logging.getLogger(__name__).exception("文本模型预热失败（不影响按需加载）")


def main() -> int:
    """程序入口：解析参数、装配并启动整个应用；返回进程退出码。"""
    parser = argparse.ArgumentParser(description="本地多模态AI助手")
    parser.add_argument("--health-check", action="store_true", help="只检查核心模型")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="无窗口后端启动并自动退出")
    args = parser.parse_args()
    if args.health_check:
        return health_check()

    if args.smoke_test:
        # 冒烟模式：强制离屏渲染、禁用音视频，1.5 秒后自动关窗退出
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        args.no_audio = True
        args.no_video = True

    # Qt 相关导入放在参数分支之后：--health-check 路径完全不需要图形环境
    from PyQt5.QtWidgets import QApplication

    from src.gui import MainWindow
    from src.qt_runtime import configure_qt_plugins

    configure_qt_plugins()
    app = QApplication(sys.argv)
    bridge, service, video_capture, player = build_components()
    video_thread = None
    audio_thread = None
    hotwords = HotwordStore(PATHS['data'] / 'hotwords.json')
    detector = None
    detector_error = ""
    startup_errors: list[str] = []

    # 逐组件启动硬件：任何一项失败只记录错误，其余功能照常（异常隔离）
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
                max_segment_duration_ms=AUDIO_CONFIG['max_segment_duration_ms'],
            )
            asr = ASREngine(MODEL_CONFIG["asr"], DEVICE, hotwords=hotwords)
            # ASR 加载较慢，扔到后台线程热身，避免卡住启动
            threading.Thread(
                target=_prewarm_asr, args=(asr, bridge), daemon=True, name="asr-prewarm"
            ).start()
            audio_thread = AudioStreamThread(
                audio_capture,
                vad,
                asr,
                bridge.asr_result.emit,
                lambda exc: bridge.error.emit(str(exc)),
                PATHS["outputs"],
                on_status=bridge.audio_status.emit,
            )
            player.on_activity = audio_thread.set_suppressed
        except Exception as exc:
            startup_errors.append(f"麦克风不可用: {exc}")

    try:
        from src.human_detector import HumanDetector

        detector = HumanDetector(model_complexity=1)  # 复杂度1检测约46ms/帧（复杂度2约94ms），配合采集线程更流畅
    except Exception as exc:
        detector_error = str(exc)
        startup_errors.append(f"人体与手势检测不可用: {exc}")
        logging.getLogger(__name__).exception("人体与手势检测初始化失败")

    window = MainWindow(
        service,
        video_capture,
        audio_thread=audio_thread,
        video_thread=video_thread,
        detector=detector,
        detector_error=detector_error,
        gesture_recognizer=GestureRecognizer(),
        action_mapper=GestureActionMapper(),
        tts_player=player,
        output_dir=PATHS["outputs"],
        bridge=bridge,
        hotwords=hotwords,
    )
    window.show()
    # 把启动阶段的降级信息统一推给界面（横幅/状态栏提示）
    for message in startup_errors:
        bridge.error.emit(message)
    player.start()
    if video_thread:
        video_thread.start()
    if audio_thread:
        audio_thread.start()
    if not args.smoke_test:
        # 界面已就绪：后台预载文本模型，首个问题无需等待加载
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
