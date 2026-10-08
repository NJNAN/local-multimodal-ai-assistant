"""GUI 与业务逻辑之间的信号桥模块。

``Bridge`` 是 PyQt5 项目里 UI 线程与后台工作线程之间解耦用的统一信号中心
——后台（语音、视觉、LLM、知识库等）只负责 ``emit`` 对应信号，UI 层只负
责 ``connect``，谁也不直接持有对方的引用。这样：

- 后台工作线程里 ``bridge.frame_ready.emit(frame)``，UI 主线程在
  ``bridge.frame_ready.connect(update_label)`` 处由 Qt 自动切到主线程；
- 单测或无 PyQt5 的脚本里仍然可以 ``bridge.frame_ready.emit(...)`` 并通
  过 ``bridge.frame_ready.callbacks`` 验证回调是否被注册。

为了在没有安装 PyQt5 的环境（例如纯命令行单元测试、无头 CI、纯语音脚本）
下仍能 ``import`` 本文件，本模块做了 ImportError 的双实现 fallback：

- 有 PyQt5 时：``Bridge`` 继承 ``QObject``，所有信号都是 ``pyqtSignal``，
  会被 Qt 的事件循环线程切换正确地投递到主线程；
- 没有 PyQt5 时：``Bridge`` 是普通类，每个信号退化为 ``_Signal`` 实例，
  提供 ``connect(callback)`` / ``emit(*args)``，但**不在**线程间切换
  ——调用方需要自己保证线程安全。

信号清单见类注释。被主程序实例化并注册到各业务模块；GUI 通过
``bridge.<signal_name>.connect(...)`` 订阅。
"""
from __future__ import annotations

try:
    from PyQt5.QtCore import QObject, pyqtSignal

    class Bridge(QObject):
        """PyQt5 信号中心，继承 ``QObject`` 以便跨线程投递。

        每个属性都是一个 ``pyqtSignal``，后台 ``emit``、UI ``connect``。
        信号语义：
            - frame_ready(object): 摄像头捕获到一帧（已标注）时发射；
            - asr_result(str): 语音识别最终文本就绪；
            - assistant_chunk(str): LLM 流式输出片段；
            - assistant_done(str): LLM 完整回复结束（参数为完整文本）；
            - status(str): 任意状态文字（"正在识别" / "思考中" 等）；
            - error(str): 错误消息（可被 UI 用于红色横幅）；
            - gesture(str): 已确认的手势标签（已通过状态机稳定）；
            - mode_changed(str): 工作模式切换（语音 / 视觉 / 文本）；
            - knowledge_imported(str): 知识库导入完成（含路径）；
            - user_query(str): 用户原始查询（来自 GUI / 语音）；
            - response_details(object): LLM 调用的耗时 / token 等元信息；
            - request_started(object): 请求开始（含请求 id / 时间戳）；
            - detection_status(object): 视觉检测的状态（人数 / 关键点摘要）；
            - gesture_analysis(object): 手势分析的扩展结果（角度 / 候选）。
        """

        # 视频帧（标注好关键点后的 numpy ndarray）
        frame_ready = pyqtSignal(object)
        camera_state = pyqtSignal(object)
        # 语音识别最终文本
        asr_result = pyqtSignal(str)
        audio_status = pyqtSignal(str)
        # 大模型流式输出片段
        assistant_chunk = pyqtSignal(str)
        # 大模型完整回复
        assistant_done = pyqtSignal(str)
        # 任意状态文字（用于 UI 状态栏）
        status = pyqtSignal(str)
        # 错误消息
        error = pyqtSignal(str)
        # 已确认的手势标签（经过状态机稳定 + 冷却）
        gesture = pyqtSignal(str)
        # 工作模式切换
        mode_changed = pyqtSignal(str)
        # 知识库导入完成
        knowledge_imported = pyqtSignal(str)
        # 用户原始查询
        user_query = pyqtSignal(str)
        # 大模型调用的耗时 / token 等元信息
        response_details = pyqtSignal(object)
        # 请求开始（含 id / 时间戳）
        request_started = pyqtSignal(object)
        # 视觉检测状态
        detection_status = pyqtSignal(object)
        # 手势分析扩展结果
        gesture_analysis = pyqtSignal(object)

except ImportError:

    class _Signal:
        """无 PyQt5 时的回调信号占位实现。

        提供 ``connect(callback)`` / ``emit(*args)`` 两个方法，多个回调按
        注册顺序依次触发；``emit`` 期间用回调列表的副本遍历，允许回调内
        部 ``disconnect``（虽然本实现没有提供 ``disconnect``，但预留了防
        御性）。
        """

        def __init__(self):
            # 注册顺序即触发顺序；不提供去重，调用方自己负责
            self.callbacks = []

        def connect(self, callback):
            """注册一个回调到信号末尾。"""
            self.callbacks.append(callback)

        def emit(self, *args):
            """依次调用所有已注册回调；遍历时使用列表副本以允许回调内部修改。"""
            for callback in list(self.callbacks):
                callback(*args)

    class Bridge:
        """无 PyQt5 时的信号中心 fallback。

        用法与 PyQt5 版本一致：``bridge.frame_ready.emit(...)`` /
        ``bridge.frame_ready.connect(callback)``，但**不跨线程**，
        线程安全由调用方自行保证。
        """

        def __init__(self):
            # 在循环里统一创建 14 个 _Signal 实例，保持与 PyQt5 版本同构
            for name in (
                "frame_ready",
                "camera_state",
                "asr_result",
                "audio_status",
                "assistant_chunk",
                "assistant_done",
                "status",
                "error",
                "gesture",
                "mode_changed",
                "knowledge_imported",
                "user_query",
                "response_details",
                "request_started",
                "detection_status",
                "gesture_analysis",
            ):
                setattr(self, name, _Signal())
