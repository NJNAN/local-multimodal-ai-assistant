"""离屏渲染真实 Qt 界面，使用示例对话检查布局，不加载推理模型或硬件。"""
from _bootstrap import PROJECT_ROOT
import os
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
# Windows 下需先初始化 PyTorch，再加载 Qt DLL。
from config import PATHS
from PyQt5.QtWidgets import QApplication
from src.qt_runtime import configure_qt_plugins
from src.assistant_service import AssistantService
from src.modal_router import ModalRouter
from src.gui import MainWindow
from src.hotword_dialog import HotwordDialog


def main():
    configure_qt_plugins()
    app = QApplication([])
    video = SimpleNamespace(recent_frames=lambda count: [], stop=lambda: None)
    rag = SimpleNamespace(vector_store=SimpleNamespace(documents=[]))
    service = AssistantService(SimpleNamespace(), rag, SimpleNamespace(), ModalRouter(), video)
    window = MainWindow(service, video, output_dir=PATHS['outputs'] / 'ui_preview')
    window.show()
    directory = PROJECT_ROOT / 'assets' / 'screenshots'
    directory.mkdir(exist_ok=True, parents=True)

    def render(widget, filename):
        for _ in range(10):
            app.processEvents()
            time.sleep(.015)
        widget.grab().save(str(directory / filename))

    render(window, 'chat-welcome.png')
    window._show_user_query('你好，我想了解多模态 AI 助手可以帮我做什么？')
    answer = '你好！我可以和你聊问题，也可以帮你读懂图片和资料。\n\n你可以这样开始：\n• 直接输入问题，或用麦克风提问\n• 上传图片，让我帮你识别内容\n• 导入课程资料，带着资料一起讨论\n\n如果专业词听错了，可以在左侧添加自定义语音热词。'
    window._append_assistant_chunk(answer)
    window._assistant_done(answer)
    window._show_user_query('“通义千问”经常被听成“同义千问”，可以自己添加纠正吗？')
    answer = '可以。打开“自定义语音热词”，点击“添加热词”：\n\n标准词：通义千问\n常见误识别词：同义千问\n\n保存后会立即生效，下次启动也会保留。'
    window._append_assistant_chunk(answer)
    window._assistant_done(answer)
    render(window, 'chat-redesign.png')
    window.resize(1000, 700)
    render(window, 'chat-compact.png')
    dialog = HotwordDialog(window.hotwords, window)
    dialog.show()
    render(dialog, 'hotword-editor.png')
    dialog.close()
    window.close()
    window._query_thread.join(3)
    print('Rendered four Qt previews with illustrative conversation content.')


if __name__ == '__main__':
    main()
