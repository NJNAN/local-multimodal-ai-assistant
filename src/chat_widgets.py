"""用户聊天组件：左右消息气泡、独立流式回答、可复制文本和多行输入。"""
from html import escape
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                            QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)


class MessageInput(QPlainTextEdit):
    returnPressed = pyqtSignal()

    def text(self):
        return self.toPlainText()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & Qt.ShiftModifier:
            self.returnPressed.emit()
        else:
            super().keyPressEvent(event)


class MessageBubble(QWidget):
    def __init__(self, role, text, parent=None):
        super().__init__(parent)
        self.role, self.text = role, text
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(10)
        avatar = QLabel('我' if role == 'user' else 'AI')
        avatar.setObjectName('userAvatar' if role == 'user' else 'aiAvatar')
        avatar.setFixedSize(36, 36)
        avatar.setAlignment(Qt.AlignCenter)
        self.card = QFrame()
        self.card.setObjectName('userBubble' if role == 'user' else 'aiBubble')
        body = QVBoxLayout(self.card)
        body.setContentsMargins(16, 12, 16, 12)
        body.setSpacing(7)
        self.label = QLabel()
        self.label.setObjectName('messageText')
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.RichText)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        body.addWidget(self.label)
        self.meta = QLabel()
        self.meta.setTextFormat(Qt.PlainText)
        self.meta.setObjectName('messageMeta')
        self.meta.setWordWrap(True)
        self.meta.hide()
        body.addWidget(self.meta)
        if role == 'assistant':
            footer = QHBoxLayout()
            footer.addStretch()
            copy = QPushButton('复制')
            copy.setObjectName('copyButton')
            copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text))
            footer.addWidget(copy)
            body.addLayout(footer)
        if role == 'user':
            row.addStretch(1)
            row.addWidget(self.card)
            row.addWidget(avatar, 0, Qt.AlignTop)
        else:
            row.addWidget(avatar, 0, Qt.AlignTop)
            row.addWidget(self.card)
            row.addStretch(1)
        self.set_text(text)

    def set_text(self, text):
        self.text = text
        self.label.setText(escape(text).replace('\n', '<br>') or '正在思考…')

    def set_meta(self, text):
        self.meta.setText(text)
        self.meta.setVisible(bool(text))

    def resizeEvent(self, event):
        limit = max(180, min(760, int(self.width() * .82)))
        natural = max((QFontMetrics(self.label.font()).horizontalAdvance(line) for line in self.text.splitlines()), default=140)
        self.card.setFixedWidth(min(limit, max(170, natural + 36)))
        super().resizeEvent(event)


class ChatView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.surface = QWidget()
        self.surface.setObjectName('chatSurface')
        self.messages_layout = QVBoxLayout(self.surface)
        self.messages_layout.setContentsMargins(24, 24, 24, 24)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch(1)
        self.setWidget(self.surface)
        self.messages = []
        self.welcome = None
        self._follow_bottom = True
        self.verticalScrollBar().rangeChanged.connect(self._range_changed)
        self.verticalScrollBar().valueChanged.connect(self._scroll_changed)

    def _range_changed(self, minimum, maximum):
        if self._follow_bottom:
            self.verticalScrollBar().setValue(maximum)

    def _scroll_changed(self, value):
        self._follow_bottom = self.at_bottom()

    def at_bottom(self):
        bar = self.verticalScrollBar()
        return bar.maximum() - bar.value() < 36

    def scroll_to_bottom(self):
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(self.verticalScrollBar().maximum()))

    def add_message(self, role, text):
        follow = role == 'user' or self.at_bottom()
        self._follow_bottom = follow
        if self.welcome:
            self.welcome.hide()
        message = MessageBubble(role, text)
        self.messages.append(message)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, message)
        if follow:
            self.scroll_to_bottom()
        return message

    def update_message(self, message, text):
        follow = self.at_bottom()
        self._follow_bottom = follow
        message.set_text(text)
        if follow:
            self.scroll_to_bottom()

    def clear(self):
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.messages.clear()
        self.welcome = None

    def show_welcome(self, on_prompt, on_image, on_document):
        self.clear()
        self.welcome = QFrame()
        self.welcome.setObjectName('welcomeCard')
        body = QVBoxLayout(self.welcome)
        body.setContentsMargins(30, 30, 30, 30)
        title = QLabel('你好，今天有什么想聊的？')
        title.setObjectName('welcomeTitle')
        body.addWidget(title)
        hint = QLabel('用文字或语音提问，也可以带上一张图片。')
        hint.setObjectName('caption')
        body.addWidget(hint)
        body.addSpacing(16)
        for label, callback in [('聊一聊  →', lambda: on_prompt('用三句话介绍你能做什么')),
                                ('读懂一张图片  →', on_image),
                                ('问问我的资料  →', on_document)]:
            button = QPushButton(label)
            button.setObjectName('promptButton')
            button.clicked.connect(lambda checked=False, action=callback: action())
            body.addWidget(button)
        self.messages_layout.insertWidget(0, self.welcome)

    def toPlainText(self):
        return '\n'.join(m.text + ('\n' + m.meta.text() if m.meta.text() else '') for m in self.messages)
