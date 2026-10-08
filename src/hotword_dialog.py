"""热词编辑器，保存成功后同步更新识别线程共用的词表。"""
import re
from PyQt5.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
                            QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)


class HotwordDialog(QDialog):
    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self.setWindowTitle('自定义语音热词')
        self.resize(640, 470)
        layout = QVBoxLayout(self)
        title = QLabel('让助手听懂你的专业词')
        title.setObjectName('sectionTitle')
        layout.addWidget(title)
        hint = QLabel('填写标准词和常见误识别写法，例如：通义千问 ← 同义千问。\n多个误识别词用逗号分隔。直接双击表格可修改，保存后立即生效。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.enabled = QCheckBox('启用热词纠正')
        self.enabled.setChecked(store.enabled)
        layout.addWidget(self.enabled)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['标准词（必填）', '常见误识别词（逗号分隔）'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table, 1)
        for entry in store.entries:
            self.add_row(entry['term'], '，'.join(entry['aliases']))
        row = QHBoxLayout()
        self.add_button = QPushButton('＋ 添加热词')
        self.add_button.clicked.connect(lambda: self.add_row())
        row.addWidget(self.add_button)
        self.remove_button = QPushButton('删除所选')
        self.remove_button.clicked.connect(self.remove_rows)
        row.addWidget(self.remove_button)
        row.addStretch()
        cancel = QPushButton('取消')
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        self.save_button = QPushButton('保存并应用')
        self.save_button.setObjectName('sendButton')
        self.save_button.clicked.connect(self.save)
        row.addWidget(self.save_button)
        self.feedback = QLabel('仅纠正词表中明确列出的写法；可随时关闭。')
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        layout.addLayout(row)

    def add_row(self, term='', aliases=''):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(term))
        self.table.setItem(row, 1, QTableWidgetItem(aliases))
        self.table.setCurrentCell(row, 0)
        if not term:
            self.table.editItem(self.table.item(row, 0))

    def remove_rows(self):
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def save(self):
        entries = []
        for row in range(self.table.rowCount()):
            term = self.table.item(row, 0).text().strip()
            aliases = re.split(r'[,，;；]', self.table.item(row, 1).text())
            entries.append({'term': term, 'aliases': aliases})
        try:
            self.store.update(entries, self.enabled.isChecked())
        except (ValueError, OSError) as exc:
            self.feedback.setText(str(exc))
            return
        self.accept()
