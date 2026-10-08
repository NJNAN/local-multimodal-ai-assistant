"""桌面助手的暖白与青绿主题。"""
THEME_QSS = '''
QMainWindow, QWidget#root { background: #f4f6f5; }
QWidget { font-family: "Microsoft YaHei UI", "Microsoft YaHei"; font-size: 13px; color: #263c36; }
QLabel { background: transparent; }
QFrame#sidebar { background: #ffffff; border: 1px solid #e0e8e3; border-radius: 16px; }
QWidget#sidebarContent { background: #ffffff; }
QFrame#chatPanel { background: #fafcfb; border: 1px solid #e0e8e3; border-radius: 16px; }
QLabel#headerTitle { font-size: 23px; font-weight: 700; color: #163e33; }
QLabel#headerSub, QLabel#caption { color: #74877f; font-size: 12px; }
QLabel#sectionTitle { color: #254c3f; font-size: 15px; font-weight: 700; }
QLabel#badge { color: #4f7465; background: #eaf3ee; border-radius: 10px; padding: 5px 9px; font-size: 11px; }
QLabel#videoLabel { background: #eef3f0; color: #7a9185; border: 1px solid #dfe8e2; border-radius: 12px; }
QWidget#chatSurface, QScrollArea#chat { background: #fafcfb; border: none; }
QScrollArea#sidebarScroll { background: transparent; border: none; }
QFrame#userBubble { background: #dceee3; border: 1px solid #cce1d4; border-radius: 14px; }
QFrame#aiBubble { background: #ffffff; border: 1px solid #e0e8e3; border-radius: 14px; }
QLabel#messageText { color: #263c36; font-size: 15px; line-height: 1.6; }
QLabel#messageMeta { color: #75897e; font-size: 11px; }
QLabel#userAvatar { background: #486b5a; color: white; border-radius: 12px; font-weight: 700; }
QLabel#aiAvatar { background: #187957; color: white; border-radius: 12px; font-weight: 700; }
QFrame#welcomeCard { background: transparent; }
QLabel#welcomeTitle { font-size: 23px; font-weight: 700; color: #234d3e; }
QPushButton { background: #ffffff; border: 1px solid #dae5dd; border-radius: 8px; padding: 8px 12px; color: #426453; }
QPushButton:hover { background: #edf5ef; border-color: #93b9a1; }
QPushButton:pressed { background: #dcebdd; }
QPushButton:checked { background: #e4f3e9; border-color: #8fbea0; color: #187957; }
QPushButton:disabled { background: #f2f4f2; color: #acb8b0; border-color: #e7ece8; }
QPushButton#sendButton { background: #187957; color: white; border: none; font-weight: 700; padding: 10px 22px; }
QPushButton#sendButton:hover { background: #126443; }
QPushButton#stopButton { color: #9b6250; background: #faf2ed; border-color: #e7d5ca; }
QPushButton#copyButton { border: none; background: transparent; color: #84968b; font-size: 11px; padding: 0 3px; }
QPushButton#promptButton { text-align: left; padding: 14px 16px; background: #ffffff; }
QFrame#composer { background: white; border: 1px solid #cbded1; border-radius: 14px; }
QPlainTextEdit#input { background: transparent; border: none; padding: 5px; font-size: 14px; selection-background-color: #bde0cc; }
QLabel#metrics { color: #7c9085; font-size: 12px; padding: 2px 4px; }
QComboBox { background: #ffffff; border: 1px solid #dae5dd; border-radius: 8px; padding: 7px 10px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #dceee3; color: #263c36; }
QStatusBar { background: #f4f6f5; color: #7c9085; font-size: 12px; }
QStatusBar::item { border: none; }
QSplitter::handle { background: transparent; }
QScrollBar:vertical { background: transparent; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: #c7d7cb; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QDialog { background: #f8faf8; }
QTableWidget { background: white; border: 1px solid #dae5dd; gridline-color: #e9eee9; selection-background-color: #dceee3; selection-color: #263c36; }
QHeaderView::section { background: #edf5ef; border: none; padding: 10px; }
'''
