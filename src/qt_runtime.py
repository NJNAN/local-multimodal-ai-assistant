"""Use PyQt's real installation path when Qt's baked-in path loses Unicode."""
from __future__ import annotations

import os
from pathlib import Path


def configure_qt_plugins() -> Path:
    import PyQt5
    from PyQt5.QtCore import QCoreApplication

    package_dir = Path(PyQt5.__file__).resolve().parent
    candidates = [package_dir / "Qt5" / "plugins", package_dir / "Qt" / "plugins"]
    plugin_dir = next((path for path in candidates if (path / "platforms").is_dir()), None)
    if plugin_dir is None:
        raise RuntimeError("PyQt5 平台插件缺失，请在当前 Python 环境重新安装 PyQt5-Qt5")
    # Do this before QApplication: Qt 5's default installation path can contain
    # literal '?' on Windows when the virtual environment lives in a CJK folder.
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_dir / "platforms")
    if os.name == "nt" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        if fonts.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
    QCoreApplication.setLibraryPaths([str(plugin_dir)])
    return plugin_dir
