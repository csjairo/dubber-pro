import sys
from PyQt6.QtWidgets import QApplication

# 1. Configura ambiente (Paths, FFmpeg) importando de src.ui.utils
from src.ui.utils import setup_environment

setup_environment()

if __name__ == "__main__":
    # 2. Importa a Janela Principal
    from src.ui.window import MainWindow

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
