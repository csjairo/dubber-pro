from PyQt6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
)
from PyQt6.QtGui import QFont
from datetime import datetime


class ConsoleComponent(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("logFrame")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QHBoxLayout()
        lbl_log = QLabel("Log de Processamento")
        lbl_log.setObjectName("lblLogHeader")
        lbl_log.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.addWidget(lbl_log)

        self.btn_clear = QPushButton("Limpar")
        self.btn_clear.setObjectName("secondaryButton")
        self.btn_clear.clicked.connect(self.clear_log)
        self.btn_clear.setMaximumWidth(100)
        header.addWidget(self.btn_clear)

        layout.addLayout(header)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 10))
        self.txt_log.setObjectName("logConsole")
        self.txt_log.setMinimumHeight(250)
        layout.addWidget(self.txt_log)

    def append_log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {msg}")
        # Auto-scroll para o final
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        self.txt_log.clear()
