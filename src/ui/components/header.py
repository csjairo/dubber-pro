from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


class HeaderComponent(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("headerFrame")  # ID para o CSS

        layout = QVBoxLayout(self)

        lbl_title = QLabel("Dubber Pro")
        lbl_title.setObjectName("lblTitle")
        lbl_title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        lbl_subtitle = QLabel("DUBLAGEM AUTOMÁTICA DE VÍDEOS COM IA.")
        lbl_subtitle.setObjectName("lblSubtitle")
        lbl_subtitle.setFont(QFont("Segoe UI", 10))
        lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_subtitle)
