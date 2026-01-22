import os
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt6.QtGui import QFont
from PyQt6.QtCore import pyqtSignal

class SelectionComponent(QFrame):
    # Sinal personalizado: envia o caminho do arquivo (str) quando selecionado
    fileSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cardFrame")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_card_title = QLabel("Seleção de Vídeo")
        lbl_card_title.setObjectName("lblCardTitle")
        lbl_card_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(lbl_card_title)
        
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(15)
        
        self.btn_select = QPushButton("SELECIONAR VÍDEO")
        self.btn_select.setMinimumHeight(45)
        self.btn_select.setObjectName("primaryButton")
        self.btn_select.clicked.connect(self._open_file_dialog)
        selection_layout.addWidget(self.btn_select)
        
        file_container = QFrame()
        file_container.setObjectName("fileContainer")
        file_layout = QHBoxLayout(file_container)
        file_layout.setContentsMargins(15, 0, 15, 0)
        
        self.lbl_file = QLabel("NENHUM ARQUIVO SELECIONADO.")
        self.lbl_file.setObjectName("lblFile")
        self.lbl_file.setFont(QFont("Segoe UI", 10))
        self.lbl_file.setMinimumHeight(45)
        file_layout.addWidget(self.lbl_file)
        
        selection_layout.addWidget(file_container, stretch=1)
        layout.addLayout(selection_layout)

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecione o Vídeo", "", "Arquivos de Vídeo (*.mp4 *.mkv *.avi *.mov);;Todos (*.*)"
        )
        if file_path:
            self.lbl_file.setText(f"✓ {os.path.basename(file_path)}")
            # Injeta estilo inline para feedback visual imediato
            self.lbl_file.setStyleSheet("color: #28a745; font-weight: bold;")
            self.fileSelected.emit(file_path)

    def set_enabled(self, enabled: bool):
        """Bloqueia ou desbloqueia a interação"""
        self.btn_select.setEnabled(enabled)