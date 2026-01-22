import os
from pathlib import Path
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QMessageBox
from PyQt6.QtGui import QFont

# Importações relativas dentro do pacote UI
from .components.header import HeaderComponent
from .components.selection import SelectionComponent
from .components.console import ConsoleComponent
from .worker import DubbingWorker
from .utils import resource_path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dubber PRO — Dublagem de Vídeos")
        self.resize(900, 700)
        
        self.load_styles()
        self.selected_file = None
        
        # --- Configuração do Layout Principal ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setSpacing(15)
        self.layout.setContentsMargins(25, 25, 25, 25)

        # 1. Header
        self.header = HeaderComponent()
        self.layout.addWidget(self.header)

        # 2. Seleção
        self.selection_card = SelectionComponent()
        # Conecta o sinal do componente ao método da janela
        self.selection_card.fileSelected.connect(self.on_file_selected)
        self.layout.addWidget(self.selection_card)

        # 3. Botão de Ação (Ainda controlado pela janela pois orquestra tudo)
        self.btn_run = QPushButton("Dublar")
        self.btn_run.setMinimumHeight(55)
        self.btn_run.setFont(QFont("Segoe UI", 12))
        self.btn_run.setObjectName("actionButton")
        self.btn_run.clicked.connect(self.start_dubbing)
        self.btn_run.setEnabled(False)
        self.layout.addWidget(self.btn_run)

        # 4. Console
        self.console = ConsoleComponent()
        self.layout.addWidget(self.console)

    def load_styles(self):
        """ Carrega o arquivo CSS externo """
        style_path = Path(resource_path(os.path.join("styles", "main.qss")))
        try:
            if style_path.exists():
                with open(style_path, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            else:
                print(f"⚠️ Estilo não encontrado: {style_path}")
        except Exception as e:
            print(f"❌ Erro ao carregar estilos: {e}")

    def on_file_selected(self, file_path):
        """ Callback disparado quando o componente de seleção escolhe um arquivo """
        self.selected_file = file_path
        self.btn_run.setEnabled(True)

    def start_dubbing(self):
        if not self.selected_file: return
        
        # Atualiza Estado da UI
        self.selection_card.set_enabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Processando... Aguarde")
        self.console.clear_log()
        
        # Inicia Worker
        self.worker = DubbingWorker(self.selected_file)
        self.worker.log_signal.connect(self.console.append_log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self, output_dir):
        QMessageBox.information(self, "Sucesso", f"Dublagem concluída!\nSalvo em: {output_dir}")
        self.reset_ui()

    def on_error(self, error_msg):
        QMessageBox.critical(self, "Erro", f"Ocorreu um erro:\n{error_msg[:500]}...")
        self.reset_ui()

    def reset_ui(self):
        self.selection_card.set_enabled(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Dublar")