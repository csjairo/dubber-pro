import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QMessageBox, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor

# Garante que módulos sejam encontrados
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from modules.dubber import Dubber
except ImportError as e:
    print(f"Erro de importação: {e}")
    sys.exit(1)

# ==========================================
# WORKER THREAD
# ==========================================
class DubbingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def run(self):
        try:
            def gui_logger(msg):
                self.log_signal.emit(msg)

            self.log_signal.emit("Inicializando Dubber PRO...")
            
            dubber = Dubber(logger_func=gui_logger)
            
            # O Dubber agora gerencia suas próprias pastas temporárias.
            # Não definimos base_dir manualmente.
            
            input_p = Path(self.input_path)
            
            # O process retorna o caminho do arquivo final
            final_file_path = dubber.process(str(input_p))
            
            # Emitimos o diretório pai (onde o arquivo final foi salvo)
            self.finished_signal.emit(str(Path(final_file_path).parent))
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)

# ==========================================
# JANELA PRINCIPAL (Sem alterações lógicas profundas, apenas a Worker acima mudou)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dubber PRO — Dublagem de Vídeos")
        self.resize(900, 700)
        self.apply_styles()
        
        self.selected_file = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        # Header
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_layout = QVBoxLayout(header_frame)
        
        lbl_title = QLabel("Dubber Pro")
        lbl_title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #6a11cb; margin-bottom: 5px;")
        header_layout.addWidget(lbl_title)
        
        lbl_subtitle = QLabel("DUBLAGEM AUTOMÁTICA DE VÍDEOS COM IA.")
        lbl_subtitle.setFont(QFont("Segoe UI", 10))
        lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_subtitle.setStyleSheet("color: #888; margin-bottom: 10px;")
        header_layout.addWidget(lbl_subtitle)
        layout.addWidget(header_frame)

        # Seleção
        card_frame = QFrame()
        card_frame.setObjectName("cardFrame")
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_card_title = QLabel("Seleção de Vídeo")
        lbl_card_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_card_title.setStyleSheet("color: #6a11cb; margin-bottom: 15px;")
        card_layout.addWidget(lbl_card_title)
        
        selection_layout = QHBoxLayout()
        selection_layout.setSpacing(15)
        
        self.btn_select = QPushButton("SELECIONAR VÍDEO")
        self.btn_select.setMinimumHeight(45)
        self.btn_select.setObjectName("primaryButton")
        self.btn_select.clicked.connect(self.select_file)
        selection_layout.addWidget(self.btn_select)
        
        file_container = QFrame()
        file_container.setObjectName("fileContainer")
        file_layout = QHBoxLayout(file_container)
        file_layout.setContentsMargins(15, 0, 15, 0)
        
        self.lbl_file = QLabel("NENHUM ARQUIVO SELECIONADO.")
        self.lbl_file.setFont(QFont("Segoe UI", 10))
        self.lbl_file.setMinimumHeight(45)
        self.lbl_file.setStyleSheet("color: #666;")
        file_layout.addWidget(self.lbl_file)
        
        selection_layout.addWidget(file_container, stretch=1)
        card_layout.addLayout(selection_layout)
        layout.addWidget(card_frame)

        # Botão Ação
        self.btn_run = QPushButton("Dublar")
        self.btn_run.setMinimumHeight(55)
        self.btn_run.setFont(QFont("Segoe UI", 12))
        self.btn_run.setObjectName("actionButton")
        self.btn_run.clicked.connect(self.start_dubbing)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)

        # Logs
        log_frame = QFrame()
        log_frame.setObjectName("logFrame")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(20, 20, 20, 20)
        
        log_header = QHBoxLayout()
        lbl_log = QLabel("Log de Processamento")
        lbl_log.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_log.setStyleSheet("color: #6a11cb;")
        log_header.addWidget(lbl_log)
        
        self.btn_clear = QPushButton("Limpar")
        self.btn_clear.setObjectName("secondaryButton")
        self.btn_clear.clicked.connect(lambda: self.txt_log.clear())
        self.btn_clear.setMaximumWidth(100)
        log_header.addWidget(self.btn_clear)
        
        log_layout.addLayout(log_header)
        
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 10))
        self.txt_log.setObjectName("logConsole")
        self.txt_log.setMinimumHeight(250)
        log_layout.addWidget(self.txt_log)
        layout.addWidget(log_frame)

    def apply_styles(self):
        style = """
        QMainWindow { background-color: #f8f9fa; }
        QFrame#headerFrame, QFrame#cardFrame, QFrame#logFrame {
            background-color: white; border-radius: 10px; border: 1px solid #e9ecef;
        }
        QPushButton#primaryButton {
            background-color: #6a11cb; color: white; font-weight: bold; border-radius: 8px; border: none;
        }
        QPushButton#primaryButton:hover { background-color: #5a0db0; }
        QPushButton#actionButton {
            background-color: #6a11cb; color: white; font-weight: bold; border-radius: 8px; border: none;
        }
        QPushButton#actionButton:hover { background-color: #5a0db0; }
        QPushButton#actionButton:disabled { background-color: #b19cd9; }
        QPushButton#secondaryButton {
            background-color: #f0f0f0; color: #6a11cb; border: 1px solid #ddd; border-radius: 6px;
        }
        QFrame#fileContainer {
            background-color: #f8f9fa; border-radius: 8px; border: 2px dashed #dee2e6;
        }
        QTextEdit#logConsole {
            background-color: #0d1117; color: #00ff88; border-radius: 8px; border: 1px solid #30363d;
        }
        """
        self.setStyleSheet(style)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecione o Vídeo", "", "Arquivos de Vídeo (*.mp4 *.mkv *.avi *.mov);;Todos (*.*)"
        )
        if file_path:
            self.selected_file = file_path
            self.lbl_file.setText(f"✓ {os.path.basename(file_path)}")
            self.lbl_file.setStyleSheet("color: #28a745; font-weight: bold;")
            self.btn_run.setEnabled(True)

    def start_dubbing(self):
        if not self.selected_file: return
        self.btn_select.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Processando... Aguarde")
        self.txt_log.clear()
        
        self.worker = DubbingWorker(self.selected_file)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def log(self, msg):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {msg}")
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_finished(self, output_dir):
        QMessageBox.information(self, "Sucesso", f"Dublagem concluída!\nSalvo em: {output_dir}")
        self.reset_ui()

    def on_error(self, error_msg):
        QMessageBox.critical(self, "Erro", f"Erro:\n{error_msg[:500]}...")
        self.reset_ui()

    def reset_ui(self):
        self.btn_select.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Dublar")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())