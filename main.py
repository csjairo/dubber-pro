import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QMessageBox, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon

# CORREÇÃO 1: Adiciona 'src' ao path para encontrar o pacote 'dubber_pro'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from modules.dubber import Dubber
except ImportError as e:
    # Mostra o erro real para facilitar o debug
    print(f"Erro de importação: {e}")
    print("Verifique se a pasta 'src' existe e contém o pacote 'dubber_pro'.")
    sys.exit(1)

# ==========================================
# WORKER THREAD (Processamento em 2º Plano)
# ==========================================
class DubbingWorker(QThread):
    """Executa a dublagem em uma thread separada para não congelar a interface."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def run(self):
        try:
            # Função de callback para enviar logs para a GUI
            def gui_logger(msg):
                self.log_signal.emit(msg)

            self.log_signal.emit("⏳ Inicializando DubberPro e carregando modelos...")
            
            # Instancia o DubberPro passando o logger da GUI
            dubber = Dubber(logger_func=gui_logger)
            
            # Define o diretório base como o diretório do arquivo de entrada
            input_p = Path(self.input_path)
            dubber.base_dir = input_p.parent
            
            self.log_signal.emit(f"📂 Diretório de trabalho definido: {dubber.base_dir}")
            
            # Executa o processo
            dubber.process(str(input_p))
            
            self.finished_signal.emit(str(dubber.base_dir))
            
        except Exception as e:
            # Captura erros e envia para a GUI
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)

# ==========================================
# JANELA PRINCIPAL
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dubber PRO — Dublagem de Vídeos")
        self.resize(800, 600)
        
        # Variável para guardar o caminho do arquivo
        self.selected_file = None
        
        # Widget Central e Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Cabeçalho ---
        lbl_title = QLabel("Dubber Pro")
        lbl_title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        # --- Área de Seleção ---
        selection_layout = QHBoxLayout()
        
        self.btn_select = QPushButton("📂 Selecionar Vídeo")
        self.btn_select.setMinimumHeight(40)
        self.btn_select.clicked.connect(self.select_file)
        selection_layout.addWidget(self.btn_select)
        
        self.lbl_file = QLabel("Nenhum arquivo selecionado")
        self.lbl_file.setStyleSheet("color: #666; border: 1px solid #ccc; padding: 5px; background: #f9f9f9;")
        self.lbl_file.setMinimumHeight(40)
        selection_layout.addWidget(self.lbl_file, stretch=1)
        
        layout.addLayout(selection_layout)

        # --- Botão de Ação ---
        self.btn_run = QPushButton("▶️ Iniciar Dublagem")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-size: 14px; 
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_run.clicked.connect(self.start_dubbing)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)

        # --- Área de Logs ---
        lbl_log = QLabel("Log de Processamento:")
        lbl_log.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(lbl_log)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 9))
        self.txt_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00; border-radius: 5px;")
        layout.addWidget(self.txt_log)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Selecione o Vídeo", 
            "", 
            "Arquivos de Vídeo (*.mp4 *.mkv *.avi *.mov);;Todos os Arquivos (*.*)"
        )
        
        if file_path:
            self.selected_file = file_path
            self.lbl_file.setText(os.path.basename(file_path))
            self.btn_run.setEnabled(True)
            self.log(f"Arquivo selecionado: {file_path}")

    def start_dubbing(self):
        if not self.selected_file:
            return

        # Trava a interface
        self.btn_select.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Processando... Aguarde")
        self.txt_log.clear()
        
        # Inicia a thread
        self.worker = DubbingWorker(self.selected_file)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def log(self, msg):
        self.txt_log.append(msg)
        # Rola para o final automaticamente
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_finished(self, output_dir):
        QMessageBox.information(self, "Sucesso", f"Dublagem concluída!\nArquivos salvos em:\n{output_dir}")
        self.reset_ui()

    def on_error(self, error_msg):
        QMessageBox.critical(self, "Erro", f"Ocorreu um erro durante o processamento:\n{error_msg}")
        self.reset_ui()

    def reset_ui(self):
        self.btn_select.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶️ Iniciar Dublagem")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())