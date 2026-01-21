import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QMessageBox, QFrame)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor

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

            self.log_signal.emit("Inicializando Dubber PRO e carregando modelos...")
            
            # Instancia o DubberPro passando o logger da GUI
            dubber = Dubber(logger_func=gui_logger)
            
            # Define o diretório base como o diretório do arquivo de entrada
            input_p = Path(self.input_path)
            dubber.base_dir = input_p.parent
            
            self.log_signal.emit(f"Diretório de trabalho definido: {dubber.base_dir}")
            
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
        self.resize(900, 700)
        
        # Aplicar estilo geral
        self.apply_styles()
        
        # Variável para guardar o caminho do arquivo
        self.selected_file = None
        
        # Widget Central e Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        # --- Cabeçalho ---
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

        # --- Card de Seleção de Arquivo ---
        card_frame = QFrame()
        card_frame.setObjectName("cardFrame")
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_card_title = QLabel("Seleção de Vídeo")
        lbl_card_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_card_title.setStyleSheet("color: #6a11cb; margin-bottom: 15px;")
        card_layout.addWidget(lbl_card_title)
        
        # Área de Seleção
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

        # --- Botão de Ação ---
        self.btn_run = QPushButton("Dublar")
        self.btn_run.setMinimumHeight(55)
        self.btn_run.setFont(QFont("Segoe UI", 12))
        self.btn_run.setObjectName("actionButton")
        self.btn_run.clicked.connect(self.start_dubbing)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)

        # --- Área de Logs ---
        log_frame = QFrame()
        log_frame.setObjectName("logFrame")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(20, 20, 20, 20)
        
        log_header = QHBoxLayout()
        
        lbl_log = QLabel("Debug para Experts")
        lbl_log.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_log.setStyleSheet("color: #6a11cb;")
        log_header.addWidget(lbl_log)
        
        # Botão para limpar logs
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
        """Aplica estilos CSS à aplicação"""
        style = """
        QMainWindow {
            background-color: #f8f9fa;
        }
        
        QFrame#headerFrame {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #e9ecef;
        }
        
        QFrame#cardFrame {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        
        QFrame#logFrame {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        
        QPushButton#primaryButton {
            background-color: #6a11cb;
            color: white;
            font-size: 12px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
        }
        
        QPushButton#primaryButton:hover {
            background-color: #5a0db0;
        }
        
        QPushButton#primaryButton:pressed {
            background-color: #4a0a95;
        }
        
        QPushButton#primaryButton:disabled {
            background-color: #cccccc;
        }
        
        QPushButton#actionButton {
            background-color: #6a11cb;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 15px;
        }
        
        QPushButton#actionButton:hover {
            background-color: #5a0db0;
        }
        
        QPushButton#actionButton:pressed {
            background-color: #4a0a95;
        }
        
        QPushButton#actionButton:disabled {
            background-color: #b19cd9;
        }
        
        QPushButton#secondaryButton {
            background-color: #f0f0f0;
            color: #6a11cb;
            font-size: 11px;
            font-weight: bold;
            border-radius: 6px;
            padding: 8px 15px;
            border: 1px solid #ddd;
        }
        
        QPushButton#secondaryButton:hover {
            background-color: #e6e6e6;
            border-color: #6a11cb;
        }
        
        QFrame#fileContainer {
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 2px dashed #dee2e6;
        }
        
        QTextEdit#logConsole {
            background-color: #0d1117;
            color: #00ff88;
            border-radius: 8px;
            border: 1px solid #30363d;
            padding: 10px;
        }
        
        QScrollBar:vertical {
            background-color: #1a1a1a;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #6a11cb;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #5a0db0;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        """
        
        self.setStyleSheet(style)
        
        # Configura paleta de cores para alguns elementos
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(248, 249, 250))
        self.setPalette(palette)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Selecione o Vídeo", 
            "", 
            "Arquivos de Vídeo (*.mp4 *.mkv *.avi *.mov);;Todos os Arquivos (*.*)"
        )
        
        if file_path:
            self.selected_file = file_path
            filename = os.path.basename(file_path)
            self.lbl_file.setText(f"✓ {filename}")
            self.lbl_file.setStyleSheet("color: #28a745; font-weight: bold;")
            self.btn_run.setEnabled(True)
            self.log(f"Arquivo selecionado: {filename}")

    def start_dubbing(self):
        if not self.selected_file:
            return

        # Trava a interface
        self.btn_select.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Processando... Aguarde")
        self.txt_log.clear()
        
        # Inicia a thread
        self.worker = DubbingWorker(self.selected_file)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def log(self, msg):
        # Adiciona timestamp aos logs
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        
        self.txt_log.append(formatted_msg)
        # Rola para o final automaticamente
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_finished(self, output_dir):
        success_msg = f"""
        <div style='color: #28a745; font-weight: bold;'>
        ✅ Dublagem concluída com sucesso!
        </div>
        <div style='margin-top: 10px; color: #555;'>
        <b>Arquivos salvos em:</b><br>
        <code style='background-color: #f8f9fa; padding: 5px; border-radius: 4px;'>{output_dir}</code>
        </div>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Sucesso")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(success_msg)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.exec()
        
        self.reset_ui()

    def on_error(self, error_msg):
        error_display = f"""
        <div style='color: #dc3545; font-weight: bold;'>
        ❌ Ocorreu um erro durante o processamento
        </div>
        <div style='margin-top: 10px; color: #555;'>
        <b>Detalhes:</b><br>
        <pre style='background-color: #f8f9fa; padding: 10px; border-radius: 4px; 
                   border: 1px solid #ddd; max-height: 200px; overflow: auto;'>
        {error_msg[:500]}...
        </pre>
        </div>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Erro")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(error_display)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.exec()
        
        self.reset_ui()

    def reset_ui(self):
        self.btn_select.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Dublar")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Configura fonte padrão
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())