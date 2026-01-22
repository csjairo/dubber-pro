import os
from pathlib import Path

# Correção F401: QToolBar removido da importação
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QToolButton,
    QMenu,
)
from PyQt6.QtGui import QFont, QAction

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

        # --- Configuração da Toolbar ---
        self.setup_toolbar()

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
        self.selection_card.fileSelected.connect(self.on_file_selected)
        self.layout.addWidget(self.selection_card)

        # 3. Botão de Ação
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

    def setup_toolbar(self):
        """Configura a toolbar com menu dropdown"""
        toolbar = self.addToolBar("MainToolbar")
        toolbar.setMovable(False)

        # CSS Inline: Toolbar e Menu Dropdown
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: white;
                border-bottom: 1px solid #d0d0d0;
                spacing: 10px;
                padding: 5px;
            }
            /* Botões da Toolbar */
            QToolButton {
                color: black;
                font-family: "Segoe UI";
                font-size: 9pt;
                font-weight: 500;
                background-color: transparent;
                border: none;
                padding: 4px 10px;
                margin: 0px;
            }
            QToolButton:hover {
                background-color: #f0f0f0;
                border-radius: 3px;
            }
            QToolButton:pressed {
                background-color: #e0e0e0;
            }
            /* Menu Dropdown (Opções) */
            QMenu {
                background-color: white;
                border: 1px solid #d0d0d0;
                font-family: "Segoe UI";
                font-size: 10pt;
                padding: 5px 0px;
            }
            QMenu::item {
                padding: 6px 25px; /* Espaçamento interno dos itens */
                color: black;
            }
            QMenu::item:selected {
                background-color: #f0f0f0; /* Cor ao passar o mouse no item */
            }
        """)

        # 2. Menu "Opções" (Dropdown)
        # Cria um botão especial para a toolbar
        btn_options = QToolButton(self)
        btn_options.setText("Opções")
        # Define que ao clicar, o menu abre imediatamente (InstantPopup)
        btn_options.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        # Cria o menu container
        menu_options = QMenu(btn_options)

        # Item: Configurações
        act_settings = QAction("Configurações", self)
        act_settings.triggered.connect(self.open_settings)
        menu_options.addAction(act_settings)

        # Item: Avançado
        act_advanced = QAction("Avançado", self)
        act_advanced.triggered.connect(self.open_advanced)
        menu_options.addAction(act_advanced)

        # Vincula o menu ao botão e adiciona à toolbar
        btn_options.setMenu(menu_options)
        toolbar.addWidget(btn_options)

        # 3. Help
        act_help = QAction("Help", self)
        act_help.triggered.connect(self.open_help)
        toolbar.addAction(act_help)

        # 4. Sobre
        act_about = QAction("Sobre", self)
        act_about.triggered.connect(self.show_about)
        toolbar.addAction(act_about)

        # 1. Sair
        act_exit = QAction("Sair", self)
        act_exit.triggered.connect(self.close)
        toolbar.addAction(act_exit)

    # --- Actions Slots (Placeholders) ---
    def open_settings(self):
        QMessageBox.information(self, "Configurações", "Janela de configurações.")

    def open_advanced(self):
        QMessageBox.information(self, "Avançado", "Opções avançadas.")

    def open_help(self):
        QMessageBox.information(self, "Help", "Ajuda do sistema.")

    def show_about(self):
        QMessageBox.about(self, "Sobre", "Dubber PRO v0.1.0")

    def load_styles(self):
        style_path = Path(resource_path(os.path.join("styles", "main.qss")))
        try:
            if style_path.exists():
                with open(style_path, "r", encoding="utf-8") as f:
                    # Carrega estilos globais, mas o CSS inline da Toolbar tem prioridade
                    self.setStyleSheet(f.read())
            else:
                print(f"⚠️ Estilo não encontrado: {style_path}")
        except Exception as e:
            print(f"❌ Erro ao carregar estilos: {e}")

    def on_file_selected(self, file_path):
        self.selected_file = file_path
        self.btn_run.setEnabled(True)

    def start_dubbing(self):
        # Correção E701: Declaração em múltiplas linhas
        if not self.selected_file:
            return

        self.selection_card.set_enabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ Processando... Aguarde")
        self.console.clear_log()

        self.worker = DubbingWorker(self.selected_file)
        self.worker.log_signal.connect(self.console.append_log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_finished(self, output_dir):
        QMessageBox.information(
            self, "Sucesso", f"Dublagem concluída!\nSalvo em: {output_dir}"
        )
        self.reset_ui()

    def on_error(self, error_msg):
        QMessageBox.critical(self, "Erro", f"Ocorreu um erro:\n{error_msg[:500]}...")
        self.reset_ui()

    def reset_ui(self):
        self.selection_card.set_enabled(True)
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Dublar")
