import os
import webbrowser
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QToolButton,
    QMenu,
)
# QAction movido para o local correto no PyQt6
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtCore import Qt

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
        """Configura a toolbar com menus dropdown organizados"""
        toolbar = self.addToolBar("MainToolbar")
        toolbar.setMovable(False)

        # CSS para Toolbar e Menus
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: white;
                border-bottom: 1px solid #d0d0d0;
                spacing: 8px;
                padding: 5px;
            }
            QToolButton {
                color: black;
                font-family: "Segoe UI";
                font-size: 9pt;
                font-weight: 500;
                background-color: transparent;
                border: none;
                padding: 5px 12px;
            }
            QToolButton:hover {
                background-color: #f0f0f0;
                border-radius: 4px;
            }
            QMenu {
                background-color: white;
                border: 1px solid #d0d0d0;
                font-family: "Segoe UI";
                font-size: 10pt;
                padding: 4px 0px;
            }
            QMenu::item {
                padding: 6px 30px;
                color: black;
            }
            QMenu::item:selected {
                background-color: #f0f0f0;
            }
            QMenu::separator {
                height: 1px;
                background: #e0e0e0;
                margin: 4px 0px;
            }
        """)

        # --- Menu: Opções ---
        btn_options = QToolButton(self)
        btn_options.setText("Opções")
        btn_options.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_options = QMenu(btn_options)
        
        act_settings = QAction("Configurações", self)
        act_settings.triggered.connect(self.open_settings)
        menu_options.addAction(act_settings)
        
        menu_options.addSeparator()
        
        act_exit = QAction("Sair", self)
        act_exit.triggered.connect(self.close)
        menu_options.addAction(act_exit)
        
        btn_options.setMenu(menu_options)
        toolbar.addWidget(btn_options)

        # --- Menu: Exibir ---
        btn_view = QToolButton(self)
        btn_view.setText("Exibir")
        btn_view.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_view = QMenu(btn_view)
        
        act_console = QAction("Console de Depuração", self)
        act_console.setCheckable(True)
        act_console.setChecked(True)
        act_console.triggered.connect(self.toggle_console)
        menu_view.addAction(act_console)
        
        btn_view.setMenu(menu_view)
        toolbar.addWidget(btn_view)

        # --- Menu: Ajuda ---
        btn_help = QToolButton(self)
        btn_help.setText("Ajuda")
        btn_help.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_help = QMenu(btn_help)
        
        act_github = QAction("Link para GitHub", self)
        act_github.triggered.connect(self.open_github)
        menu_help.addAction(act_github)
        
        btn_help.setMenu(menu_help)
        toolbar.addWidget(btn_help)

        # --- Menu: Sobre ---
        btn_about = QToolButton(self)
        btn_about.setText("Sobre")
        btn_about.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_about = QMenu(btn_about)
        
        act_about_info = QAction("Informações sobre o projeto", self)
        act_about_info.triggered.connect(self.show_about)
        menu_about.addAction(act_about_info)
        
        btn_about.setMenu(menu_about)
        toolbar.addWidget(btn_about)

    # --- Slots de Ação ---
    def toggle_console(self, visible):
        """Alterna a visibilidade do ConsoleComponent"""
        self.console.setVisible(visible)

    def open_github(self):
        """Abre o repositório no navegador"""
        webbrowser.open("https://github.com/csjairo/dubber-pro")

    def open_settings(self):
        QMessageBox.information(self, "Configurações", "Janela de configurações.")

    def show_about(self):
        QMessageBox.about(
            self, 
            "Sobre", 
            "Dubber PRO v0.1.0\n\nSoftware de dublagem automática com IA."
        )

    def load_styles(self):
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
        self.selected_file = file_path
        self.btn_run.setEnabled(True)

    def start_dubbing(self):
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