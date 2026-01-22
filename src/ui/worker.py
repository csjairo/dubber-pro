from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
import traceback
from src.modules.dubber import Dubber


class DubbingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def run(self):
        if not Dubber:
            self.error_signal.emit(
                "❌ Erro: Módulo 'Dubber' não encontrado. Verifique o src."
            )
            return

        try:
            # Função de callback para redirecionar logs do Dubber para a UI
            def gui_logger(msg):
                self.log_signal.emit(msg)

            self.log_signal.emit("🚀 Inicializando pipeline de dublagem...")

            dubber = Dubber(logger_func=gui_logger)
            input_p = Path(self.input_path)

            # Executa o processamento pesado
            final_file_path = dubber.process(str(input_p))

            # Retorna apenas a pasta onde o arquivo foi salvo
            self.finished_signal.emit(str(Path(final_file_path).parent))

        except Exception as e:
            # Captura traceback completo para debug
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)
