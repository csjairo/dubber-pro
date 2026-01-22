from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Callable
from .resources import ResourceManager


class PipelinePhase(ABC):
    """Classe abstrata que define o contrato de uma fase isolada."""

    def __init__(self, base_dir: Path, logger: Callable[[str], None]):
        self.base_dir = base_dir
        self.logger = logger
        self.device = ResourceManager.get_best_device()

    def log(self, msg: str):
        if self.logger:
            self.logger(f"[{self.__class__.__name__}] {msg}")
        else:
            print(f"[{self.__class__.__name__}] {msg}")

    @abstractmethod
    def execute(self, context: Dict) -> Dict:
        """Executa a lógica da fase e retorna o contexto atualizado."""
        pass
