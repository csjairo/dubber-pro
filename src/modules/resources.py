import torch
import gc

class ResourceManager:
    """Responsável por gerenciar recursos de hardware e limpeza de memória."""
    
    @staticmethod
    def get_best_device():
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def force_cleanup(logger=None):
        """Força a liberação de RAM e VRAM agressivamente."""
        if logger: logger("🧹 Executando Garbage Collection e Limpeza de VRAM...")
        
        # 1. Coleta de lixo do Python
        gc.collect()
        
        # 2. Limpeza de Cache da NVIDIA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # 3. Limpeza de Cache da Apple (MPS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass