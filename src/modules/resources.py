import torch
import gc
import sys
import logging

class ResourceManager:
    """Responsável por gerenciar recursos de hardware e limpeza de memória."""
    
    _device_cache = None

    @staticmethod
    def get_best_device():
        """
        Retorna:
            device_type (str): 'cuda', 'privateuseone' (DirectML), 'mps' ou 'cpu'
            backend_suggestion (str): 'faster-whisper' ou 'openai-whisper'
        """
        if ResourceManager._device_cache:
            return ResourceManager._device_cache

        device_type = "cpu"
        backend = "faster-whisper" # Padrão mais rápido

        # 1. Tenta CUDA (NVIDIA)
        if torch.cuda.is_available():
            device_type = "cuda"
            backend = "faster-whisper"
        
        # 2. Tenta MPS (Mac)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            backend = "openai-whisper" # faster-whisper tem suporte instável a MPS
            
        # 3. Tenta DirectML (AMD/Intel no Windows)
        else:
            try:
                # DirectML é um pacote separado, só importamos se necessário
                import torch_directml
                if torch_directml.is_available():
                    device_type = torch_directml.device() # Retorna 'privateuseone:0'
                    backend = "openai-whisper" # faster-whisper NÃO roda em DirectML (exige CTranslate2)
            except ImportError:
                pass

        ResourceManager._device_cache = (device_type, backend)
        return ResourceManager._device_cache

    @staticmethod
    def force_cleanup(logger=print):
        """Força a liberação de RAM e VRAM agressivamente."""
        if logger:
            logger("🧹 Executando Garbage Collection e Limpeza de VRAM...")

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass