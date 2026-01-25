import torch
import gc
import logging

class ResourceManager:
    """Gerencia recursos focando exclusivamente em NVIDIA CUDA e CPU."""
    
    _device_cache = None

    @staticmethod
    def get_best_device():
        """
        Retorna:
            device_type (str): 'cuda' ou 'cpu'
            backend_suggestion (str): 'faster-whisper'
        """
        if ResourceManager._device_cache:
            return ResourceManager._device_cache

        # Padrão: CPU
        device_type = "cpu"
        backend = "faster-whisper"

        # Verificação Única: NVIDIA CUDA
        if torch.cuda.is_available():
            device_type = "cuda"
            backend = "faster-whisper"
            # O backend 'faster-whisper' é altamente otimizado para CUDA (via CTranslate2)
        
        # Removemos verificações de MPS (Mac) e DirectML (AMD/Intel legado)

        ResourceManager._device_cache = (device_type, backend)
        return ResourceManager._device_cache

    @staticmethod
    def force_cleanup(logger=print):
        """Libera RAM e VRAM (CUDA)."""
        if logger:
            logger("🧹 Executando Garbage Collection e Limpeza de VRAM...")

        gc.collect()

        # Limpeza exclusiva para CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()