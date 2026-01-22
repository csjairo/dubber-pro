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
            device_type (str): 'cuda', 'mps' ou 'cpu'
            backend_suggestion (str): 'faster-whisper' ou 'openai-whisper'
        """
        if ResourceManager._device_cache:
            return ResourceManager._device_cache

        device_type = "cpu"
        backend = "faster-whisper" # Padrão mais rápido e eficiente para CPU

        # 1. Tenta CUDA (NVIDIA) - Prioridade Máxima
        if torch.cuda.is_available():
            device_type = "cuda"
            backend = "faster-whisper"
        
        # 2. Tenta MPS (Mac / Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            backend = "openai-whisper" # faster-whisper ainda tem suporte instável a MPS
            
        # Nota: O suporte legado a DirectML foi removido para garantir estabilidade.
        # Caso não haja GPU dedicada suportada (CUDA/MPS), o sistema usará a CPU.

        ResourceManager._device_cache = (device_type, backend)
        return ResourceManager._device_cache

    @staticmethod
    def force_cleanup(logger=print):
        """Força a liberação de RAM e VRAM agressivamente."""
        if logger:
            logger("🧹 Executando Garbage Collection e Limpeza de VRAM...")

        gc.collect()

        # Limpeza para CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Limpeza para MPS (Mac)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass