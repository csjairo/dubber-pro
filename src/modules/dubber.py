import shutil
import traceback
from pathlib import Path
from typing import List, Type

# Importações dos módulos refatorados
from .resources import ResourceManager
from .pipeline import PipelinePhase
from .phases import (
    ExtractionPhase,
    TranscriptionPhase,
    TranslationPhase,
    TTSPhase,
    AudioMixingPhase,
    RenderingPhase
)

class Dubber:
    """
    Orquestrador que gerencia o fluxo de trabalho (pipeline),
    diretórios temporários e limpeza de recursos.
    """
    def __init__(self, logger_func=None):
        self.logger = logger_func

    def log(self, msg):
        if self.logger: self.logger(msg)
        else: print(msg)

    def process(self, video_path: str, use_cache: bool = True):
        video_path = Path(video_path).resolve()
        parent_dir = video_path.parent
        
        # 1. Cria diretório temporário ao lado do arquivo original
        temp_dir_name = f"temp_{video_path.stem}"
        temp_dir = parent_dir / temp_dir_name
        temp_dir.mkdir(exist_ok=True)
        
        self.log(f"📁 Pasta temporária criada: {temp_dir}")
        
        context = {
            'video_path': str(video_path),
            'use_cache': use_cache,
            'segments': [],
            'project_dir': str(temp_dir)
        }

        # Lista de fases a serem executadas
        pipeline_classes: List[Type[PipelinePhase]] = [
            ExtractionPhase,
            TranscriptionPhase,
            TranslationPhase,
            TTSPhase,
            AudioMixingPhase,
            RenderingPhase
        ]

        try:
            for PhaseClass in pipeline_classes:
                # Instancia fase apontando para o diretório temporário
                phase = PhaseClass(temp_dir, self.log)
                
                self.log(f"--- Iniciando Fase: {PhaseClass.__name__} ---")
                context = phase.execute(context)
                
                # Limpeza explícita após cada fase
                del phase
                ResourceManager.force_cleanup(self.log)

            # 2. Movimentação do arquivo final para fora do temp
            generated_video = Path(context['output_video_path'])
            final_destination = parent_dir / generated_video.name
            
            if final_destination.exists():
                self.log(f"⚠️ Arquivo de saída já existe, substituindo: {final_destination.name}")
                final_destination.unlink() # Garante remoção segura antes de mover
            
            shutil.move(str(generated_video), str(final_destination))
            self.log(f"✅ Vídeo final salvo em: {final_destination}")
            
            return str(final_destination)

        except Exception as e:
            self.log(f"❌ Erro Crítico no Pipeline: {e}")
            self.log(traceback.format_exc())
            raise e
        
        finally:
            # 3. Limpeza Final (Deleta a pasta temporária)
            ResourceManager.force_cleanup()
            
            if temp_dir.exists():
                try:
                    self.log(f"🧹 Removendo arquivos temporários em: {temp_dir.name}...")
                    shutil.rmtree(temp_dir)
                    self.log("✨ Limpeza concluída.")
                except Exception as e:
                    self.log(f"⚠️ Falha ao remover pasta temporária: {e}")