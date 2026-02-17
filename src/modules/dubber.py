import shutil
import traceback
import hashlib
import json
import time
from pathlib import Path
from typing import List, Type

from src.config import Config

# Importações dos módulos refatorados
from .resources import ResourceManager
from .pipeline import PipelinePhase
from .phases import (
    ExtractionPhase,
    SeparationPhase,
    TranscriptionPhase,
    PostProcessingPhase,
    TranslationPhase,
    TTSPhase,
    AudioMixingPhase,
    RenderingPhase,
)


class Dubber:
    """
    Orquestrador que gerencia o fluxo de trabalho (pipeline),
    diretórios temporários e limpeza de recursos.
    """

    def __init__(self, logger_func=None):
        self.logger = logger_func

    def log(self, msg):
        if self.logger:
            self.logger(msg)
        else:
            print(msg)

    def _build_pipeline_cache_key(self, video_path: Path) -> str:
        if not video_path.exists():
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

        stats = video_path.stat()
        payload = {
            "video_path": str(video_path),
            "video_size": stats.st_size,
            "video_mtime_ns": stats.st_mtime_ns,
            "audio_rate": Config.AUDIO_RATE,
            "whisper_model": Config.WHISPER_MODEL,
            "whisper_lang": Config.WHISPER_LANG,
            "whisper_beam": Config.WHISPER_BEAM,
            "translation_model": Config.TRANS_MODEL,
            "tts_voice": Config.TTS_VOICE,
            "duck_threshold": Config.DUCK_THRESH,
            "duck_ratio": Config.DUCK_RATIO,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return digest

    def process(self, video_path: str, use_cache: bool = True):
        video_path = Path(video_path).resolve()
        parent_dir = video_path.parent

        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        pipeline_cache_key = self._build_pipeline_cache_key(video_path)

        # 1. Cria diretório temporário ao lado do arquivo original
        temp_dir_name = f"temp_{video_path.stem}"
        temp_dir = parent_dir / temp_dir_name
        temp_dir.mkdir(exist_ok=True)

        self.log(f"📁 Pasta temporária criada: {temp_dir}")
        self.log(f"🧩 Cache key do pipeline: {pipeline_cache_key[:12]}...")

        context = {
            "video_path": str(video_path),
            "use_cache": use_cache,
            "segments": [],
            "project_dir": str(temp_dir),
            "pipeline_cache_key": pipeline_cache_key,
        }

        # Lista de fases a serem executadas
        pipeline_classes: List[Type[PipelinePhase]] = [
            ExtractionPhase,
            SeparationPhase,
            TranscriptionPhase,
            PostProcessingPhase,
            TranslationPhase,
            TTSPhase,
            AudioMixingPhase,
            RenderingPhase,
        ]

        try:
            pipeline_start = time.perf_counter()
            for PhaseClass in pipeline_classes:
                # Instancia fase apontando para o diretório temporário
                phase = PhaseClass(temp_dir, self.log)

                self.log(f"--- Iniciando Fase: {PhaseClass.__name__} ---")
                phase_start = time.perf_counter()
                context = phase.execute(context)
                phase_elapsed = time.perf_counter() - phase_start
                self.log(f"--- Fase concluída: {PhaseClass.__name__} ({phase_elapsed:.2f}s) ---")

                # Limpeza explícita após cada fase
                del phase
                ResourceManager.force_cleanup(self.log)

            # 2. Movimentação do arquivo final para fora do temp
            generated_video = Path(context["output_video_path"])
            final_destination = parent_dir / generated_video.name

            if final_destination.exists():
                self.log(
                    f"⚠️ Arquivo de saída já existe, substituindo: {final_destination.name}"
                )
                final_destination.unlink()  # Garante remoção segura antes de mover

            shutil.move(str(generated_video), str(final_destination))
            self.log(f"✅ Vídeo final salvo em: {final_destination}")
            self.log(f"🏁 Pipeline finalizado em {time.perf_counter() - pipeline_start:.2f}s")

            return str(final_destination)

        except Exception as e:
            self.log(f"[!] Erro Crítico no Pipeline: {e}")
            self.log(traceback.format_exc())
            raise e

        finally:
            # 3. Limpeza Final (Deleta a pasta temporária)
            ResourceManager.force_cleanup()

            if temp_dir.exists():
                if use_cache:
                    self.log(f"♻️ Cache preservado em: {temp_dir}")
                else:
                    try:
                        self.log(
                            f"🧹 Removendo arquivos temporários em: {temp_dir.name}..."
                        )
                        shutil.rmtree(temp_dir)
                        self.log("✨ Limpeza concluída.")
                    except Exception as e:
                        self.log(f"⚠️ Falha ao remover pasta temporária: {e}")
