import json
import asyncio
import torch
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Dict
import re
import edge_tts

from src.config import Config

# Bibliotecas de Mídia
from moviepy import VideoFileClip
from pydub import AudioSegment
from pydub.effects import normalize

# Importações locais
from .pipeline import PipelinePhase

# ==========================================
# FASES DE PROCESSAMENTO
# ==========================================

class ExtractionPhase(PipelinePhase):
    """Fase 1: Extrai áudio do vídeo para WAV."""

    def execute(self, context: Dict) -> Dict:
        video_path = Path(context["video_path"])
        output_audio = self.base_dir / "original.wav"

        if output_audio.exists() and context.get("use_cache"):
            self.log("Arquivo de áudio já existe (Cache).")
            try:
                context["video_duration"] = AudioSegment.from_file(output_audio).duration_seconds
            except:
                pass
        else:
            self.log(f"Extraindo áudio de {video_path.name}...")
            with VideoFileClip(str(video_path)) as video:
                video.audio.write_audiofile(
                    str(output_audio),
                    fps=Config.AUDIO_RATE,
                    nbytes=2,
                    codec="pcm_s16le",
                    logger=None,
                )
                context["video_duration"] = video.duration

        context["original_audio_path"] = str(output_audio)
        return context


class TranscriptionPhase(PipelinePhase):
    """Fase 2: Transcreve áudio (Suporta Faster-Whisper e OpenAI-Whisper)."""

    def execute(self, context: Dict) -> Dict:
        segments_path = self.base_dir / "segments.json"

        if segments_path.exists() and context.get("use_cache"):
            self.log("Transcrição encontrada em cache.")
            with open(segments_path, "r", encoding="utf-8") as f:
                context["segments"] = json.load(f)
            return context

        segments = []

        # ====================================================
        # MODO 1: Faster Whisper (NVIDIA / CPU Otimizado)
        # ====================================================
        if self.backend == "faster-whisper":
            self.log(f"Carregando Faster-Whisper em {self.device}...")
            from faster_whisper import WhisperModel

            # Ajusta precisão baseado no hardware
            if self.device == "cuda":
                compute_type = Config.WHISPER_COMPUTE # float16 geralmente
            else:
                compute_type = "int8" # CPU/Outros

            model = WhisperModel(Config.WHISPER_MODEL, device=self.device, compute_type=compute_type)

            self.log("Iniciando transcrição (Faster)...")
            segments_gen, _ = model.transcribe(
                context["original_audio_path"], 
                language=Config.WHISPER_LANG, 
                beam_size=Config.WHISPER_BEAM
            )

            for seg in segments_gen:
                segments.append(
                    {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                )
            
            del model

        # ====================================================
        # MODO 2: OpenAI Whisper (DirectML / Legacy / MPS)
        # ====================================================
        else:
            self.log(f"Carregando OpenAI-Whisper (Legacy) em {self.device}...")
            import whisper

            # Nota: 'self.device' aqui pode ser um objeto torch.device (DirectML) ou string
            # O whisper.load_model suporta ambos.
            model = whisper.load_model(Config.WHISPER_MODEL, device=self.device)

            self.log("Iniciando transcrição (Standard)...")
            
            # OpenAI Whisper retorna um dict completo, não um generator
            result = model.transcribe(
                context["original_audio_path"],
                language=Config.WHISPER_LANG,
                beam_size=Config.WHISPER_BEAM,
                fp16=False # Forçar FP32 para evitar erros em DirectML/CPU antigo
            )

            for seg in result.get("segments", []):
                segments.append(
                    {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
                )
            
            del model

        # Salva resultados
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        context["segments"] = segments
        self.log(f"Transcrição concluída ({len(segments)} segmentos).")
        return context


class PostProcessingPhase(PipelinePhase):
    """Fase Intermediária: Limpeza e correção dos segmentos transcritos."""

    def execute(self, context: Dict) -> Dict:
        segments = context.get("segments", [])

        if not segments:
            self.log("Nenhum segmento para processar.")
            return context

        self.log(f"Processando {len(segments)} segmentos de texto...")

        cleaned_segments = []
        for seg in segments:
            original_text = seg["text"]

            # 1. Remover alucinações comuns do Whisper
            new_text = re.sub(r"\[.*?\]|\(.*?\)", "", original_text)

            # 2. Remover espaços extras
            new_text = new_text.strip()

            # 3. Filtrar segmentos vazios ou muito curtos
            if len(new_text) < 2:
                continue

            seg["text"] = new_text
            cleaned_segments.append(seg)

        processed_path = self.base_dir / "segments_cleaned.json"
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_segments, f, ensure_ascii=False, indent=2)

        context["segments"] = cleaned_segments
        return context


class TranslationPhase(PipelinePhase):
    """Fase 3: Traduz texto usando MarianMT (HuggingFace)."""

    def execute(self, context: Dict) -> Dict:
        segments = context.get("segments", [])

        if not segments:
            self.log("Nenhum segmento detectado para tradução.")
            return context

        if "text_pt" in segments[0] and context.get("use_cache"):
            self.log("Tradução já presente nos segmentos.")
            return context

        self.log(f"Carregando MarianMT em {self.device}...")
        from transformers import MarianMTModel, MarianTokenizer

        model_name = Config.TRANS_MODEL
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        
        # 'self.device' agora está limpo (sem backend info)
        model = MarianMTModel.from_pretrained(model_name).to(self.device)

        # Otimização FP16 apenas para CUDA/MPS explícitos
        if isinstance(self.device, str) and self.device in ["cuda", "mps"]:
            model = model.half()

        self.log(f"Traduzindo {len(segments)} segmentos...")

        # Se for CPU ou DirectML (objeto torch.device), usa batch menor se necessário
        is_gpu = str(self.device) != "cpu"
        batch_size = Config.TRANS_BATCH if is_gpu else 8
        
        texts = [s["text"] for s in segments]
        translated_texts = []
        inputs = None

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs)
            decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            translated_texts.extend(decoded)

        for i, txt in enumerate(translated_texts):
            segments[i]["text_pt"] = txt

        with open(self.base_dir / "segments.json", "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        self.log("Descarregando MarianMT...")
        del model
        del tokenizer
        if inputs is not None:
            del inputs

        return context


class TTSPhase(PipelinePhase):
    """Fase 4: Gera áudios em português usando Edge-TTS."""

    def execute(self, context: Dict) -> Dict:
        segments = context["segments"]
        chunks_dir = self.base_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        tasks = []
        for i, seg in enumerate(segments):
            path = chunks_dir / f"seg_{i:04d}.wav"
            if not (path.exists() and context.get("use_cache")):
                duration = seg["end"] - seg["start"]
                tasks.append((seg["text_pt"], str(path), duration))

        if not tasks:
            self.log("Todos os áudios já existem em cache.")
            return context

        self.log(f"Gerando {len(tasks)} arquivos de áudio...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.TTS_WORKERS) as executor:
            list(executor.map(self._generate_single, tasks))

        return context

    def _generate_single(self, args):
        text, path, target_dur = args
        try:
            asyncio.run(self._synthesize(text, path, target_dur))
            seg = AudioSegment.from_file(path)
            seg = normalize(seg).high_pass_filter(80)
            seg.export(path, format="wav")
            del seg
        except Exception as e:
            self.log(f"Erro TTS: {e}")

    async def _synthesize(self, text, path, target_dur):
        voice = Config.TTS_VOICE
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)

        seg = AudioSegment.from_file(path)
        actual_dur = len(seg) / 1000.0

        if actual_dur > target_dur:
            ratio = (actual_dur / target_dur) - 1
            percentage = min(int(ratio * 100) + 5, 50)
            if percentage > 5:
                rate_str = f"+{percentage}%"
                communicate = edge_tts.Communicate(text, voice, rate=rate_str)
                await communicate.save(path)


class AudioMixingPhase(PipelinePhase):
    """
    Fase 5: Mixagem Otimizada com FFmpeg (Sidechain Ducking).
    Reduz drasticamente o uso de RAM e processa o ducking de forma automática.
    """

    def execute(self, context: Dict) -> Dict:
        output_mix = self.base_dir / "final_mix.wav"

        if output_mix.exists() and context.get("use_cache"):
            self.log("Mixagem já existe (Cache).")
            context["final_audio_path"] = str(output_mix)
            return context

        self.log("Iniciando mixagem com correção de sincronia (Anti-Drift)...")

        orig_path = Path(context["original_audio_path"])
        segments = context["segments"]
        chunks_dir = self.base_dir / "chunks"
        speech_track_path = self.base_dir / "speech_track.wav"

        duration = context.get("video_duration")
        if not duration:
            duration = self._get_duration(orig_path)
            context["video_duration"] = duration

        self.log(f"Montando faixa de voz ({duration:.2f}s) em Mono...")
        
        timeline_parts = []
        cursor_ms = 0
        
        # Garante ordenação temporal
        segments.sort(key=lambda x: x["start"])

        # Import para o speedup de emergência
        from pydub.effects import speedup

        for i, seg in enumerate(segments):
            chunk_path = chunks_dir / f"seg_{i:04d}.wav"
            if not chunk_path.exists():
                continue

            start_ms = int(seg["start"] * 1000)
            
            gap = start_ms - cursor_ms

            if gap > 0:
                timeline_parts.append(AudioSegment.silent(duration=gap, frame_rate=44100).set_channels(1))
                cursor_ms += gap # Agora cursor_ms == start_ms
            
            # Carrega o áudio atual
            voice_chunk = AudioSegment.from_file(chunk_path).set_channels(1)

            # 2. Verificar quanto tempo temos até o PRÓXIMO segmento
            if i < len(segments) - 1:
                next_start_ms = int(segments[i+1]["start"] * 1000)
                time_until_next = next_start_ms - cursor_ms # Tempo real disponível a partir de AGORA
                
                # Se o áudio é maior que o espaço até o próximo (com uma margem de erro pequena)
                if len(voice_chunk) > time_until_next and time_until_next > 100:
                    self.log(f"⚠️ Ajustando segmento {i} para evitar cascata: {len(voice_chunk)}ms -> {time_until_next}ms")
                    
                    ratio = len(voice_chunk) / time_until_next 
                    
                    if ratio > 1.0:
                        voice_chunk = speedup(voice_chunk, playback_speed=ratio * 1.05, chunk_size=50, crossfade=0)
                        
                        if len(voice_chunk) > time_until_next:
                             voice_chunk = voice_chunk[:time_until_next]

            timeline_parts.append(voice_chunk)
            cursor_ms += len(voice_chunk)

        # Preenche o restante
        total_dur_ms = int(duration * 1000)
        final_gap = total_dur_ms - cursor_ms
        if final_gap > 0:
            timeline_parts.append(AudioSegment.silent(duration=final_gap, frame_rate=44100).set_channels(1))

        if timeline_parts:
            speech_track = sum(timeline_parts, AudioSegment.empty())
        else:
            speech_track = AudioSegment.silent(duration=total_dur_ms, frame_rate=44100).set_channels(1)

        speech_track.export(str(speech_track_path), format="wav")
        del speech_track
        del timeline_parts
        import gc
        gc.collect()

        cmd = [
            "ffmpeg", "-y", "-i", str(orig_path), "-i", str(speech_track_path),
            "-filter_complex",
            f"[1:a]asplit=2[sc][voice_mix];[0:a][sc]sidechaincompress=threshold={Config.DUCK_THRESH}:ratio={Config.DUCK_ATTACK}:attack=50:release={Config.DUCK_RELEASE}[ducked_bg];[ducked_bg][voice_mix]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]",
            "-map", "[out]", str(output_mix)
        ]

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            self.log(f"Erro no FFmpeg Mixing: {process.stderr.decode()}")
            raise Exception("Falha na mixagem FFmpeg")

        context["final_audio_path"] = str(output_mix)
        try:
            speech_track_path.unlink()
        except:
            pass

        return context

    def _get_duration(self, file_path: Path) -> float:
        """Obtém duração precisa usando FFprobe (evita carregar arquivo na RAM)."""
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            str(file_path)
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())
        except Exception:
            self.log("Aviso: Não foi possível obter duração via ffprobe. Usando fallback.")
            return 0.0


class RenderingPhase(PipelinePhase):
    """Fase 6: Combina vídeo original com novo áudio usando FFmpeg."""

    def execute(self, context: Dict) -> Dict:
        self.log("Renderizando vídeo final com FFmpeg...")

        video_path = context["video_path"]
        audio_path = context["final_audio_path"]

        output_video = self.base_dir / f"{Path(video_path).stem}{Config.OUTPUT_SUFFIX}.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(output_video),
        ]

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            self.log(f"Erro no FFmpeg: {process.stderr.decode()}")
            raise Exception("Falha na renderização FFmpeg")

        context["output_video_path"] = str(output_video)
        self.log(f"Vídeo renderizado em: {output_video}")
        return context