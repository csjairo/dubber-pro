import os
import json
import asyncio
import torch
import librosa
import subprocess # Para chamar o FFmpeg direto
import numpy as np
import concurrent.futures # Para o Multithreading
from pathlib import Path
from typing import List, Dict

# Bibliotecas de Mídia
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from pydub import AudioSegment
from pydub.effects import normalize

# Bibliotecas de IA
from transformers import MarianMTModel, MarianTokenizer
from faster_whisper import WhisperModel # Otimização 1: Faster Whisper

# ==========================================
# 1. MÓDULO DE DETECÇÃO DE HARDWARE UNIVERSAL
# ==========================================
def get_best_device():
    """Detecta o melhor hardware disponível (NVIDIA, AMD, Intel ou Apple)."""
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"   # Apple Silicon (M1/M2/M3)
    return "cpu"

# ==========================================
# 2. MÓDULO DE TTS (Edge-TTS)
# ==========================================
class TTSModule:
    def __init__(self, voice: str = "pt-BR-AntonioNeural", logger=None):
        self.voice = voice
        self.logger = logger

    def log(self, msg):
        if self.logger: self.logger(msg)
        else: print(msg)

    async def _synthesize_raw(self, text: str, output_path: str, rate: str = "+0%"):
        import edge_tts
        communicate = edge_tts.Communicate(text, self.voice, rate=rate)
        await communicate.save(output_path)

    def generate_audio(self, text: str, output_path: str, target_duration: float = None) -> float:
        try:
            rate_str = "+0%"
            # Se precisar ajustar a duração (Speed-up)
            if target_duration:
                temp_path = output_path + ".temp.mp3"
                asyncio.run(self._synthesize_raw(text, temp_path, "+0%"))
                
                # Verifica duração
                base_dur = librosa.get_duration(path=temp_path)
                os.remove(temp_path)
                
                if base_dur > target_duration:
                    speed_up = int(((base_dur / target_duration) - 1) * 100)
                    speed_up = min(max(speed_up, 0), 50) # Limita a 50% de aceleração
                    rate_str = f"+{speed_up}%"
            
            # Gera o final
            asyncio.run(self._synthesize_raw(text, output_path, rate_str))
            
            # Retorna duração real para logs
            return librosa.get_duration(path=output_path)
        except Exception as e:
            self.log(f"❌ Erro TTS: {e}")
            return 0.0

# ==========================================
# 3. MÓDULO DE ÁUDIO
# ==========================================
class AudioEngine:
    def __init__(self, logger=None):
        self.logger = logger

    def log(self, msg):
        if self.logger: self.logger(msg)
        else: print(msg)

    def apply_vocal_chain(self, audio_path: str):
        try:
            seg = AudioSegment.from_file(audio_path)
            # Filtro passa-alta e compressão básica
            seg = normalize(seg).high_pass_filter(80)
            seg = seg.compress_dynamic_range(threshold=-20.0, ratio=3.0, attack=5.0, release=50.0)
            seg.export(audio_path, format="wav", parameters=["-ar", "44100"])
        except Exception as e:
            self.log(f"⚠️ Erro no Vocal Chain: {e}")

    def create_ducking_track(self, original_audio_path: str, segments: List[Dict], output_path: str):
        self.log("🎚️ Aplicando Ducking (Algoritmo Linear)...")
        original = AudioSegment.from_file(original_audio_path)
        
        if not segments:
            original.export(output_path, format="wav", parameters=["-ar", "44100"])
            return

        FADE_TIME = 300
        DUCK_VOL = -15
        
        sorted_segs = sorted(segments, key=lambda x: x['start'])
        final_parts = []
        current_pos = 0
        
        for seg in sorted_segs:
            start_ms = max(0, int(seg['start'] * 1000) - 150)
            end_ms = min(len(original), int(seg['end'] * 1000) + 150)
            
            if start_ms >= end_ms: continue
            
            if start_ms > current_pos:
                final_parts.append(original[current_pos:start_ms])
            
            if end_ms > current_pos:
                effective_start = max(start_ms, current_pos)
                duck_part = original[effective_start:end_ms].apply_gain(DUCK_VOL).fade_in(FADE_TIME).fade_out(FADE_TIME)
                final_parts.append(duck_part)
                current_pos = end_ms
        
        if current_pos < len(original):
            final_parts.append(original[current_pos:])
            
        ducked_audio = sum(final_parts)
        ducked_audio.export(output_path, format="wav", parameters=["-ar", "44100"])

# ==========================================
# 4. TRADUÇÃO (Mantido igual)
# ==========================================
class TranslationModule:
    def __init__(self, device="cpu"):
        model_name = 'Helsinki-NLP/opus-mt-tc-big-en-pt'
        self.device = device
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
        
        if self.device in ["cuda", "mps"]:
            self.model = self.model.half()

    def translate_batch(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            translated = self.model.generate(**inputs)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# ==========================================
# 5. ORQUESTRADOR UNIVERSAL (OTIMIZADO)
# ==========================================
class Dubber:
    def __init__(self, logger_func=None):
        self.logger = logger_func
        self.device = get_best_device()
        self.base_dir = Path("workspace_universal")
        self.base_dir.mkdir(exist_ok=True)
        
        self.log(f"🚀 DubberPro UNIVERSAL iniciado em: {self.device.upper()}")
        
        # OTIMIZAÇÃO 1: Faster Whisper (mais rápido e gasta menos RAM)
        # compute_type="float16" (GPU) ou "int8" (CPU/Mais rápido)
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.whisper = WhisperModel("small", device=self.device, compute_type=compute_type)
        
        self.translator = TranslationModule(self.device)
        self.tts = TTSModule(logger=self.log)
        self.audio_engine = AudioEngine(logger=self.log)

    def log(self, msg):
        if self.logger: self.logger(msg)
        else: print(msg)

    # Função Worker para Threads
    def _worker_generate_audio(self, task_data):
        text, path, duration = task_data
        try:
            # Gera o áudio (Asyncio dentro da thread é isolado e seguro)
            real_dur = self.tts.generate_audio(text, path, target_duration=duration)
            if real_dur > 0:
                self.audio_engine.apply_vocal_chain(path)
                return True
            return False
        except Exception as e:
            self.log(f"❌ Erro na thread: {e}")
            return False

    def process(self, video_path: str, use_cache: bool = True):
        video_path = Path(video_path)
        project_dir = self.base_dir / video_path.stem
        project_dir.mkdir(exist_ok=True)
        
        # 1. Extração
        orig_audio_path = project_dir / "original.wav"
        if not (use_cache and orig_audio_path.exists()):
            self.log("🔊 Extraindo áudio original...")
            with VideoFileClip(str(video_path)) as video:
                video.audio.write_audiofile(str(orig_audio_path), fps=44100, nbytes=2, codec='pcm_s16le', logger=None)
        
        # 2. Transcrição (Com Faster-Whisper)
        segments_cache = project_dir / "segments.json"
        if use_cache and segments_cache.exists():
            with open(segments_cache, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        else:
            self.log(f"📝 Transcrevendo (Faster-Whisper / {self.device})...")
            
            # Faster-Whisper retorna um gerador
            segments_gen, info = self.whisper.transcribe(
                str(orig_audio_path), 
                language="en",
                beam_size=5 
            )

            # Convertemos para lista de dicts (para compatibilidade com o resto do código)
            segments = []
            for seg in segments_gen:
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip()
                })

            self.log(f"🌍 Traduzindo {len(segments)} segmentos...")
            batch_size = 32 if self.device != "cpu" else 8
            
            # Processamento em lote da tradução
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i+batch_size]
                texts = [s['text'] for s in batch]
                translated = self.translator.translate_batch(texts)
                for j, t in enumerate(translated):
                    segments[i+j]['text_pt'] = t
            
            with open(segments_cache, 'w', encoding='utf-8') as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

        # 3. Geração de Áudio (OTIMIZAÇÃO 2: Multithreading)
        self.log(f"🎙️ Gerando vozes (Pool de 5 Threads)...")
        audio_chunks_dir = project_dir / "chunks"
        audio_chunks_dir.mkdir(exist_ok=True)
        
        tasks_to_process = []
        for i, seg in enumerate(segments):
            chunk_path = audio_chunks_dir / f"seg_{i:04d}.wav"
            if not (use_cache and chunk_path.exists()):
                duration = seg['end'] - seg['start']
                tasks_to_process.append((seg['text_pt'], str(chunk_path), duration))

        if tasks_to_process:
            self.log(f"🔥 Processando {len(tasks_to_process)} novos áudios...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # O list() força a espera de todas as tarefas completarem
                list(executor.map(self._worker_generate_audio, tasks_to_process))
        else:
            self.log("✅ Áudios em cache.")

        # 3.1 Carregamento Seguro (Pós-Threads)
        self.log("🎞️ Montando clipes de áudio...")
        dub_clips = []
        for i, seg in enumerate(segments):
            chunk_path = audio_chunks_dir / f"seg_{i:04d}.wav"
            if chunk_path.exists():
                dub_clips.append(AudioFileClip(str(chunk_path)).with_start(seg['start']))

        # 4. Mixagem (Ducking)
        ducked_bg_path = project_dir / "bg_ducked.wav"
        if not (use_cache and ducked_bg_path.exists()):
            self.audio_engine.create_ducking_track(str(orig_audio_path), segments, str(ducked_bg_path))
        
        # 5. Renderização (OTIMIZAÇÃO 3: FFmpeg Direto)
        self.log("🎬 Renderizando Final (Modo Rápido: Cópia de Vídeo)...")
        
        # Primeiro, geramos apenas o áudio final mixado usando MoviePy (rápido)
        final_audio_path = project_dir / "final_mix.wav"
        
        # Cria a composição de áudio
        bg_clip = AudioFileClip(str(ducked_bg_path))
        if dub_clips:
            final_audio = CompositeAudioClip([bg_clip] + dub_clips)
        else:
            final_audio = bg_clip
            
        # Pega a duração original do vídeo para cortar o áudio se precisar
        with VideoFileClip(str(video_path)) as video:
            video_duration = video.duration
            
        final_audio = final_audio.with_duration(video_duration)
        final_audio.write_audiofile(str(final_audio_path), fps=44100, logger=None)
        
        # Agora usamos FFmpeg para juntar Video Original + Audio Novo (Sem re-renderizar video)
        output_video = self.base_dir / f"{video_path.stem}_DUB_FAST.mp4"
        
        try:
            # Comando: ffmpeg -i VIDEO -i AUDIO -c:v copy -c:a aac -map 0:v -map 1:a OUTPUT
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(final_audio_path),
                "-c:v", "copy",        # COPIA o stream de vídeo (Ultra rápido)
                "-c:a", "aac",         # Codifica o áudio para AAC
                "-strict", "experimental",
                "-map", "0:v:0",       # Pega o vídeo do primeiro arquivo
                "-map", "1:a:0",       # Pega o áudio do segundo arquivo (nossa mixagem)
                "-shortest",           # Garante que termine com o menor stream
                str(output_video)
            ]
            
            # Executa silenciosamente
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.log(f"✅ Renderização Instantânea concluída: {output_video}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("⚠️ FFmpeg não encontrado ou erro. Usando renderizador lento (fallback)...")
            # Fallback para o método lento se o usuário não tiver FFmpeg no PATH
            with VideoFileClip(str(video_path)) as video:
                video.with_audio(final_audio).write_videofile(
                    str(output_video), codec="libx264", audio_codec="aac", threads=4, preset="ultrafast"
                )