import os
import json
import asyncio
import torch
import whisper
import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from transformers import MarianMTModel, MarianTokenizer
from pydub import AudioSegment
from pydub.effects import normalize

# ==========================================
# 1. MÓDULO DE DETECÇÃO DE HARDWARE UNIVERSAL
# ==========================================
def get_best_device():
    """Detecta o melhor hardware disponível (NVIDIA, AMD, Intel ou Apple)."""
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"   # Apple Silicon (M1/M2/M3)
    # Para AMD/Intel no Windows, o PyTorch usa DirectML (requer torch-directml)
    # Mas o Whisper padrão e Transformers usam CPU se não houver CUDA/MPS.
    # A solução universal é garantir que o código não quebre e use o que puder.
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
            if target_duration:
                temp_path = output_path + ".temp.mp3"
                asyncio.run(self._synthesize_raw(text, temp_path, "+0%"))
                base_dur = librosa.get_duration(path=temp_path)
                os.remove(temp_path)
                if base_dur > target_duration:
                    speed_up = int(((base_dur / target_duration) - 1) * 100)
                    speed_up = min(max(speed_up, 0), 50)
                    rate_str = f"+{speed_up}%"
            
            asyncio.run(self._synthesize_raw(text, output_path, rate_str))
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
            seg = normalize(seg).high_pass_filter(80)
            seg = seg.compress_dynamic_range(threshold=-20.0, ratio=3.0, attack=5.0, release=50.0)
            seg.export(audio_path, format="wav", parameters=["-ar", "44100"])
        except Exception as e:
            self.log(f"⚠️ Erro no Vocal Chain: {e}")

    def create_ducking_track(self, original_audio_path: str, segments: List[Dict], output_path: str):
        self.log("🎚️ Aplicando Ducking...")
        original = AudioSegment.from_file(original_audio_path)
        ducked_audio = original
        FADE_TIME, DUCK_VOL = 300, -15
        for seg in sorted(segments, key=lambda x: x['start']):
            start_ms = max(0, int(seg['start'] * 1000) - 150)
            end_ms = min(len(original), int(seg['end'] * 1000) + 150)
            if start_ms >= end_ms: continue
            duck_part = original[start_ms:end_ms].apply_gain(DUCK_VOL).fade_in(FADE_TIME).fade_out(FADE_TIME)
            ducked_audio = ducked_audio[:start_ms] + duck_part + ducked_audio[end_ms:]
        ducked_audio.export(output_path, format="wav", parameters=["-ar", "44100"])

# ==========================================
# 4. TRADUÇÃO COM ACELERAÇÃO UNIVERSAL
# ==========================================
class TranslationModule:
    def __init__(self, device="cpu"):
        model_name = 'Helsinki-NLP/opus-mt-tc-big-en-pt'
        self.device = device
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
        
        # Se for GPU (NVIDIA ou Apple), usa precisão mista para velocidade
        if self.device in ["cuda", "mps"]:
            self.model = self.model.half()

    def translate_batch(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            translated = self.model.generate(**inputs)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# ==========================================
# 5. ORQUESTRADOR UNIVERSAL
# ==========================================
class DubberPro:
    def __init__(self, logger_func=None):
        self.logger = logger_func
        self.device = get_best_device()
        self.base_dir = Path("workspace_universal")
        self.base_dir.mkdir(exist_ok=True)
        
        self.log(f"🚀 DubberPro UNIVERSAL iniciado em: {self.device.upper()}")
        
        # Whisper detecta automaticamente CUDA ou CPU. 
        # Para AMD/Intel, ele usará CPU de forma otimizada via bibliotecas BLAS.
        self.whisper = whisper.load_model("small", device=self.device)
        
        self.translator = TranslationModule(self.device)
        self.tts = TTSModule(logger=self.log)
        self.audio_engine = AudioEngine(logger=self.log)

    def log(self, msg):
        if self.logger: self.logger(msg)
        else: print(msg)

    def process(self, video_path: str, use_cache: bool = True):
        video_path = Path(video_path)
        project_dir = self.base_dir / video_path.stem
        project_dir.mkdir(exist_ok=True)
        
        # 1. Extração
        orig_audio_path = project_dir / "original.wav"
        if not (use_cache and orig_audio_path.exists()):
            self.log("🔊 Extraindo áudio...")
            with VideoFileClip(str(video_path)) as video:
                video.audio.write_audiofile(str(orig_audio_path), fps=44100, nbytes=2, codec='pcm_s16le', logger=None)
        
        # 2. Transcrição (Auto-detecção de Hardware)
        segments_cache = project_dir / "segments.json"
        if use_cache and segments_cache.exists():
            with open(segments_cache, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        else:
            self.log(f"📝 Transcrevendo ({self.device})...")
            # fp16=True apenas para CUDA. Para outros, False garante compatibilidade.
            result = self.whisper.transcribe(str(orig_audio_path), language="en", fp16=(self.device == "cuda"))
            segments = result['segments']

            self.log("🌍 Traduzindo...")
            batch_size = 32 if self.device != "cpu" else 8
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i+batch_size]
                texts = [s['text'].strip() for s in batch]
                translated = self.translator.translate_batch(texts)
                for j, t in enumerate(translated):
                    segments[i+j]['text_pt'] = t
            
            with open(segments_cache, 'w', encoding='utf-8') as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

        # 3. Geração de Áudio
        self.log("🎙️ Gerando vozes...")
        audio_chunks_dir = project_dir / "chunks"
        audio_chunks_dir.mkdir(exist_ok=True)
        dub_clips = []
        for i, seg in enumerate(segments):
            chunk_path = audio_chunks_dir / f"seg_{i:04d}.wav"
            if not (use_cache and chunk_path.exists()):
                self.tts.generate_audio(seg['text_pt'], str(chunk_path), target_duration=seg['end'] - seg['start'])
                self.audio_engine.apply_vocal_chain(str(chunk_path))
            dub_clips.append(AudioFileClip(str(chunk_path)).with_start(seg['start']))

        # 4. Mixagem
        ducked_bg_path = project_dir / "bg_ducked.wav"
        if not (use_cache and ducked_bg_path.exists()):
            self.audio_engine.create_ducking_track(str(orig_audio_path), segments, str(ducked_bg_path))
        
        # 5. Renderização Final Otimizada
        self.log("🎬 Renderizando (Multi-core Hardware)...")
        with VideoFileClip(str(video_path)) as video:
            bg_clip = AudioFileClip(str(ducked_bg_path))
            final_audio = CompositeAudioClip([bg_clip] + dub_clips).with_duration(video.duration)
            output_video = self.base_dir / f"{video_path.stem}_UNIVERSAL_DUB.mp4"
            
            # Otimização de Renderização:
            # - threads: Usa todos os núcleos da CPU (AMD/Intel/NVIDIA)
            # - preset: 'ultrafast' para velocidade máxima
            final_video = video.with_audio(final_audio)
            final_video.write_videofile(
                str(output_video), 
                codec="libx264", 
                audio_codec="aac", 
                threads=os.cpu_count(), 
                preset="ultrafast",
                logger=None
            )
            
        self.log(f"✅ Sucesso! Salvo em: {output_video}")
