import json
import asyncio
import torch
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Dict

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
        video_path = Path(context['video_path'])
        output_audio = self.base_dir / "original.wav"
        
        if output_audio.exists() and context.get('use_cache'):
            self.log("Arquivo de áudio já existe (Cache).")
        else:
            self.log(f"Extraindo áudio de {video_path.name}...")
            with VideoFileClip(str(video_path)) as video:
                video.audio.write_audiofile(
                    str(output_audio), 
                    fps=44100, 
                    nbytes=2, 
                    codec='pcm_s16le', 
                    logger=None
                )
                context['video_duration'] = video.duration
        
        context['original_audio_path'] = str(output_audio)
        return context

class TranscriptionPhase(PipelinePhase):
    """Fase 2: Transcreve áudio usando Faster-Whisper."""
    
    def execute(self, context: Dict) -> Dict:
        segments_path = self.base_dir / "segments.json"
        
        if segments_path.exists() and context.get('use_cache'):
            self.log("Transcrição encontrada em cache.")
            with open(segments_path, 'r', encoding='utf-8') as f:
                context['segments'] = json.load(f)
            return context

        self.log(f"Carregando WhisperModel em {self.device}...")
        from faster_whisper import WhisperModel
        
        compute_type = "float16" if self.device == "cuda" else "int8"
        model = WhisperModel("tiny", device=self.device, compute_type=compute_type)
        
        self.log("Iniciando transcrição...")
        segments_gen, _ = model.transcribe(
            context['original_audio_path'], 
            language="en",
            beam_size=5
        )
        
        segments = []
        for seg in segments_gen:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })
            
        with open(segments_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
            
        context['segments'] = segments
        
        self.log("Descarregando Whisper...")
        del model
        return context

class TranslationPhase(PipelinePhase):
    """Fase 3: Traduz texto usando MarianMT (HuggingFace)."""
    
    def execute(self, context: Dict) -> Dict:
        segments = context.get('segments', [])
        
        if segments and 'text_pt' in segments[0] and context.get('use_cache'):
            self.log("Tradução já presente nos segmentos.")
            return context

        self.log(f"Carregando MarianMT em {self.device}...")
        from transformers import MarianMTModel, MarianTokenizer
        
        model_name = 'Helsinki-NLP/opus-mt-tc-big-en-pt'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(self.device)
        
        if self.device in ["cuda", "mps"]:
            model = model.half()

        self.log(f"Traduzindo {len(segments)} segmentos...")
        
        batch_size = 32 if self.device != "cpu" else 8
        texts = [s['text'] for s in segments]
        translated_texts = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs)
            decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            translated_texts.extend(decoded)

        for i, txt in enumerate(translated_texts):
            segments[i]['text_pt'] = txt
            
        with open(self.base_dir / "segments.json", 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        self.log("Descarregando MarianMT...")
        del model
        del tokenizer
        del inputs
        return context

class TTSPhase(PipelinePhase):
    """Fase 4: Gera áudios em português usando Edge-TTS."""
    
    def execute(self, context: Dict) -> Dict:
        segments = context['segments']
        chunks_dir = self.base_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        tasks = []
        for i, seg in enumerate(segments):
            path = chunks_dir / f"seg_{i:04d}.wav"
            if not (path.exists() and context.get('use_cache')):
                duration = seg['end'] - seg['start']
                tasks.append((seg['text_pt'], str(path), duration))

        if not tasks:
            self.log("Todos os áudios já existem em cache.")
            return context

        self.log(f"Gerando {len(tasks)} arquivos de áudio...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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
        import edge_tts
        voice = "pt-BR-AntonioNeural"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)

class AudioMixingPhase(PipelinePhase):
    """Fase 5: Mixagem do áudio traduzido sobre o fundo (ducking)."""
    
    def execute(self, context: Dict) -> Dict:
        self.log("Iniciando mixagem de áudio...")
        
        orig_path = context['original_audio_path']
        segments = context['segments']
        chunks_dir = self.base_dir / "chunks"
        output_mix = self.base_dir / "final_mix.wav"
        
        bg_audio = AudioSegment.from_file(orig_path)
        ducked_bg = self._apply_ducking(bg_audio, segments)
        
        final_audio = ducked_bg
        for i, seg in enumerate(segments):
            chunk_path = chunks_dir / f"seg_{i:04d}.wav"
            if chunk_path.exists():
                voice = AudioSegment.from_file(chunk_path)
                start_ms = int(seg['start'] * 1000)
                final_audio = final_audio.overlay(voice, position=start_ms)
        
        final_audio.export(str(output_mix), format="wav")
        context['final_audio_path'] = str(output_mix)
        
        del bg_audio
        del final_audio
        del ducked_bg
        return context

    def _apply_ducking(self, original: AudioSegment, segments: List[Dict]) -> AudioSegment:
        # Implementação simplificada de ducking (silenciar original)
        return original.apply_gain(0)

class RenderingPhase(PipelinePhase):
    """Fase 6: Combina vídeo original com novo áudio usando FFmpeg."""
    
    def execute(self, context: Dict) -> Dict:
        self.log("Renderizando vídeo final com FFmpeg...")
        
        video_path = context['video_path']
        audio_path = context['final_audio_path']
        
        output_video = self.base_dir / f"{Path(video_path).stem}_DUB_PRO.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_video)
        ]
        
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0:
            self.log(f"Erro no FFmpeg: {process.stderr.decode()}")
            raise Exception("Falha na renderização FFmpeg")
            
        context['output_video_path'] = str(output_video)
        self.log(f"Vídeo renderizado em: {output_video}")
        return context