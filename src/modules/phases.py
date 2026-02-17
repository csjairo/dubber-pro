import asyncio
import concurrent.futures
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import edge_tts
import torch
from moviepy import VideoFileClip
from pydub import AudioSegment
from pydub.effects import normalize, speedup
from transformers import MarianMTModel, MarianTokenizer

from src.config import Config
from .pipeline import PipelinePhase


def _file_size_mb(path: Path) -> str:
    if not path.exists():
        return "0.00 MB"
    return f"{path.stat().st_size / (1024 * 1024):.2f} MB"


def _cache_meta_path(base_dir: Path, phase_name: str) -> Path:
    return base_dir / f".cache_{phase_name}.json"


def _signature_hash(signature: Dict) -> str:
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_cache_valid(
    base_dir: Path,
    phase_name: str,
    pipeline_cache_key: str,
    signature: Dict,
) -> Tuple[bool, str]:
    meta_path = _cache_meta_path(base_dir, phase_name)
    if not meta_path.exists():
        return False, "metadado ausente"

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        return False, f"metadado inválido: {exc}"

    expected_signature = _signature_hash(signature)
    if meta.get("pipeline_cache_key") != pipeline_cache_key:
        return False, "chave do pipeline mudou"
    if meta.get("signature_hash") != expected_signature:
        return False, "configuração da fase mudou"
    return True, "cache válido"


def _write_cache_meta(
    base_dir: Path,
    phase_name: str,
    pipeline_cache_key: str,
    signature: Dict,
) -> None:
    meta_path = _cache_meta_path(base_dir, phase_name)
    meta = {
        "pipeline_cache_key": pipeline_cache_key,
        "signature_hash": _signature_hash(signature),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _ensure_segment_ids(segments: List[Dict]) -> bool:
    changed = False
    seen = set()
    next_id = 0

    for seg in segments:
        seg_id = seg.get("seg_id")
        valid = isinstance(seg_id, int) and seg_id >= 0 and seg_id not in seen
        if not valid:
            while next_id in seen:
                next_id += 1
            seg["seg_id"] = next_id
            changed = True
            seg_id = next_id

        seen.add(seg_id)
        next_id = max(next_id, seg_id + 1)

    return changed


class ExtractionPhase(PipelinePhase):
    """Fase 1: Extrai áudio do vídeo para WAV."""

    def execute(self, context: Dict) -> Dict:
        video_path = Path(context["video_path"])
        output_audio = self.base_dir / "original.wav"
        use_cache = bool(context.get("use_cache"))
        pipeline_key = context.get("pipeline_cache_key", "")
        signature = {"audio_rate": Config.AUDIO_RATE}

        self.log(f"Input vídeo: {video_path}")

        cache_valid, cache_reason = _is_cache_valid(
            self.base_dir, "extraction", pipeline_key, signature
        )
        if output_audio.exists() and use_cache and cache_valid:
            self.log(
                f"Cache HIT (extração): {output_audio.name} ({_file_size_mb(output_audio)})"
            )
            try:
                context["video_duration"] = AudioSegment.from_file(output_audio).duration_seconds
            except Exception as exc:
                self.log(
                    f"Falha ao ler duração do áudio em cache: {exc}. Recalculando pelo vídeo..."
                )
                with VideoFileClip(str(video_path)) as video:
                    context["video_duration"] = video.duration
        else:
            if use_cache and output_audio.exists() and not cache_valid:
                self.log(f"Cache MISS (extração): {cache_reason}. Regerando áudio.")
            self.log(f"Extraindo áudio de {video_path.name}...")
            with VideoFileClip(str(video_path)) as video:
                if video.audio is None:
                    raise RuntimeError("O vídeo não possui trilha de áudio.")
                video.audio.write_audiofile(
                    str(output_audio),
                    fps=Config.AUDIO_RATE,
                    nbytes=2,
                    codec="pcm_s16le",
                    logger=None,
                )
                context["video_duration"] = video.duration
            _write_cache_meta(self.base_dir, "extraction", pipeline_key, signature)

        context["original_audio_path"] = str(output_audio)
        self.log(
            f"Output áudio: {output_audio} ({_file_size_mb(output_audio)}), "
            f"duração={context.get('video_duration', 0):.2f}s"
        )
        return context


class SeparationPhase(PipelinePhase):
    """Fase 2: Realce de fala para transcrição + trilha base para mixagem."""

    def execute(self, context: Dict) -> Dict:
        original_audio = Path(context["original_audio_path"])
        speech_path = self.base_dir / "speech_enhanced.wav"
        background_path = self.base_dir / "background.wav"
        use_cache = bool(context.get("use_cache"))
        pipeline_key = context.get("pipeline_cache_key", "")
        signature = {
            "enabled": Config.SOURCE_SEPARATION_ENABLED,
            "highpass_hz": Config.SEPARATION_HIGHPASS_HZ,
            "lowpass_hz": Config.SEPARATION_LOWPASS_HZ,
        }

        if not Config.SOURCE_SEPARATION_ENABLED:
            self.log("Separação desativada por configuração. Usando áudio original.")
            context["transcription_audio_path"] = str(original_audio)
            context["background_audio_path"] = str(original_audio)
            return context

        cache_valid, cache_reason = _is_cache_valid(
            self.base_dir, "separation", pipeline_key, signature
        )

        if speech_path.exists() and background_path.exists() and use_cache and cache_valid:
            self.log(
                "Cache HIT (separação): "
                f"{speech_path.name} ({_file_size_mb(speech_path)}), "
                f"{background_path.name} ({_file_size_mb(background_path)})"
            )
        else:
            if use_cache and (speech_path.exists() or background_path.exists()) and not cache_valid:
                self.log(f"Cache MISS (separação): {cache_reason}. Regerando artefatos.")

            self.log(
                "Executando separação/realce para transcrição "
                f"(highpass={Config.SEPARATION_HIGHPASS_HZ}Hz, "
                f"lowpass={Config.SEPARATION_LOWPASS_HZ}Hz)..."
            )
            audio = AudioSegment.from_file(original_audio)
            speech = audio.set_channels(1).high_pass_filter(
                Config.SEPARATION_HIGHPASS_HZ
            ).low_pass_filter(Config.SEPARATION_LOWPASS_HZ)
            speech = normalize(speech)
            speech.export(speech_path, format="wav")
            audio.export(background_path, format="wav")

            _write_cache_meta(self.base_dir, "separation", pipeline_key, signature)

        context["transcription_audio_path"] = str(speech_path)
        context["background_audio_path"] = str(background_path)
        self.log(
            f"Output separação: speech={speech_path} ({_file_size_mb(speech_path)}), "
            f"background={background_path} ({_file_size_mb(background_path)})"
        )
        return context


class TranscriptionPhase(PipelinePhase):
    """Fase 3: Transcreve áudio (Faster-Whisper ou OpenAI-Whisper)."""

    def execute(self, context: Dict) -> Dict:
        transcription_input = Path(
            context.get("transcription_audio_path", context["original_audio_path"])
        )
        segments_path = self.base_dir / "segments_transcribed.json"
        use_cache = bool(context.get("use_cache"))
        pipeline_key = context.get("pipeline_cache_key", "")
        signature = {
            "backend": self.backend,
            "device": self.device,
            "model": Config.WHISPER_MODEL,
            "lang": Config.WHISPER_LANG,
            "beam": Config.WHISPER_BEAM,
            "compute": Config.WHISPER_COMPUTE,
        }

        cache_valid, cache_reason = _is_cache_valid(
            self.base_dir, "transcription", pipeline_key, signature
        )
        if segments_path.exists() and use_cache and cache_valid:
            self.log(f"Cache HIT (transcrição): {segments_path.name}")
            with open(segments_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
            if _ensure_segment_ids(segments):
                with open(segments_path, "w", encoding="utf-8") as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
            context["segments"] = segments
            self.log(f"Segmentos carregados do cache: {len(segments)}")
            return context

        if use_cache and segments_path.exists() and not cache_valid:
            self.log(f"Cache MISS (transcrição): {cache_reason}. Reprocessando.")

        self.log(
            f"Input transcrição: {transcription_input} ({_file_size_mb(transcription_input)})"
        )
        segments: List[Dict] = []

        if self.backend == "faster-whisper":
            self.log(f"Carregando Faster-Whisper em {self.device}...")
            from faster_whisper import WhisperModel

            compute_type = Config.WHISPER_COMPUTE if self.device == "cuda" else "int8"
            model = WhisperModel(
                Config.WHISPER_MODEL,
                device=self.device,
                compute_type=compute_type,
            )

            self.log("Iniciando transcrição (Faster)...")
            segments_gen, _ = model.transcribe(
                str(transcription_input),
                language=Config.WHISPER_LANG,
                beam_size=Config.WHISPER_BEAM,
            )
            for idx, seg in enumerate(segments_gen):
                segments.append(
                    {
                        "seg_id": idx,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                    }
                )
            del model
        else:
            self.log(f"Carregando OpenAI-Whisper (Legacy) em {self.device}...")
            import whisper

            model = whisper.load_model(Config.WHISPER_MODEL, device=self.device)
            self.log("Iniciando transcrição (Standard)...")

            result = model.transcribe(
                str(transcription_input),
                language=Config.WHISPER_LANG,
                beam_size=Config.WHISPER_BEAM,
                fp16=False,
            )
            for idx, seg in enumerate(result.get("segments", [])):
                segments.append(
                    {
                        "seg_id": idx,
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                    }
                )
            del model

        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        _write_cache_meta(self.base_dir, "transcription", pipeline_key, signature)

        context["segments"] = segments
        self.log(
            f"Transcrição concluída: {len(segments)} segmentos -> {segments_path} "
            f"({_file_size_mb(segments_path)})"
        )
        return context


class PostProcessingPhase(PipelinePhase):
    """Fase 4: Limpeza e correção dos segmentos transcritos."""

    def execute(self, context: Dict) -> Dict:
        segments = context.get("segments", [])
        if not segments:
            self.log("Nenhum segmento para processar.")
            return context

        removed = 0
        cleaned_segments = []
        for seg in segments:
            original_text = seg["text"]
            new_text = re.sub(r"\[.*?\]|\(.*?\)", "", original_text).strip()
            if len(new_text) < 2:
                removed += 1
                continue
            seg["text"] = new_text
            cleaned_segments.append(seg)

        _ensure_segment_ids(cleaned_segments)
        processed_path = self.base_dir / "segments_cleaned.json"
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_segments, f, ensure_ascii=False, indent=2)

        context["segments"] = cleaned_segments
        self.log(
            f"Pós-processamento: entrada={len(segments)}, removidos={removed}, "
            f"saída={len(cleaned_segments)} -> {processed_path} ({_file_size_mb(processed_path)})"
        )
        return context


class TranslationPhase(PipelinePhase):
    """Fase 5: Traduz texto usando MarianMT (Transformers)."""

    def execute(self, context: Dict) -> Dict:
        segments = context.get("segments", [])
        if not segments:
            self.log("Nenhum segmento para tradução.")
            return context

        translated_path = self.base_dir / "segments_translated.json"
        use_cache = bool(context.get("use_cache"))
        pipeline_key = context.get("pipeline_cache_key", "")
        signature = {
            "model": Config.TRANS_MODEL,
            "batch": Config.TRANS_BATCH,
            "device": self.device,
        }

        cache_valid, cache_reason = _is_cache_valid(
            self.base_dir, "translation", pipeline_key, signature
        )
        if translated_path.exists() and use_cache and cache_valid:
            self.log(f"Cache HIT (tradução): {translated_path.name}")
            with open(translated_path, "r", encoding="utf-8") as f:
                cached_segments = json.load(f)

            if any("text_pt" not in seg for seg in cached_segments):
                self.log("Cache inválido de tradução: campo 'text_pt' ausente. Reprocessando.")
            else:
                _ensure_segment_ids(cached_segments)
                context["segments"] = cached_segments
                self.log(f"Segmentos traduzidos carregados do cache: {len(cached_segments)}")
                return context

        if use_cache and translated_path.exists() and not cache_valid:
            self.log(f"Cache MISS (tradução): {cache_reason}. Reprocessando.")

        self.log(f"Traduzindo {len(segments)} segmentos com {Config.TRANS_MODEL} em {self.device}...")
        tokenizer = MarianTokenizer.from_pretrained(Config.TRANS_MODEL)
        model = MarianMTModel.from_pretrained(Config.TRANS_MODEL).to(self.device)

        texts = [s["text"] for s in segments]
        translated_texts: List[str] = []

        batch_size = max(1, Config.TRANS_BATCH)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        self.log(
            f"Tradução em lotes: batch_size={batch_size}, total_batches={total_batches}"
        )
        for i in range(0, len(texts), batch_size):
            batch_idx = (i // batch_size) + 1
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                translated_tokens = model.generate(**encoded)
            translated_texts.extend(
                tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            )
            if batch_idx == 1 or batch_idx == total_batches:
                self.log(f"Lote de tradução {batch_idx}/{total_batches} concluído.")

        for i, txt in enumerate(translated_texts):
            segments[i]["text_pt"] = txt

        with open(translated_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        _write_cache_meta(self.base_dir, "translation", pipeline_key, signature)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        context["segments"] = segments
        self.log(
            f"Tradução concluída: {len(segments)} segmentos -> {translated_path} "
            f"({_file_size_mb(translated_path)})"
        )
        return context


class TTSPhase(PipelinePhase):
    """Fase 6: Gera áudios em português usando Edge-TTS."""

    def execute(self, context: Dict) -> Dict:
        segments = context.get("segments", [])
        if not segments:
            self.log("Nenhum segmento para TTS.")
            return context

        _ensure_segment_ids(segments)
        chunks_dir = self.base_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        use_cache = bool(context.get("use_cache"))
        pipeline_key = context.get("pipeline_cache_key", "")
        signature = {
            "voice": Config.TTS_VOICE,
            "workers": Config.TTS_WORKERS,
        }
        cache_valid, cache_reason = _is_cache_valid(
            self.base_dir, "tts", pipeline_key, signature
        )
        if use_cache and not cache_valid:
            self.log(f"Cache MISS (tts): {cache_reason}. Regenerando chunks necessários.")

        tasks = []
        cache_hits = 0
        missing_ids = []
        for seg in segments:
            seg_id = seg["seg_id"]
            if "text_pt" not in seg:
                raise RuntimeError(
                    f"Segmento {seg_id} sem 'text_pt'. A tradução não foi concluída."
                )

            path = chunks_dir / f"seg_{seg_id:04d}.wav"
            duration = seg["end"] - seg["start"]
            if use_cache and cache_valid and path.exists():
                cache_hits += 1
                continue

            if not path.exists():
                missing_ids.append(seg_id)
            tasks.append((seg_id, seg["text_pt"], str(path), duration))

        self.log(
            f"TTS: total={len(segments)}, cache_hits={cache_hits}, synth_needed={len(tasks)}"
        )
        if missing_ids:
            preview = ", ".join(str(seg_id) for seg_id in missing_ids[:10])
            suffix = "..." if len(missing_ids) > 10 else ""
            self.log(f"TTS chunks ausentes antes da síntese: {preview}{suffix}")

        if not tasks:
            self.log("Todos os áudios já existem em cache válido.")
            return context

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=Config.TTS_WORKERS
        ) as executor:
            failures = [err for err in executor.map(self._generate_single, tasks) if err]

        if failures:
            raise RuntimeError(
                f"Falha ao gerar {len(failures)} áudio(s) no TTS. Primeiro erro: {failures[0]}"
            )

        _write_cache_meta(self.base_dir, "tts", pipeline_key, signature)
        self.log(f"TTS concluído: chunks em {chunks_dir}")
        return context

    def _generate_single(self, args) -> Optional[str]:
        seg_id, text, path, target_dur = args
        try:
            asyncio.run(self._synthesize(text, path, target_dur))
            seg = AudioSegment.from_file(path)
            seg = normalize(seg).high_pass_filter(80)
            seg.export(path, format="wav")
            del seg
            return None
        except Exception as e:
            message = f"Erro TTS no seg_id={seg_id} ({Path(path).name}): {e}"
            self.log(message)
            return message

    async def _synthesize(self, text: str, path: str, target_dur: float) -> None:
        voice = Config.TTS_VOICE
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)

        seg = AudioSegment.from_file(path)
        actual_dur = len(seg) / 1000.0

        if target_dur <= 0:
            return

        if actual_dur > target_dur:
            ratio = (actual_dur / target_dur) - 1
            percentage = min(int(ratio * 100) + 5, 50)
            if percentage > 5:
                rate_str = f"+{percentage}%"
                communicate = edge_tts.Communicate(text, voice, rate=rate_str)
                await communicate.save(path)


class AudioMixingPhase(PipelinePhase):
    """Fase 7: Mixagem com sidechain ducking (FFmpeg)."""

    def _get_duration(self, audio_path: Path) -> float:
        """Obtém a duração do áudio com fallback para ffprobe."""
        try:
            return AudioSegment.from_file(audio_path).duration_seconds
        except Exception as audio_err:
            self.log(f"Falha ao obter duração via pydub: {audio_err}. Tentando ffprobe...")

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode == 0:
            output = (process.stdout or "").strip()
            try:
                return float(output)
            except ValueError:
                pass

        ffprobe_err = (process.stderr or "").strip()
        raise RuntimeError(
            f"Não foi possível obter duração de '{audio_path}'. Erro ffprobe: {ffprobe_err}"
        )

    def execute(self, context: Dict) -> Dict:
        output_mix = self.base_dir / "final_mix.wav"
        chunks_dir = self.base_dir / "chunks"
        speech_track_path = self.base_dir / "speech_track.wav"
        use_cache = bool(context.get("use_cache"))
        pipeline_key = context.get("pipeline_cache_key", "")
        signature = {
            "duck_threshold": Config.DUCK_THRESH,
            "duck_ratio": Config.DUCK_RATIO,
            "duck_attack": Config.DUCK_ATTACK,
            "duck_release": Config.DUCK_RELEASE,
        }

        cache_valid, cache_reason = _is_cache_valid(
            self.base_dir, "mixing", pipeline_key, signature
        )
        if output_mix.exists() and use_cache and cache_valid:
            self.log(f"Cache HIT (mixagem): {output_mix.name} ({_file_size_mb(output_mix)})")
            context["final_audio_path"] = str(output_mix)
            return context

        if use_cache and output_mix.exists() and not cache_valid:
            self.log(f"Cache MISS (mixagem): {cache_reason}. Reprocessando.")

        background_path = Path(
            context.get("background_audio_path", context["original_audio_path"])
        )
        segments = context.get("segments", [])
        if not segments:
            raise RuntimeError("Sem segmentos para mixagem.")

        sorted_segments = sorted(segments, key=lambda x: x["start"])
        duration = context.get("video_duration")
        if not duration:
            duration = self._get_duration(background_path)
            context["video_duration"] = duration

        self.log(
            f"Montando faixa de voz sincronizada ({duration:.2f}s). "
            f"Input base={background_path}"
        )

        timeline_parts = []
        cursor_ms = 0
        used_chunks = 0
        missing_chunks: List[int] = []

        for i, seg in enumerate(sorted_segments):
            seg_id = seg.get("seg_id", i)
            chunk_path = chunks_dir / f"seg_{seg_id:04d}.wav"
            if not chunk_path.exists():
                missing_chunks.append(seg_id)
                continue

            start_ms = int(seg["start"] * 1000)
            gap = start_ms - cursor_ms
            if gap > 0:
                timeline_parts.append(
                    AudioSegment.silent(duration=gap, frame_rate=44100).set_channels(1)
                )
                cursor_ms += gap

            voice_chunk = AudioSegment.from_file(chunk_path).set_channels(1)
            if i < len(sorted_segments) - 1:
                next_start_ms = int(sorted_segments[i + 1]["start"] * 1000)
                time_until_next = next_start_ms - cursor_ms
                if len(voice_chunk) > time_until_next and time_until_next > 100:
                    ratio = len(voice_chunk) / time_until_next
                    voice_chunk = speedup(
                        voice_chunk, playback_speed=max(1.0, ratio * 1.05)
                    )
                    if len(voice_chunk) > time_until_next:
                        voice_chunk = voice_chunk[:time_until_next]

            timeline_parts.append(voice_chunk)
            cursor_ms += len(voice_chunk)
            used_chunks += 1

        if missing_chunks:
            preview = ", ".join(str(seg_id) for seg_id in missing_chunks[:12])
            suffix = "..." if len(missing_chunks) > 12 else ""
            self.log(
                f"Chunks ausentes na mixagem: {len(missing_chunks)} ({preview}{suffix})"
            )

        if used_chunks == 0:
            raise RuntimeError("Nenhum chunk de voz disponível para montar a mixagem.")

        total_dur_ms = int(duration * 1000)
        if total_dur_ms > cursor_ms:
            timeline_parts.append(
                AudioSegment.silent(
                    duration=total_dur_ms - cursor_ms, frame_rate=44100
                ).set_channels(1)
            )

        speech_track = sum(timeline_parts, AudioSegment.empty())
        speech_track.export(str(speech_track_path), format="wav")
        self.log(
            f"Faixa de voz montada: {speech_track_path} ({_file_size_mb(speech_track_path)}), "
            f"chunks_usados={used_chunks}/{len(sorted_segments)}"
        )

        thresh = Config.DUCK_THRESH
        ratio = Config.DUCK_RATIO
        attack = Config.DUCK_ATTACK
        release = Config.DUCK_RELEASE

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(background_path),
            "-i",
            str(speech_track_path),
            "-filter_complex",
            f"[1:a]asplit=2[sc][voice];"
            f"[0:a][sc]sidechaincompress=threshold={thresh}:ratio={ratio}:attack={attack}:release={release}[bg_ducked];"
            f"[bg_ducked][voice]amix=inputs=2:duration=first:dropout_transition=0[out]",
            "-map",
            "[out]",
            "-ac",
            "2",
            str(output_mix),
        ]

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            self.log(f"FFmpeg mixagem falhou (rc={process.returncode}).")
            self.log(f"Comando: {' '.join(cmd)}")
            stderr = (process.stderr or "").strip()
            if stderr:
                self.log(stderr[:3000])
            raise RuntimeError("Falha na mixagem com ducking.")

        _write_cache_meta(self.base_dir, "mixing", pipeline_key, signature)
        context["final_audio_path"] = str(output_mix)
        self.log(f"Output mixagem: {output_mix} ({_file_size_mb(output_mix)})")
        return context


class RenderingPhase(PipelinePhase):
    """Fase 8: Combina vídeo original com novo áudio usando FFmpeg."""

    def execute(self, context: Dict) -> Dict:
        video_path = Path(context["video_path"])
        audio_path = Path(context["final_audio_path"])
        output_video = self.base_dir / f"{video_path.stem}{Config.OUTPUT_SUFFIX}.mp4"

        self.log(f"Renderizando vídeo final: vídeo={video_path}, áudio={audio_path}")

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

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            self.log(f"FFmpeg render falhou (rc={process.returncode}).")
            self.log(f"Comando: {' '.join(cmd)}")
            stderr = (process.stderr or "").strip()
            if stderr:
                self.log(stderr[:3000])
            raise RuntimeError("Falha na renderização FFmpeg.")

        context["output_video_path"] = str(output_video)
        self.log(f"Output render: {output_video} ({_file_size_mb(output_video)})")
        return context
