import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio

from config import (
    VAD_ENABLED,
    VAD_MAX_CHUNK_SEC,
    VAD_MIN_SPEECH_SEC,
    VAD_MIN_SILENCE_SEC,
    MAX_CHUNK_DURATION,
    OVERLAP_SECONDS,
)

logger = logging.getLogger(__name__)


def convert_to_wav(input_path: Path) -> Path:
    """Конвертация любого аудио/видео в WAV 16kHz mono"""
    if input_path.suffix.lower() == ".wav":
        output_path = input_path.with_name(f"{input_path.stem}_16k.wav")
    else:
        output_path = input_path.with_suffix(".wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    logger.info(f"FFmpeg cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"Конвертация {input_path.name} -> {output_path.name}")
    return output_path


def get_duration_seconds(path: Path) -> float:
    """Длительность через ffprobe"""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        logger.error(f"Не удалось получить длительность для {path}")
        return 0.0


def time_chunking_fallback(
    audio_path: Path, max_duration: int = MAX_CHUNK_DURATION, overlap: int = OVERLAP_SECONDS
) -> List[Tuple[Path, dict]]:
    """Нарезка по времени с overlap"""
    duration = get_duration_seconds(audio_path)
    chunks: List[Tuple[Path, dict]] = []
    if duration <= 0:
        return chunks

    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + max_duration, duration)
        chunk_path = audio_path.parent / f"{audio_path.stem}_chunk_{idx:03d}.wav"
        idx += 1

        cmd = [
            "ffmpeg",
            "-ss",
            str(start),
            "-i",
            str(audio_path),
            "-t",
            str(end - start),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(chunk_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        chunks.append((chunk_path, {"start": start, "end": end}))
        logger.info(f"Создан fallback-чанк {chunk_path.name} [{start:.1f} - {end:.1f}]")

        if end >= duration:
            break
        start = end - overlap

    return chunks


def vad_chunking(
    audio_path: Path, max_chunk_sec: int = VAD_MAX_CHUNK_SEC, overlap_sec: int = OVERLAP_SECONDS
) -> List[Tuple[Path, dict]]:
    """
    Нарезка по VAD (Silero):
    - берём интервалы речи
    - собираем из них чанки примерно по max_chunk_sec
    - между чанками добавляем overlap_sec
    """
    duration = get_duration_seconds(audio_path)
    chunks: List[Tuple[Path, dict]] = []
    if duration <= 0:
        return chunks

    # читаем аудио (16k mono wav)
    wav, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        logger.warning(f"Ожидался sample_rate=16000, а пришёл {sr}, VAD может работать хуже")
    wav = wav.squeeze(0)

    logger.info("Загрузка Silero VAD для chunking...")
    torch.set_num_threads(1)
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
    (get_speech_timestamps, _, _, _, _) = utils

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        return_seconds=True,
    )

    # фильтр коротких кусочков речи
    filtered_stamps = []
    for ts in speech_timestamps:
        s = float(ts["start"])
        e = float(ts["end"])
        if e - s >= VAD_MIN_SPEECH_SEC:
            filtered_stamps.append({"start": s, "end": e})

    speech_timestamps = filtered_stamps

    if not speech_timestamps:
        logger.warning("VAD не нашёл речи, fallback на time-based chunking")
        return time_chunking_fallback(audio_path, max_chunk_sec, overlap_sec)

    current_chunk_start = None
    current_chunk_end = None
    idx = 0

    def _flush_chunk(start_sec: float, end_sec: float, idx: int):
        chunk_path = audio_path.parent / f"{audio_path.stem}_chunk_{idx:03d}.wav"
        start_with_overlap = max(start_sec - overlap_sec, 0.0)
        cmd = [
            "ffmpeg",
            "-ss",
            str(start_with_overlap),
            "-i",
            str(audio_path),
            "-t",
            str(end_sec - start_with_overlap),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(chunk_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        chunks.append((chunk_path, {"start": start_with_overlap, "end": end_sec}))
        logger.info(
            f"Создан VAD-чанк {chunk_path.name} [{start_with_overlap:.1f} - {end_sec:.1f}]"
        )

    for ts in speech_timestamps:
        s = float(ts["start"])
        e = float(ts["end"])

        if current_chunk_start is None:
            current_chunk_start = s
            current_chunk_end = e
            continue

        if e - current_chunk_start <= max_chunk_sec:
            current_chunk_end = e
        else:
            _flush_chunk(current_chunk_start, current_chunk_end, idx)
            idx += 1
            current_chunk_start = s
            current_chunk_end = e

    if current_chunk_start is not None:
        _flush_chunk(current_chunk_start, current_chunk_end, idx)

    return chunks


def choose_chunks(audio_path: Path) -> List[Tuple[Path, dict]]:
    """Выбор стратегии чанкования"""
    if not VAD_ENABLED:
        return time_chunking_fallback(audio_path)
    try:
        return vad_chunking(audio_path)
    except Exception as e:
        logger.error(f"VAD chunking failed, fallback to time-based: {e}")
        return time_chunking_fallback(audio_path)
