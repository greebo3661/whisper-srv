import subprocess
import asyncio
from pathlib import Path
from typing import List, Tuple
import torch
import torchaudio
from silero_vad.utils import load_silero_vad, read_audio, VADIterator

async def get_duration(path: Path) -> float:
    """Длительность через ffprobe"""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
           "-of", "csv=p=0", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def simple_vad_chunking(audio_path: Path, max_duration: int, overlap: int) -> List[Tuple[Path, dict]]:
    """Простая нарезка по времени с overlap (VAD позже)"""
    duration = asyncio.run(get_duration(audio_path))
    chunks = []
    
    start = 0
    while start < duration:
        end = min(start + max_duration, duration)
        chunk_path = audio_path.parent / f"{audio_path.stem}_chunk_{len(chunks):03d}.wav"
        
        cmd = [
            "ffmpeg", "-ss", str(start), "-i", str(audio_path),
            "-t", str(end - start), "-ar", "16000", "-ac", "1", "-y",
            str(chunk_path)
        ]
        subprocess.run(cmd, check=True)
        
        chunks.append((chunk_path, {"start": start, "end": end}))
        start = end - overlap
    
    return chunks

# В проде замени на silero-vad full
