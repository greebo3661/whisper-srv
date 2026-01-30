import uuid
import time
import logging
import json
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from asyncio import Semaphore
from logging.handlers import RotatingFileHandler

from faster_whisper import WhisperModel

from config import (
    BASE_DIR,
    UPLOAD_DIR,
    RESULTS_DIR,
    WHISPER_MODEL_NAME,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_TEMPERATURE,
    ENABLE_CHUNKING,
    MAX_DURATION_SHORT_SEC,
    MAX_FILE_SIZE_MB,
    MAX_CONCURRENT,
    HF_TOKEN,
    DIARIZE_BY_DEFAULT,
    CLEANUP_HOURS,
)
from audio_utils import convert_to_wav, get_duration_seconds, choose_chunks
from anti_hallucination import apply_anti_hallucination

# ============ LOGGING ============

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            str(BASE_DIR / "whisper.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ============ GLOBALS ============

app = FastAPI(title="Whisper Transcription Service")

whisper_model: Optional[WhisperModel] = None
diarization_pipeline = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

transcription_semaphore = Semaphore(MAX_CONCURRENT)

# ============ UTILS ============


def get_whisper_model() -> WhisperModel:
    global whisper_model
    if whisper_model is None:
        logger.info(f"Загрузка Whisper {WHISPER_MODEL_NAME} на {DEVICE} ({COMPUTE_TYPE})...")
        whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=str(BASE_DIR / "models"),
        )
        logger.info("Whisper модель загружена")
    return whisper_model


def get_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is None:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN не найден в окружении. Диаризация недоступна.")
        logger.info("Загрузка pyannote.audio для диаризации...")
        from pyannote.audio import Pipeline

        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        if DEVICE == "cuda":
            diarization_pipeline.to(torch.device("cuda"))
        logger.info("Диаризация загружена")
    return diarization_pipeline


def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None,
    word_timestamps: bool = False,
    vad_filter: bool = True,
):
    model = get_whisper_model()

    logger.info(f"Транскрибация {audio_path.name} (lang={language}, vad={vad_filter})")
    start = time.time()

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        beam_size=WHISPER_BEAM_SIZE,
        best_of=WHISPER_BEST_OF,
        temperature=WHISPER_TEMPERATURE,
    )

    result_segments = []
    full_text_parts = []

    for seg in segments:
        seg_dict = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        }
        if word_timestamps and seg.words:
            seg_dict["words"] = [
                {
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "word": w.word,
                    "probability": round(w.probability, 3),
                }
                for w in seg.words
            ]
        result_segments.append(seg_dict)
        full_text_parts.append(seg.text.strip())

    elapsed = time.time() - start
    logger.info(f"Транскрибация {audio_path.name} завершена за {elapsed:.1f}s")

    return {
        "segments": result_segments,
        "text": " ".join(full_text_parts),
        "language": info.language,
        "duration": info.duration,
        "transcription_time": round(elapsed, 2),
    }


def apply_diarization(audio_path: Path, segments: list) -> list:
    pipeline = get_diarization_pipeline()

    logger.info(f"Диаризация {audio_path.name}...")
    start = time.time()

    diarization = pipeline(str(audio_path))

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        speakers = {}

        for turn, _, spk in diarization.itertracks(yield_label=True):
            overlap = min(turn.end, seg_end) - max(turn.start, seg_start)
            if overlap > 0:
                speakers[spk] = speakers.get(spk, 0.0) + overlap

        seg["speaker"] = max(speakers, key=speakers.get) if speakers else "UNKNOWN"

    elapsed = time.time() - start
    logger.info(f"Диаризация завершена за {elapsed:.1f}s")

    return segments


def merge_chunk_results(chunk_results: List[Tuple[dict, dict]]) -> dict:
    all_segments = []
    full_text_parts = []
    language = None
    total_duration = 0.0
    total_time = 0.0

    for meta, res in chunk_results:
        offset = meta["start"]
        for seg in res["segments"]:
            seg = seg.copy()
            seg["start"] = round(seg["start"] + offset, 2)
            seg["end"] = round(seg["end"] + offset, 2)
            all_segments.append(seg)
            full_text_parts.append(seg["text"])
        language = language or res.get("language")
        total_duration = max(total_duration, meta["end"])
        total_time += res.get("transcription_time", 0.0)

    return {
        "segments": all_segments,
        "text": " ".join(full_text_parts),
        "language": language,
        "duration": total_duration,
        "transcription_time": round(total_time, 2),
    }


def cleanup_old_files():
    import time as _time

    cutoff = _time.time() - CLEANUP_HOURS * 3600
    for directory in (UPLOAD_DIR, RESULTS_DIR):
        for f in directory.iterdir():
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    logger.info(f"Удален старый файл: {f}")
            except Exception as e:
                logger.warning(f"Не удалось удалить {f}: {e}")


def format_timestamp(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    total = int(sec)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ============ SMART TRANSCRIBE ============


async def smart_transcribe(
    audio_path: Path,
    language: Optional[str],
    diarize: bool,
    word_timestamps: bool,
    vad_filter: bool,
) -> dict:
    duration = get_duration_seconds(audio_path)
    logger.info(f"Файл {audio_path.name} длительностью ~{duration:.1f}s")

    async with transcription_semaphore:
        if (not ENABLE_CHUNKING) or duration <= 0 or duration < MAX_DURATION_SHORT_SEC:
            result = transcribe_audio(
                audio_path,
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
            )
            if diarize:
                try:
                    result["segments"] = apply_diarization(
                        audio_path, result["segments"]
                    )
                except Exception as e:
                    logger.error(f"Ошибка диаризации: {e}")
                    result["diarization_error"] = str(e)

            segs = apply_anti_hallucination(result.get("segments", []))
            result["segments"] = segs
            result["text"] = " ".join(s.get("text", "").strip() for s in segs)
            return result

        # длинный — режем
        chunks = choose_chunks(audio_path)
        if not chunks:
            result = transcribe_audio(
                audio_path,
                language=language,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
            )
            segs = apply_anti_hallucination(result.get("segments", []))
            result["segments"] = segs
            result["text"] = " ".join(s.get("text", "").strip() for s in segs)
            return result

        chunk_results: List[Tuple[dict, dict]] = []

        for chunk_path, meta in chunks:
            try:
                res = transcribe_audio(
                    chunk_path,
                    language=language,
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                )
                if diarize:
                    try:
                        res["segments"] = apply_diarization(chunk_path, res["segments"])
                    except Exception as e:
                        logger.error(
                            f"Ошибка диаризации чанка {chunk_path.name}: {e}"
                        )
                        res["diarization_error"] = str(e)
                chunk_results.append((meta, res))
            finally:
                try:
                    if chunk_path.exists():
                        chunk_path.unlink()
                except Exception as e:
                    logger.warning(f"Не удалось удалить чанк {chunk_path}: {e}")

        merged = merge_chunk_results(chunk_results)
        segs = apply_anti_hallucination(merged.get("segments", []))
        merged["segments"] = segs
        merged["text"] = " ".join(s.get("text", "").strip() for s in segs)
        return merged


# ============ API ============


@app.on_event("startup")
async def on_startup():
    cleanup_old_files()


@app.post("/api/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(False),
    word_timestamps: bool = Form(False),
    vad_filter: bool = Form(True),
):
    task_id = str(uuid.uuid4())
    logger.info(f"[{task_id}] Новая задача: {file.filename}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Файл слишком большой")

    file_ext = Path(file.filename).suffix.lower()
    upload_path = UPLOAD_DIR / f"{task_id}{file_ext}"

    try:
        with open(upload_path, "wb") as f:
            f.write(content)

        audio_path = convert_to_wav(upload_path)

        if audio_path != upload_path and upload_path.exists():
            upload_path.unlink()

        result = await smart_transcribe(
            audio_path=audio_path,
            language=language,
            diarize=diarize or DIARIZE_BY_DEFAULT,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
        )

        base_name = f"{task_id}_{Path(file.filename).stem}"
        txt_file = RESULTS_DIR / f"{base_name}.txt"
        json_file = RESULTS_DIR / f"{base_name}.json"

        segments = result.get("segments", []) or []
        formatted_lines = []
        for seg in segments:
            start = format_timestamp(seg.get("start", 0.0))
            end = format_timestamp(seg.get("end", seg.get("start", 0.0)))
            speaker = seg.get("speaker")
            text = seg.get("text", "").strip()
            if not text:
                continue
            if speaker:
                formatted_lines.append(f"[{start} --> {end}] [{speaker}] {text}")
            else:
                formatted_lines.append(f"[{start} --> {end}] {text}")

        if formatted_lines:
            txt_content = "\n".join(formatted_lines)
        else:
            txt_content = result.get("text", "")

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(txt_content)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        try:
            if upload_path.exists():
                upload_path.unlink()
        except Exception as e:
            logger.warning(f"Не удалось удалить upload {upload_path}: {e}")

        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception as e:
            logger.warning(f"Не удалось удалить временный аудио файл {audio_path}: {e}")

        logger.info(f"[{task_id}] Завершено успешно")

        return JSONResponse(
            {
                "task_id": task_id,
                "filename": file.filename,
                "text": result.get("text", ""),
                "language": result.get("language"),
                "duration": result.get("duration"),
                "transcription_time": result.get("transcription_time"),
                "segments_count": len(result.get("segments", [])),
                "files": {
                    "txt": f"/api/v1/dl/{txt_file.name}",
                    "json": f"/api/v1/dl/{json_file.name}",
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{task_id}] Ошибка: {e}")
        try:
            if upload_path.exists():
                upload_path.unlink()
        except Exception:
            pass
        try:
            if "audio_path" in locals() and audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dl/{filename}")
async def download_file(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Недопустимое имя файла")
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(file_path, filename=filename)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": whisper_model is not None,
        "diarization_available": HF_TOKEN is not None,
        "chunking_enabled": ENABLE_CHUNKING,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
