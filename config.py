import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path("/opt/whisper-service")
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", 5))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", 5))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", 0.0))

ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "true").lower() == "true"
MAX_DURATION_SHORT_SEC = int(os.getenv("MAX_DURATION_SHORT_SEC", 600))
MAX_CHUNK_DURATION = int(os.getenv("MAX_CHUNK_DURATION", 600))
OVERLAP_SECONDS = int(os.getenv("OVERLAP_SECONDS", 15))

VAD_ENABLED = os.getenv("VAD_ENABLED", "true").lower() == "true"
VAD_MAX_CHUNK_SEC = int(os.getenv("VAD_MAX_CHUNK_SEC", 600))
VAD_MIN_SPEECH_SEC = float(os.getenv("VAD_MIN_SPEECH_SEC", 1.0))
VAD_MIN_SILENCE_SEC = float(os.getenv("VAD_MIN_SILENCE_SEC", 0.5))

MIN_WORD_CONFIDENCE = float(os.getenv("MIN_WORD_CONFIDENCE", 0.5))
MAX_REPEAT_WINDOW_SEC = float(os.getenv("MAX_REPEAT_WINDOW_SEC", 60))
REPEAT_SIMILARITY_THRESHOLD = float(os.getenv("REPEAT_SIMILARITY_THRESHOLD", 0.9))
INTRA_REPEAT_NGRAM = int(os.getenv("INTRA_REPEAT_NGRAM", 3))
INTRA_REPEAT_MAX_RATIO = float(os.getenv("INTRA_REPEAT_MAX_RATIO", 0.7))

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 3000))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 2))

HF_TOKEN = os.getenv("HF_TOKEN")
DIARIZE_BY_DEFAULT = os.getenv("DIARIZE_BY_DEFAULT", "false").lower() == "true"

CLEANUP_HOURS = int(os.getenv("CLEANUP_HOURS", 24))
