# whisper-srv

Сервис транскрибации на базе faster-whisper:

- поддержка длинных записей (VAD-чанкование, overlap)
- антигаллюцинатор (фильтры по prob, n-граммам, повторам)
- диаризация (опционально, через pyannote.audio)
- экспорт TXT и JSON, таймкоды по сегментам

## Быстрый старт

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
python whisper_server.py
Сервис поднимется на http://0.0.0.0:5001.

Основные эндпоинты
POST /api/v1/transcribe — загрузка файла и транскрибация

GET /api/v1/dl/{filename} — скачать TXT/JSON результат

GET /health — статус сервиса
