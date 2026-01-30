#!/usr/bin/env python3
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from config import config

async def cleanup_old():
    cutoff = datetime.now() - timedelta(hours=config.CLEANUP_HOURS)
    for dir_path in [config.UPLOAD_DIR, config.RESULTS_DIR]:
        for f in dir_path.iterdir():
            if f.stat().st_mtime < cutoff.timestamp():
                f.unlink()

if __name__ == "__main__":
    asyncio.run(cleanup_old())
