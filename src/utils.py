"""
Ortak yardımcı fonksiyonlar.
İki kişi de bu dosyayı kullanır — büyük değişiklikler yapmadan önce diğerine haber verin.
"""
import json
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Standart logger oluşturur."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_json(path: str | Path) -> dict:
    """JSON dosyasını okur."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """JSON dosyasına yazar."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def format_timestamp(seconds: float) -> str:
    """Saniyeyi HH:MM:SS formatına çevirir."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def generate_meeting_id() -> str:
    """Benzersiz toplantı ID'si üretir."""
    return f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
