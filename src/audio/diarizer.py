"""
🔴 Kişi A — Konuşmacı diyarizasyon modülü.
Kim ne zaman konuştu sorusunu yanıtlar.

Sorumluluğu:
- Pyannote.audio ile konuşmacı segmentasyonu
- ECAPA-TDNN embedding ile konuşmacı tanıma
- Üst üste konuşma (overlap) tespiti
- VAD kalibrasyonu
"""

# TODO: Kişi A bu dosyayı geliştirecek
# Kullanılacak kütüphaneler: pyannote.audio, torch


class SpeakerDiarizer:
    """Pyannote tabanlı konuşmacı diyarizasyon."""

    def __init__(self, hf_token: str = ""):
        self.hf_token = hf_token
        self.pipeline = None  # lazy loading

    def load_pipeline(self):
        """Pyannote diyarizasyon pipeline'ını yükler."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")

    def diarize(self, audio_path: str) -> list[dict]:
        """
        Konuşmacı segmentlerini döner.

        Returns:
            [
                {"speaker": "Speaker_0", "start": 0.0, "end": 4.2, "overlap": False},
                {"speaker": "Speaker_1", "start": 4.5, "end": 8.1, "overlap": False},
                ...
            ]
        """
        raise NotImplementedError("Kişi A tarafından implement edilecek")

    def detect_overlaps(self, audio_path: str) -> list[dict]:
        """Üst üste konuşma bölgelerini tespit eder."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")
