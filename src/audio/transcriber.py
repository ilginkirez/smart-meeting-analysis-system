"""
🔴 Kişi A — Whisper tabanlı konuşma tanıma (ASR) modülü.
Ses dosyasından metin transkripsiyon üretir.

Sorumluluğu:
- OpenAI Whisper ile metne çevirme
- Kelime ve segment düzeyinde zaman damgalı çıktı
- Çoklu dil desteği (auto-detect veya manuel)
"""

# TODO: Kişi A bu dosyayı geliştirecek
# Kullanılacak kütüphaneler: openai-whisper, torch


class WhisperTranscriber:
    """OpenAI Whisper ile ses → metin dönüşümü."""

    def __init__(self, model_size: str = "base", language: str | None = None):
        self.model_size = model_size
        self.language = language
        self.model = None  # lazy loading

    def load_model(self):
        """Whisper modelini yükler."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")

    def transcribe(self, audio_path: str) -> dict:
        """
        Ses dosyasını metne çevirir.

        Returns:
            {
                "text": "full transcript...",
                "segments": [
                    {"start": 0.0, "end": 4.2, "text": "...", "confidence": 0.95},
                    ...
                ],
                "language": "en"
            }
        """
        raise NotImplementedError("Kişi A tarafından implement edilecek")
