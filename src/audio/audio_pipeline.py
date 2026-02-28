"""
🔴 Kişi A — Uçtan uca ses pipeline.
Preprocessor → Whisper → Pyannote çıktılarını birleştirir.

Bu modül interface_schema.json formatında çıktı üretir.
Bu format Kişi B'nin NLP pipeline'ının girdisidir — DEĞİŞTİRMEYİN.
"""

# TODO: Kişi A bu dosyayı geliştirecek


class AudioPipeline:
    """Ses dosyasından yapılandırılmış transkript üretir."""

    def __init__(self, whisper_model_size: str = "base", hf_token: str = ""):
        self.whisper_model_size = whisper_model_size
        self.hf_token = hf_token

    def process(self, audio_path: str) -> dict:
        """
        Tam ses işleme pipeline'ı çalıştırır.

        Args:
            audio_path: Girdi ses dosyasının yolu

        Returns:
            interface_schema.json formatında dict:
            {
                "meeting_id": "...",
                "duration_seconds": 120.5,
                "num_speakers": 3,
                "segments": [...]
            }
        """
        raise NotImplementedError("Kişi A tarafından implement edilecek")

    def _align_transcription_with_diarization(
        self, transcription: dict, diarization: list[dict]
    ) -> list[dict]:
        """Whisper transkriptini diyarizasyon segmentleriyle eşleştirir."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")
