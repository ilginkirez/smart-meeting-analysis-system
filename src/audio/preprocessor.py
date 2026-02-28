"""
🔴 Kişi A — Ses ön işleme modülü.
Ses dosyasını Whisper ve Pyannote için uygun formata dönüştürür.

Sorumluluğu:
- Herhangi bir ses formatını 16kHz mono WAV'a dönüştürme
- Gürültü azaltma (opsiyonel)
- Sessizlik kırpma
"""

# TODO: Kişi A bu dosyayı geliştirecek
# Kullanılacak kütüphaneler: pydub, librosa, noisereduce, webrtcvad


class AudioPreprocessor:
    """Ses dosyasını pipeline için hazırlar."""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def convert_to_wav(self, input_path: str, output_path: str) -> str:
        """Herhangi bir ses formatını 16kHz mono WAV'a dönüştürür."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")

    def reduce_noise(self, audio_path: str) -> str:
        """Gürültü azaltma uygular."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")

    def trim_silence(self, audio_path: str, threshold_db: float = -40.0) -> str:
        """Baş ve sondaki sessizliği kırpar."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")
