"""
🔴 Kişi A — Metin tabanlı duygu sınıflandırma.
Fine-tuned RoBERTa ile segment düzeyinde duygu analizi.
"""

# TODO: Kişi A bu dosyayı geliştirecek (Faz 2 — duygu analizi aşamasında)


class TextEmotionClassifier:
    """RoBERTa tabanlı duygu sınıflandırıcı."""

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name

    def classify(self, text: str) -> dict:
        """Metindeki duyguyu sınıflandırır."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")

    def classify_segments(self, segments: list[dict]) -> list[dict]:
        """Her segment için duygu etiketi ekler."""
        raise NotImplementedError("Kişi A tarafından implement edilecek")
