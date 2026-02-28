"""
🔵 Kişi B — Toplantı özetleme modülü.
Transkriptten anlamlı ve kısa bir özet üretir.

Sorumluluğu:
- Extractive-abstractive hibrit özetleme
- Uzun toplantıları chunk'lara bölüp map-reduce ile özetleme
- LangChain + GPT-4 entegrasyonu
"""

# TODO: Kişi B bu dosyayı geliştirecek
# Kullanılacak kütüphaneler: langchain, langchain-openai


class MeetingSummarizer:
    """LLM tabanlı toplantı özetleyici."""

    def __init__(self, model_name: str = "gpt-4", api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key

    def summarize(self, transcript: dict) -> dict:
        """
        Transkriptten özet üretir.

        Args:
            transcript: interface_schema.json formatında transkript

        Returns:
            {
                "summary": "Toplantı özeti...",
                "key_topics": ["konu1", "konu2"],
                "decisions": ["karar1", "karar2"]
            }
        """
        raise NotImplementedError("Kişi B tarafından implement edilecek")
