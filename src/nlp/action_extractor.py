"""
🔵 Kişi B — Aksiyon maddesi çıkarım modülü.
Transkriptten görevleri, sorumlularını ve teslim tarihlerini çıkarır.

Sorumluluğu:
- "Bu raporu ben cuma hazırlayayım" → görev + kişi + tarih
- spaCy NER ile kişi ve tarih tespiti
- GPT-4 ile görev yapılandırma
"""

# TODO: Kişi B bu dosyayı geliştirecek
# Kullanılacak kütüphaneler: spacy, langchain, langchain-openai


class ActionExtractor:
    """Toplantıdan aksiyon maddelerini çıkarır."""

    def __init__(self, model_name: str = "gpt-4", api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key

    def extract(self, transcript: dict) -> list[dict]:
        """
        Transkriptten aksiyon maddelerini çıkarır.

        Args:
            transcript: interface_schema.json formatında transkript

        Returns:
            [
                {
                    "task": "Raporu hazırla",
                    "assignee": "Ahmet",
                    "deadline": "2025-03-07",
                    "priority": "high",
                    "source_segment": {"start": 45.2, "end": 48.5}
                },
                ...
            ]
        """
        raise NotImplementedError("Kişi B tarafından implement edilecek")
