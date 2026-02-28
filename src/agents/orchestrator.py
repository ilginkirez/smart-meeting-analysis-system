"""
🔵 Kişi B — LangGraph orkestratör.
Tüm ajanları koordine eden ana kontrol noktası.

Akış: audio_input → transcription → diarization → [summarize, extract_actions, analyze_emotion] → merge
"""

# TODO: Kişi B bu dosyayı geliştirecek
# Kullanılacak kütüphaneler: langgraph


class MeetingOrchestrator:
    """LangGraph tabanlı çok ajanlı orkestratör."""

    def __init__(self):
        self.graph = None

    def build_graph(self):
        """Ajan akış grafını oluşturur."""
        raise NotImplementedError("Kişi B tarafından implement edilecek")

    def run(self, transcript: dict) -> dict:
        """
        Tüm ajanları çalıştırır ve sonuçları birleştirir.

        Returns:
            {
                "meeting_id": "...",
                "transcript": {...},
                "summary": {...},
                "action_items": [...],
                "emotions": {...}
            }
        """
        raise NotImplementedError("Kişi B tarafından implement edilecek")
