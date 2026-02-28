"""
🔵 Kişi B — RAG (Retrieval-Augmented Generation) pipeline.
Toplantı öncesi paylaşılan dokümanları bağlam olarak kullanır.

Sorumluluğu:
- Dokümanları embedding'e çevirme
- FAISS vektör deposunda saklama
- Özetleme sırasında ilgili bağlamı getirme
"""

# TODO: Kişi B bu dosyayı geliştirecek
# Kullanılacak kütüphaneler: langchain, faiss-cpu


class RAGPipeline:
    """Doküman tabanlı bağlam zenginleştirme."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.vector_store = None

    def ingest_documents(self, document_paths: list[str]) -> None:
        """Dokümanları vektör deposuna ekler."""
        raise NotImplementedError("Kişi B tarafından implement edilecek")

    def retrieve_context(self, query: str, top_k: int = 3) -> list[str]:
        """Sorguya en uygun doküman parçalarını getirir."""
        raise NotImplementedError("Kişi B tarafından implement edilecek")
