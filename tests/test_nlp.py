"""
🔵 Kişi B — NLP pipeline testleri.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MOCK_PATH = PROJECT_ROOT / "data" / "sample" / "mock_transcript.json"


def test_mock_transcript_loadable():
    """Mock transkript dosyasının yüklendiğini doğrular."""
    with open(MOCK_PATH, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    assert len(transcript["segments"]) > 0


def test_summarizer_placeholder():
    """Summarizer sınıfının import edildiğini doğrular."""
    from src.nlp.summarizer import MeetingSummarizer
    summarizer = MeetingSummarizer()
    assert summarizer.model_name == "gpt-4"


def test_action_extractor_placeholder():
    """ActionExtractor sınıfının import edildiğini doğrular."""
    from src.nlp.action_extractor import ActionExtractor
    extractor = ActionExtractor()
    assert extractor.model_name == "gpt-4"


def test_rag_pipeline_placeholder():
    """RAGPipeline sınıfının import edildiğini doğrular."""
    from src.nlp.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    assert rag.vector_store is None
