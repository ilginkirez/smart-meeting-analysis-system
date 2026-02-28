"""
🔴 Kişi A — Ses pipeline testleri.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = PROJECT_ROOT / "interface_schema.json"
MOCK_PATH = PROJECT_ROOT / "data" / "sample" / "mock_transcript.json"


def test_mock_transcript_matches_schema():
    """Mock transkript'in interface_schema.json formatına uyduğunu doğrular."""
    with open(MOCK_PATH, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Zorunlu alanlar
    assert "meeting_id" in transcript
    assert "duration_seconds" in transcript
    assert "segments" in transcript
    assert isinstance(transcript["segments"], list)

    # Segment yapısı
    for seg in transcript["segments"]:
        assert "speaker" in seg
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert seg["end"] > seg["start"], "end zamanı start'tan büyük olmalı"


def test_audio_preprocessor_placeholder():
    """Preprocessor sınıfının import edildiğini doğrular."""
    from src.audio.preprocessor import AudioPreprocessor
    preprocessor = AudioPreprocessor()
    assert preprocessor.target_sample_rate == 16000


def test_transcriber_placeholder():
    """Transcriber sınıfının import edildiğini doğrular."""
    from src.audio.transcriber import WhisperTranscriber
    transcriber = WhisperTranscriber(model_size="base")
    assert transcriber.model_size == "base"


def test_diarizer_placeholder():
    """Diarizer sınıfının import edildiğini doğrular."""
    from src.audio.diarizer import SpeakerDiarizer
    diarizer = SpeakerDiarizer()
    assert diarizer.pipeline is None
