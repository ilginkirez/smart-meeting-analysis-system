"""
🔵 Emotion Agent — LangGraph node.

Wraps EmotionFusion from src/emotion/fusion.py.
If emotion analysis is unavailable (e.g., missing audio features),
the node writes an empty dict and never crashes the pipeline.

State inputs:
    transcript   (str | dict) : text-only path — used if audio not available
    audio_path   (str)        : optional path to audio file for acoustic emotion

State outputs:
    emotions     (dict)       : { per_speaker: {speaker: emotion_label}, overall: str }
                                or {} if unavailable
"""

from __future__ import annotations


# ──────────────────────────────────────────────
# LangGraph Node
# ──────────────────────────────────────────────
def emotion_node(state: dict) -> dict:
    """
    LangGraph node — runs emotion fusion; safe no-op if unavailable.

    Attempts to import and run EmotionFusion from src.emotion.fusion.
    On any ImportError or runtime error, writes {} to state["emotions"]
    so the rest of the pipeline is never blocked.
    """
    try:
        from src.emotion.fusion import EmotionFusion  # type: ignore

        fusion = EmotionFusion()
        transcript = state.get("transcript", "")
        audio_path = state.get("audio_path")

        emotions = fusion.analyze(
            transcript=transcript,
            audio_path=audio_path,
        )
        return {**state, "emotions": emotions or {}}

    except (ImportError, NotImplementedError):
        # EmotionFusion not yet implemented — graceful passthrough
        return {**state, "emotions": {}}
    except Exception as exc:
        print(f"⚠️  EmotionAgent error (non-fatal): {exc}")
        return {**state, "emotions": {}}


# ──────────────────────────────────────────────
# Legacy class shim
# ──────────────────────────────────────────────
class EmotionAgent:
    """Thin wrapper around emotion_node for non-LangGraph usage."""

    def __call__(self, state: dict) -> dict:
        return emotion_node(state)
