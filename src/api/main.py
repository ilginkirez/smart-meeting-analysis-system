"""
🔵 FastAPI Application — Meeting Intelligence API.

Endpoints:
  GET  /health    — health check
  POST /analyze   — run the full NLP pipeline on a transcript

Request body for /analyze follows interface_schema.json OR a plain text variant.
Response returns the canonical structured output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.orchestrator import MeetingOrchestrator

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Meeting Analysis API",
    description=(
        "Transforms raw meeting transcripts into structured meeting intelligence: "
        "dual-level summaries, action items, and emotion signals."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────────────────────────────────────

class TranscriptSegment(BaseModel):
    speaker:    str
    start:      float
    end:        float
    text:       str
    overlap:    bool    = False
    confidence: Optional[float] = None


class TranscriptRequest(BaseModel):
    """
    Request body for POST /analyze.
    Accepts either structured segments or a plain text transcript.
    """
    meeting_id:       Optional[str]                  = Field(default=None, description="Unique meeting identifier")
    meeting_date:     Optional[str]                  = Field(default=None, description="Meeting date YYYY-MM-DD (for deadline resolution)")
    participants:     Optional[list[str]]            = Field(default=None)
    # Structured format (interface_schema.json)
    duration_seconds: Optional[float]                = None
    language:         Optional[str]                  = "en"
    num_speakers:     Optional[int]                  = None
    segments:         Optional[list[TranscriptSegment]] = None
    # Plain text alternative
    transcript_text:  Optional[str]                  = Field(default=None, description="Plain text transcript (alternative to segments)")


class ActionItem(BaseModel):
    task:     str
    owner:    str
    deadline: str


class HierarchicalMinutes(BaseModel):
    meeting_overview:       str
    key_discussion_topics:  list[dict[str, Any]]
    decisions:              list[str]


class AnalysisResponse(BaseModel):
    meeting_id:           str
    meeting_date:         str
    highlights_summary:   str
    hierarchical_minutes: HierarchicalMinutes
    action_items:         list[ActionItem]
    emotions:             dict[str, Any]
    meta:                 dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator (singleton, initialised at startup)
# ──────────────────────────────────────────────────────────────────────────────
orchestrator = MeetingOrchestrator()


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/analyze", response_model=AnalysisResponse, summary="Analyze a meeting transcript")
def analyze(request: TranscriptRequest):
    """
    Run the full meeting intelligence pipeline on a transcript.

    Accepts:
    - Structured segments (interface_schema.json format)
    - OR plain text via `transcript_text`

    Returns a structured analysis with:
    - `highlights_summary` — 3–4 sentence quick overview
    - `hierarchical_minutes` — detailed structured summary
    - `action_items` — concrete tasks with owner + deadline
    - `emotions` — emotion data (if available)
    """
    # Build transcript payload for the pipeline
    if request.segments:
        transcript = {
            "meeting_id":       request.meeting_id or "",
            "duration_seconds": request.duration_seconds or 0,
            "language":         request.language or "en",
            "num_speakers":     request.num_speakers,
            "segments": [seg.model_dump() for seg in request.segments],
        }
    elif request.transcript_text:
        transcript = request.transcript_text.strip()
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'segments' (structured) or 'transcript_text' (plain text).",
        )

    state = {
        "meeting_id":   request.meeting_id or "",
        "meeting_date": request.meeting_date or "",
        "participants": request.participants or [],
        "transcript":   transcript,
    }

    try:
        result = orchestrator.run(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # Validate minutes sub-object
    minutes = result.get("hierarchical_minutes", {})
    return AnalysisResponse(
        meeting_id=result.get("meeting_id", ""),
        meeting_date=result.get("meeting_date", ""),
        highlights_summary=result.get("highlights_summary", ""),
        hierarchical_minutes=HierarchicalMinutes(
            meeting_overview=minutes.get("meeting_overview", ""),
            key_discussion_topics=minutes.get("key_discussion_topics", []),
            decisions=minutes.get("decisions", []),
        ),
        action_items=[ActionItem(**item) for item in result.get("action_items", [])],
        emotions=result.get("emotions", {}),
        meta=result.get("meta", {}),
    )
