"""
🔵 Summary Agent — LangGraph node.

Runs the MeetingSummarizer (hybrid extractive-abstractive, dual-level) and
writes structured JSON results into the pipeline state.

State inputs:
    transcript   (str | dict) : required — raw transcript text or schema dict
    meeting_date (str)        : optional "YYYY-MM-DD" for deadline resolution
    meeting_id   (str)        : optional, used for output file naming

State outputs:
    summary        (dict) : { highlights_summary, hierarchical_minutes }
    summary_model  (str)
    summary_tokens (int)
    summary_error  (str)  : only present if an error occurred
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TypedDict, Optional

from src.nlp.summarizer import MeetingSummarizer


# ──────────────────────────────────────────────
# Shared Pipeline State
# ──────────────────────────────────────────────
class MeetingState(TypedDict, total=False):
    meeting_id:     str
    meeting_date:   str           # "YYYY-MM-DD"
    participants:   list[str]
    transcript:     str | dict    # raw text or interface_schema dict
    summary:        dict          # structured { highlights_summary, hierarchical_minutes }
    summary_model:  str
    summary_tokens: int
    summary_error:  str
    action_items:   list[dict]    # [ {task, owner, deadline} ]
    emotions:       dict


# ──────────────────────────────────────────────
# Output persistence
# ──────────────────────────────────────────────
def _save_summary(meeting_id: str, data: dict) -> str:
    output_dir = os.path.join(os.getcwd(), "outputs", "summary")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{meeting_id}_{timestamp}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Summary saved: outputs/summary/{meeting_id}_{timestamp}.json")
    return filepath


# ──────────────────────────────────────────────
# LangGraph Node
# ──────────────────────────────────────────────
def summary_node(state: MeetingState) -> MeetingState:
    """
    LangGraph node — produces a dual-level structured summary.

    Reads:  state["transcript"], state["meeting_date"], state["meeting_id"]
    Writes: state["summary"], state["summary_model"], state["summary_tokens"]
    """
    transcript = state.get("transcript", "")
    if not transcript:
        return {**state, "summary_error": "No transcript found in state."}

    summarizer = MeetingSummarizer()
    try:
        result = summarizer.summarize(
            transcript=transcript,
            meeting_date=state.get("meeting_date"),
        )
    except Exception as exc:
        return {**state, "summary_error": str(exc)}

    meeting_id = state.get("meeting_id", "meeting")
    _save_summary(meeting_id, result)

    return {
        **state,
        "summary":        result,
        "summary_model":  result.get("model", ""),
        "summary_tokens": result.get("tokens", 0),
    }
