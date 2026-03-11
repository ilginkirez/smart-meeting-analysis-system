"""
🔵 Action Agent — LangGraph node.

Wraps ActionExtractor and writes extracted tasks into the pipeline state.

State inputs:
    transcript   (str | dict) : required
    meeting_date (str)        : optional "YYYY-MM-DD"
    meeting_id   (str)        : optional, for output file naming

State outputs:
    action_items  (list[dict]) : [ {task, owner, deadline} ]
    action_error  (str)        : only if an error occurred
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from src.nlp.action_extractor import ActionExtractor


# ──────────────────────────────────────────────
# Output persistence
# ──────────────────────────────────────────────
def _save_actions(meeting_id: str, items: list[dict]) -> str:
    output_dir = os.path.join(os.getcwd(), "outputs", "actions")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{meeting_id}_{timestamp}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"💾 Actions saved: outputs/actions/{meeting_id}_{timestamp}.json")
    return filepath


# ──────────────────────────────────────────────
# LangGraph Node
# ──────────────────────────────────────────────
def action_node(state: dict) -> dict:
    """
    LangGraph node — extracts action items from the transcript.

    Reads:  state["transcript"], state["meeting_date"], state["meeting_id"]
    Writes: state["action_items"]
    """
    transcript = state.get("transcript", "")
    if not transcript:
        return {**state, "action_items": [], "action_error": "No transcript in state."}

    extractor = ActionExtractor()
    try:
        items = extractor.extract(
            transcript=transcript,
            meeting_date=state.get("meeting_date"),
        )
    except Exception as exc:
        return {**state, "action_items": [], "action_error": str(exc)}

    meeting_id = state.get("meeting_id", "meeting")
    _save_actions(meeting_id, items)

    return {**state, "action_items": items}


# ──────────────────────────────────────────────
# Legacy class shim (kept for backward-compat)
# ──────────────────────────────────────────────
class ActionAgent:
    """Thin wrapper around action_node for non-LangGraph usage."""

    def __call__(self, state: dict) -> dict:
        return action_node(state)
