"""
🔵 Meeting Orchestrator — LangGraph multi-agent pipeline.

Graph structure:
  START → summary_node → action_node → emotion_node → merge_node → END

The merge_node assembles the final structured output matching the
JSON schema defined in the user's meeting intelligence spec:

  {
    "highlights_summary":    str,
    "hierarchical_minutes":  { meeting_overview, key_discussion_topics, decisions },
    "action_items":          [ { task, owner, deadline } ],
    "emotions":              dict
  }
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from src.agents.summary_agent import MeetingState, summary_node
from src.agents.action_agent import action_node
from src.agents.emotion_agent import emotion_node
from src.utils import generate_meeting_id


# ──────────────────────────────────────────────
# Merge node (final assembly)
# ──────────────────────────────────────────────
def merge_node(state: dict) -> dict:
    """
    Assembles all agent outputs into the canonical output schema.
    Also persists the full result to outputs/merged/.
    """
    summary: dict = state.get("summary", {})
    action_items: list = state.get("action_items", [])
    emotions: dict = state.get("emotions", {})

    output = {
        "meeting_id":           state.get("meeting_id", "unknown"),
        "meeting_date":         state.get("meeting_date", ""),
        "highlights_summary":   summary.get("highlights_summary", ""),
        "hierarchical_minutes": {
            "meeting_overview":        summary.get("hierarchical_minutes", {}).get("meeting_overview", ""),
            "key_discussion_topics":   summary.get("hierarchical_minutes", {}).get("key_discussion_topics", []),
            "decisions":               summary.get("hierarchical_minutes", {}).get("decisions", []),
        },
        "action_items":  action_items,
        "emotions":      emotions,
        "meta": {
            "summary_model":  state.get("summary_model", ""),
            "summary_tokens": state.get("summary_tokens", 0),
            "processed_at":   datetime.now().isoformat(),
        },
    }

    # Persist to outputs/merged/
    output_dir = os.path.join(os.getcwd(), "outputs", "merged")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meeting_id = state.get("meeting_id", "meeting")
    filepath = os.path.join(output_dir, f"{meeting_id}_{timestamp}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"💾 Full output saved: outputs/merged/{meeting_id}_{timestamp}.json")

    return {**state, "final_output": output}


# ──────────────────────────────────────────────
# MeetingOrchestrator
# ──────────────────────────────────────────────
class MeetingOrchestrator:
    """
    LangGraph-based multi-agent orchestrator.

    Builds and runs the pipeline:
      summary_node → action_node → emotion_node → merge_node

    Falls back to sequential execution if langgraph is not installed.
    """

    def __init__(self):
        self.graph = None
        self._use_langgraph = False
        self._try_build_graph()

    def _try_build_graph(self):
        """Attempt to build a LangGraph StateGraph; fall back to plain Python."""
        try:
            from langgraph.graph import StateGraph, END  # type: ignore

            builder = StateGraph(MeetingState)
            builder.add_node("summary",  summary_node)
            builder.add_node("actions",  action_node)
            builder.add_node("emotions", emotion_node)
            builder.add_node("merge",    merge_node)

            builder.set_entry_point("summary")
            builder.add_edge("summary",  "actions")
            builder.add_edge("actions",  "emotions")
            builder.add_edge("emotions", "merge")
            builder.add_edge("merge",    END)

            self.graph = builder.compile()
            self._use_langgraph = True
            print("✅ LangGraph graph compiled successfully.")
        except ImportError:
            print("⚠️  langgraph not installed — using sequential fallback.")

    def build_graph(self):
        """Public method to (re)build the LangGraph graph."""
        self._try_build_graph()

    def run(self, state: dict) -> dict:
        """
        Execute the full pipeline.

        Args:
            state: dict containing at minimum {"transcript": ...}.
                   Optional keys: "meeting_id", "meeting_date", "participants".

        Returns:
            The canonical output dict (see merge_node).
        """
        # Ensure meeting_id exists
        if "meeting_id" not in state or not state["meeting_id"]:
            state = {**state, "meeting_id": generate_meeting_id()}

        if self._use_langgraph and self.graph is not None:
            final_state = self.graph.invoke(state)
        else:
            # Sequential fallback
            final_state = summary_node(state)
            final_state = action_node(final_state)
            final_state = emotion_node(final_state)
            final_state = merge_node(final_state)

        return final_state.get("final_output", {})
