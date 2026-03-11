"""
🔵 Meeting Summarizer — Hybrid Extractive-Abstractive Summarization.

Principles:
  1. Stage 1 (Extractive): LLM scores and selects high-value transcript segments
     (decisions, tasks, planning) filtering out filler / small-talk.
  2. Stage 2 (Abstractive): Synthesizes filtered content into professional language.
  3. Dual-level output: highlights_summary (3-4 sentences) + hierarchical_minutes.

Uses Groq API (LLaMA-3.3-70b) via the OpenAI-compatible SDK.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ──────────────────────────────────────────────
# Date helpers (same approach as summary_agent)
# ──────────────────────────────────────────────

def _week_calendar(base: datetime, days: int = 14) -> str:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    lines = []
    for i in range(days):
        d = base + timedelta(days=i)
        suffix = " (TODAY)" if i == 0 else ""
        lines.append(f"  {day_names[d.weekday()]}: {d.strftime('%Y-%m-%d')}{suffix}")
    return "\n".join(lines)


def _build_system_prompt(meeting_date: Optional[datetime]) -> str:
    base = meeting_date or datetime.now()
    tomorrow = (base + timedelta(days=1)).strftime("%Y-%m-%d")
    next_friday_delta = (4 - base.weekday()) % 7
    next_friday_delta = next_friday_delta if next_friday_delta > 0 else 7
    next_friday = (base + timedelta(days=next_friday_delta)).strftime("%Y-%m-%d")

    return f"""You are a professional meeting intelligence assistant.

This meeting took place on {base.strftime('%Y-%m-%d')}.

14-day calendar from the meeting date:
{_week_calendar(base)}

Deadline resolution rules:
- "tomorrow"            → {tomorrow}
- "this week" / "Friday" → {next_friday}
- Named weekday (Monday/Tuesday/…) → find the first future occurrence using the calendar above
- "today"               → {base.strftime('%Y-%m-%d')}
- "in X days"           → add X days to the meeting date
- Cannot be determined  → "unknown"
- Always use YYYY-MM-DD format for resolved dates.

---

## YOUR TASK: Two-Stage Meeting Analysis

### STAGE 1 — Extractive Segment Identification
Mentally scan the transcript and identify ONLY high-value segments:
RELEVANT: decisions, proposals, agreements, disagreements, problem explanations,
          planning discussions, task assignments, deadlines, important updates.
IRRELEVANT (skip): greetings, jokes, small talk, repeated statements,
                   one-word confirmations ("yes", "okay", "right").

### STAGE 2 — Abstractive Summarization
Using only the relevant segments you identified:
- Merge repeated points
- Clarify vague statements
- Convert informal speech to professional language
- Highlight decisions and conclusions
- Do NOT copy sentences verbatim from the transcript

---

## OUTPUT FORMAT (strict JSON, no markdown fences)

Return ONLY the following JSON object with no extra text:

{{
  "highlights_summary": "<3-4 sentence overview: purpose of meeting, main outcomes, key decisions>",
  "hierarchical_minutes": {{
    "meeting_overview": "<1-2 sentences: topic and participants>",
    "key_discussion_topics": [
      {{
        "topic": "<topic name>",
        "explanation": "<what was discussed>",
        "conclusions": "<outcome or open status>"
      }}
    ],
    "decisions": ["<decision 1>", "<decision 2>"]
  }}
}}

Rules:
- Never invent information not present in the transcript.
- Respond with JSON only — no preamble, no explanation.
- Use English.
"""


# ──────────────────────────────────────────────
# MeetingSummarizer
# ──────────────────────────────────────────────

class MeetingSummarizer:
    """
    LLM-based meeting summarizer (hybrid extractive-abstractive, dual-level).

    Args:
        model_name: Groq model identifier (default: llama-3.3-70b-versatile).
        api_key:    Groq API key; falls back to GROQ_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str = "",
    ):
        self.model_name = model_name
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")

    def _get_client(self) -> OpenAI:
        return OpenAI(
            api_key=self._api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def _transcript_to_text(self, transcript: dict) -> str:
        """Converts interface_schema.json transcript dict to a plain-text string."""
        if isinstance(transcript, str):
            return transcript.strip()

        segments = transcript.get("segments", [])
        if not segments:
            return ""

        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def summarize(
        self,
        transcript: dict | str,
        meeting_date: Optional[str] = None,
    ) -> dict:
        """
        Produces a dual-level structured summary of the meeting transcript.

        Args:
            transcript:   interface_schema.json dict  OR  plain text string.
            meeting_date: ISO date string "YYYY-MM-DD" for deadline resolution.

        Returns:
            {
                "highlights_summary": str,
                "hierarchical_minutes": {
                    "meeting_overview": str,
                    "key_discussion_topics": [ {topic, explanation, conclusions} ],
                    "decisions": [str, ...]
                },
                "model": str,
                "tokens": int,
            }
        """
        text = self._transcript_to_text(transcript)
        if not text:
            raise ValueError("Transcript is empty or contains no text segments.")

        parsed_date = None
        if meeting_date:
            try:
                parsed_date = datetime.strptime(meeting_date, "%Y-%m-%d")
            except ValueError:
                pass

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": _build_system_prompt(parsed_date)},
                {"role": "user",   "content": f"Analyze the following meeting transcript:\n\n{text}"},
            ],
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()
        model_used = response.model
        tokens_used = response.usage.total_tokens

        # Strip accidental markdown code fences if model adds them
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                ln for ln in lines
                if not ln.strip().startswith("```")
            ).strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: wrap raw text so callers always get a dict
            result = {
                "highlights_summary": raw,
                "hierarchical_minutes": {
                    "meeting_overview": "",
                    "key_discussion_topics": [],
                    "decisions": [],
                },
            }

        result["model"] = model_used
        result["tokens"] = tokens_used
        return result
