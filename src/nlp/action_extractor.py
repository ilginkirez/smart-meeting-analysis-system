"""
🔵 Action Extractor — Structured action item extraction from meeting transcripts.

For each actionable commitment found in the transcript, produces:
  - task:     what needs to be done (action verb + object)
  - owner:    who is responsible
  - deadline: resolved ISO date (YYYY-MM-DD) or "unknown"

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
# Date helpers
# ──────────────────────────────────────────────

def _week_calendar(base: datetime, days: int = 14) -> str:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    lines = []
    for i in range(days):
        d = base + timedelta(days=i)
        suffix = " (TODAY)" if i == 0 else ""
        lines.append(f"  {day_names[d.weekday()]}: {d.strftime('%Y-%m-%d')}{suffix}")
    return "\n".join(lines)


def _build_extraction_prompt(meeting_date: Optional[datetime]) -> str:
    base = meeting_date or datetime.now()
    tomorrow = (base + timedelta(days=1)).strftime("%Y-%m-%d")
    next_friday_delta = (4 - base.weekday()) % 7
    next_friday_delta = next_friday_delta if next_friday_delta > 0 else 7
    next_friday = (base + timedelta(days=next_friday_delta)).strftime("%Y-%m-%d")

    return f"""You are a structured action-item extractor for meeting transcripts.

This meeting took place on {base.strftime('%Y-%m-%d')}.

14-day calendar from the meeting date (use for deadline resolution):
{_week_calendar(base)}

Deadline resolution rules:
- "tomorrow"            → {tomorrow}
- "this week" / "Friday" → {next_friday}
- Named weekday         → first future occurrence using the calendar above
- "today"               → {base.strftime('%Y-%m-%d')}
- "in X days"           → meeting date + X days
- Cannot be determined  → "unknown"
- Always output dates as YYYY-MM-DD.

---

## YOUR TASK

Read the transcript and extract every concrete action item.

An action item must contain:
  - A clear action verb (prepare, create, send, review, schedule, fix, etc.)
  - A responsible person or team
  - Sometimes a deadline

SKIP: vague intentions, general plans with no owner, and past actions already done.

---

## OUTPUT FORMAT (strict JSON array, no markdown fences)

Return ONLY a JSON array. Each element:
{{
  "task":     "<concise description — action verb + object>",
  "owner":    "<person or team name, or 'unknown'>",
  "deadline": "<YYYY-MM-DD or 'unknown'>"
}}

If no action items are found, return an empty array: []

Rules:
- Base every item on actual transcript content — do not invent tasks.
- If the owner is unclear, set owner to "unknown".
- If the deadline is unclear or not mentioned, set deadline to "unknown".
- Respond with JSON only — no preamble, no explanation.
"""


# ──────────────────────────────────────────────
# ActionExtractor
# ──────────────────────────────────────────────

class ActionExtractor:
    """
    Extracts structured action items from a meeting transcript using an LLM.

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

    def _transcript_to_text(self, transcript: dict | str) -> str:
        if isinstance(transcript, str):
            return transcript.strip()
        segments = transcript.get("segments", [])
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def extract(
        self,
        transcript: dict | str,
        meeting_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Extracts action items from the transcript.

        Args:
            transcript:   interface_schema.json dict  OR  plain text string.
            meeting_date: ISO date string "YYYY-MM-DD" for deadline resolution.

        Returns:
            List of dicts: [{"task": str, "owner": str, "deadline": str}, ...]
            Returns [] if no action items found, raises ValueError on empty input.
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
                {"role": "system", "content": _build_extraction_prompt(parsed_date)},
                {"role": "user",   "content": f"Extract action items from this meeting:\n\n{text}"},
            ],
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                ln for ln in lines
                if not ln.strip().startswith("```")
            ).strip()

        try:
            result = json.loads(raw)
            if isinstance(result, list):
                # Normalise: ensure every item has the required keys
                normalised = []
                for item in result:
                    normalised.append({
                        "task":     str(item.get("task", "")).strip(),
                        "owner":    str(item.get("owner", "unknown")).strip() or "unknown",
                        "deadline": str(item.get("deadline", "unknown")).strip() or "unknown",
                    })
                return normalised
        except json.JSONDecodeError:
            pass

        # Fallback: return empty list rather than crashing the pipeline
        return []
