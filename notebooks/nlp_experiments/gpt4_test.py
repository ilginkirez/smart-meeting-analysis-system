"""
GPT-4 Test — Groq API üzerinden toplantı analizi
Toplantı verilerini sample_meetings.json'dan okur.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Groq API — OpenAI-uyumlu endpoint
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# JSON'dan toplantı verilerini yükle
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "sample_meetings.json")

with open(json_path, "r", encoding="utf-8") as f:
    meetings = json.load(f)

# Her toplantıyı analiz et
for meeting in meetings:
    print("=" * 60)
    print(f"TOPLANTI: {meeting['title']}")
    print(f"Tarih: {meeting['date']} | Katılımcılar: {', '.join(meeting['participants'])}")
    print("=" * 60)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen bir toplantı analiz asistanısın. "
                    "Verilen transkriptten şunları çıkar:\n"
                    "1. Toplantı özeti (2-3 cümle)\n"
                    "2. Görevler (kime atandı, deadline)\n"
                    "3. Karar noktaları\n"
                    "Çıktıyı Türkçe ve düzenli formatta ver."
                ),
            },
            {
                "role": "user",
                "content": f"Bu toplantıyı analiz et:\n\n{meeting['transcript']}",
            },
        ],
    )

    print(response.choices[0].message.content)
    print(f"\n[Model: {response.model} | Tokens: {response.usage.total_tokens}]")
    print()
