"""🔵 Kişi B — Summary Agent (LangGraph Node).

Groq API üzerinden toplantı transkriptini analiz eder.
Pipeline state'inden transcript/meeting_date alır, summary yazar.
Çıktılar outputs/summary/ klasörüne otomatik kaydedilir.
"""

from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import TypedDict, Optional
import os

load_dotenv()

# ──────────────────────────────────────────────
# State Tanımı
# ──────────────────────────────────────────────
class MeetingState(TypedDict, total=False):
    meeting_id:     str           # Örn: "meeting_004"
    meeting_date:   str           # "YYYY-MM-DD"
    participants:   list[str]
    transcript:     str           # Ham transkript
    summary:        str           # Bu node'un çıktısı
    summary_model:  str
    summary_tokens: int
    summary_error:  str


# ──────────────────────────────────────────────
# Tarih Yardımcısı
# ──────────────────────────────────────────────
TR_DAYS = {
    "Pazartesi": 0, "Salı": 1, "Çarşamba": 2, "Perşembe": 3,
    "Cuma": 4, "Cumartesi": 5, "Pazar": 6,
}

def _week_calendar(base: datetime) -> str:
    lines = []
    for i in range(14):
        d = base + timedelta(days=i)
        tr_name = list(TR_DAYS.keys())[d.weekday()]
        suffix = " (BUGÜN)" if i == 0 else ""
        lines.append(f"  {tr_name}: {d.strftime('%Y-%m-%d')}{suffix}")
    return "\n".join(lines)


def _build_system_prompt(meeting_date: Optional[datetime]) -> str:
    base = meeting_date or datetime.now()
    tomorrow = (base + timedelta(days=1)).strftime("%Y-%m-%d")
    next_friday = (base + timedelta(days=(4 - base.weekday()) % 7)).strftime("%Y-%m-%d")

    return f"""Sen profesyonel bir toplantı analiz asistanısın.

Bu toplantı {base.strftime('%Y-%m-%d')} tarihinde gerçekleşti.

Toplantı tarihinden itibaren 14 günlük takvim:
{_week_calendar(base)}

Deadline çözümleme kuralları:
- "Yarın" veya "yarına kadar"   → {tomorrow}
- "Bu hafta"                    → {next_friday} (Cuma)
- "Pazartesi", "Salı" vb.       → takvimden ilk gelecek o günün tarihi (YYYY-MM-DD)
- "Bugün"                       → {base.strftime('%Y-%m-%d')}
- "X gün sonra"                 → toplantı tarihine X gün ekle
- Çıkarılamazsa                 → "Belirtilmedi"
- Tarih formatı: YYYY-MM-DD

---

Verilen transkripti analiz et ve şu bilgileri çıkar:

1) TOPLANTI ÖZETİ — en fazla 3 cümle
2) GÖREVLER — her görev için: Görev / Sorumlu / Deadline / Açıklama
3) KARARLAR — net alınan kararlar
4) AÇIK KONULAR — karara bağlanmamış / takip gereken konular

Kurallar:
- Bilgi uydurma, sadece transkriptten çıkarım yap.
- Deadline yoksa → "Belirtilmedi", sorumlu yoksa → "Atanmadı".
- Türkçe yaz.

FORMAT:

📌 Toplantı Özeti
...

📋 Görevler
1.
Görev:
Sorumlu:
Deadline:
Açıklama:

📊 Kararlar
- ...

❗ Açık Konular
- ...
"""


# ──────────────────────────────────────────────
# Output Kaydetme
# ──────────────────────────────────────────────
def _save_output(meeting_id: str, content: str, model: str, tokens: int) -> str:
    output_dir = os.path.join(os.getcwd(), "outputs", "summary")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{meeting_id}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Toplantı ID : {meeting_id}\n")
        f.write(f"Tarih       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model       : {model}\n")
        f.write(f"Token       : {tokens}\n")
        f.write("=" * 60 + "\n\n")
        f.write(content)

    print(f"💾 Çıktı kaydedildi: outputs/summary/{filename}")
    return filepath


# ──────────────────────────────────────────────
# LangGraph Node
# ──────────────────────────────────────────────
def summary_node(state: MeetingState) -> MeetingState:
    """
    LangGraph node — state'den transcript alır, özet üretir, state'e yazar.

    Beklenen state girdileri:
        state["transcript"]    : zorunlu
        state["meeting_date"]  : "YYYY-MM-DD" (deadline çözümlemesi için)
        state["meeting_id"]    : çıktı dosyası adı için

    State çıktıları:
        state["summary"]       : 📌📋📊❗ formatlı analiz
        state["summary_model"] : kullanılan model adı
        state["summary_tokens"]: harcanan token sayısı
    """
    transcript = state.get("transcript", "").strip()
    if not transcript:
        return {**state, "summary_error": "Transkript bulunamadı."}

    # Tarihi çöz
    raw_date = state.get("meeting_date")
    meeting_date = None
    if raw_date:
        try:
            meeting_date = datetime.strptime(raw_date, "%Y-%m-%d")
        except ValueError:
            pass

    # Groq çağrısı
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _build_system_prompt(meeting_date)},
            {"role": "user",   "content": f"Aşağıdaki toplantıyı analiz et:\n\n{transcript}"},
        ],
        temperature=0.2,
    )

    raw_output = response.choices[0].message.content
    model_name = response.model
    tokens     = response.usage.total_tokens

    # Kaydet
    meeting_id = state.get("meeting_id", "meeting")
    _save_output(meeting_id, raw_output, model_name, tokens)

    return {
        **state,
        "summary":        raw_output,
        "summary_model":  model_name,
        "summary_tokens": tokens,
    }
