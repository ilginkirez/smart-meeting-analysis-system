# 🎙️ Akıllı Toplantı Analiz ve Yönetim Sistemi

> Yapay zeka tabanlı çok ajanlı toplantı analiz platformu — toplantıyı kaydetmekten çıkarıp anlaşılır, yorumlanabilir ve aksiyona dönüştürülebilir hale getiren uçtan uca bir sistem.

## 🏗️ Mimari

```
Ses Dosyası
     ↓
[Preprocessor]         ← Ön işleme (FFmpeg/PyDub)
     ↓
[Whisper]              ← Transkripsiyon Ajanı
     ↓
[Pyannote + ECAPA-TDNN] ← Diyarizasyon Ajanı
     ↓
[LangGraph Orkestratör]
   ↙        ↓         ↘
[GPT-4]  [spaCy+GPT-4]  [RoBERTa+librosa]
Özetleme  Aksiyon Ajanı   Duygu Ajanı
   ↘        ↓         ↙
      [FastAPI Backend]
           ↓
      [Streamlit UI]
```

## 👥 Geliştirici Rolleri

| Modül | Sorumluluk | Branch |
|-------|-----------|--------|
| **Kişi A** — Ses & Diyarizasyon | `src/audio/`, `src/emotion/` | `feature/audio-pipeline` |
| **Kişi B** — NLP & Ajanlar | `src/nlp/`, `src/agents/` | `feature/nlp-agents` |
| **Ortak** | `src/api/`, `src/frontend/`, `tests/` | `dev` → `main` |

## 📁 Proje Yapısı

```
smart-meeting-analysis-system/
├── interface_schema.json    ← 🔒 DOKUNULMAZ — ortak sözleşme
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py            ← Ortak ayarlar
│   ├── utils.py             ← Ortak yardımcı fonksiyonlar
│   ├── audio/               ← 🔴 Kişi A
│   │   ├── preprocessor.py
│   │   ├── transcriber.py
│   │   ├── diarizer.py
│   │   └── audio_pipeline.py
│   ├── nlp/                 ← 🔵 Kişi B
│   │   ├── summarizer.py
│   │   ├── action_extractor.py
│   │   └── rag_pipeline.py
│   ├── emotion/             ← 🔴 Kişi A
│   │   ├── text_emotion.py
│   │   ├── acoustic_emotion.py
│   │   └── fusion.py
│   ├── agents/              ← 🔵 Kişi B
│   │   ├── orchestrator.py
│   │   ├── summary_agent.py
│   │   ├── action_agent.py
│   │   └── emotion_agent.py
│   ├── api/                 ← Ortak
│   │   ├── main.py
│   │   └── models.py
│   └── frontend/            ← Ortak
│       └── app.py
├── tests/
│   ├── test_audio.py        ← 🔴 Kişi A yazar
│   └── test_nlp.py          ← 🔵 Kişi B yazar
├── notebooks/
│   ├── audio_experiments/   ← 🔴 Kişi A deneyleri
│   └── nlp_experiments/     ← 🔵 Kişi B deneyleri
└── data/
    ├── ami/                 ← AMI Corpus (git'e dahil değil)
    └── sample/              ← Test için küçük örnek dosyalar
```

## 🚀 Kurulum

```bash
# 1. Repo'yu klonlayın
git clone https://github.com/<username>/smart-meeting-analysis-system.git
cd smart-meeting-analysis-system

# 2. Virtual environment oluşturun
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. .env dosyasını oluşturun
copy .env.example .env
# Sonra .env dosyasına API key'lerinizi yazın
```

## 🔀 Git Workflow

```bash
# Branch yapısı
main                        ← Sadece çalışan, test edilmiş kod
├── dev                     ← Entegrasyon branch'i
│   ├── feature/audio-pipeline   ← Kişi A çalışır
│   └── feature/nlp-agents       ← Kişi B çalışır

# Altın Kural: Birbirinizin branch'ine asla dokunmayın!
# Haftalık Cuma merge: feature/* → dev → (milestone bitince) main
```

## 🔗 Arayüz Sözleşmesi

`interface_schema.json` dosyası Kişi A'nın **ürettiği** ve Kişi B'nin **tükettiği** veri formatını tanımlar. Bu dosyayı **iki kişi birlikte karar vermeden değiştirmeyin**.

Kişi B, Kişi A'yı beklemeden `data/sample/mock_transcript.json` dosyasını kullanarak pipeline'ını geliştirebilir.

## 📊 Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| ASR | OpenAI Whisper |
| Diyarizasyon | Pyannote.audio + ECAPA-TDNN |
| NLP/Özetleme | LangChain + GPT-4 |
| Duygu Analizi | RoBERTa + librosa |
| Agent Koordinasyon | LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |
| Eval | jiwer (WER), pyannote.metrics (DER), rouge-score |
