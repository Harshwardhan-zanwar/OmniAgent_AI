# ⬡ omni-agent-ai

> An agentic AI application that accepts **Text, Images, PDFs, and Audio** simultaneously — extracts content, understands intent, and autonomously chains the right tools to complete complex multi-step tasks.

**Live Demo →** `URL`

---

## 📸 Features

| Capability | Description |
|---|---|
| 📄 PDF Parsing | Text extraction + scanned PDF OCR fallback |
| 🖼️ Image OCR | Google Cloud Vision + Gemini Vision fallback |
| 🎙️ Audio Transcription | Gemini Audio API + Google STT fallback |
| ▶️ YouTube Transcripts | Auto-detects URLs in any input, fetches transcript |
| 📝 Summarization | 1-line + 3 bullets + 5-sentence format |
| 💬 Sentiment Analysis | Label + confidence bar + emotion tags |
| 💻 Code Explanation | Language detection + bug finding + complexity |
| ⚖️ Cross-Input Reasoning | Compare content across multiple uploaded files |
| 🧠 Autonomous Planning | Agent plans minimum tool sequence, no user prompting |
| 💰 Cost Estimator | Pre-flight token + cost estimate before execution |
| 🔍 Plan Trace | Animated step-by-step tool chain visible in UI |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     frontend/index.html                 │
│         Chat UI · File Upload · Plan Trace Viewer       │
└───────────────────────┬─────────────────────────────────┘
                        │ POST /chat
                        ▼
┌─────────────────────────────────────────────────────────┐
│                       main.py                           │
│              FastAPI · CORS · File Handling             │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  agent/planner.py                       │
│                                                         │
│  1. Extract content from all files                      │
│  2. Detect YouTube URLs → fetch transcripts             │
│  3. Classify intent via Gemini (confidence scored)      │
│  4. Ask follow-up if confidence < 60%                   │
│  5. Plan minimum tool sequence                          │
│  6. Execute tools → log every step                      │
│  7. Return result + plan_trace + extracted_text         │
└───────┬───────────────────────────────────────┬─────────┘
        │                                       │
        ▼                                       ▼
┌───────────────────┐                 ┌─────────────────────┐
│  agent/tools/     │                 │  utils/             │
│                   │                 │                     │
│  pdf.py           │                 │  state.py           │
│  ocr.py           │                 │  cost_estimator.py  │
│  audio.py         │                 │  logger.py          │
│  youtube.py       │                 └──────────┬──────────┘
│  summarize.py     │                            │
│  sentiment.py     │                 ┌──────────▼──────────┐
│  code_explain.py  │                 │  Upstash Redis      │
│  cross_input.py   │                 │  Session Memory     │
└───────────────────┘                 └─────────────────────┘
        │
        ▼
┌───────────────────┐
│  Google Gemini    │
│  1.5 Flash        │
│  (main AI brain)  │
└───────────────────┘
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/Harshwardhan-zanwar/OmniAgent_AI
cd omni-agent-ai
```

### 2. Create virtual environment
```powershell
# Windows PowerShell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Requires `ffmpeg` installed on your system for audio processing.
> Windows: `winget install ffmpeg`

### 5. Run the app
```bash
python main.py
```

Visit **http://localhost:8000** 🎉

## 📁 Project Structure

```
omni-agent-ai/
├── main.py                    # FastAPI entry point
├── agent/
│   ├── __init__.py
│   ├── planner.py             # Agent brain — intent + tool orchestration
│   └── tools/
│       ├── __init__.py
│       ├── pdf.py             # PDF extraction (LangChain + OCR fallback)
│       ├── ocr.py             # Image OCR (Cloud Vision + Gemini fallback)
│       ├── audio.py           # Audio transcription (Gemini + STT fallback)
│       ├── youtube.py         # YouTube transcript fetcher
│       ├── summarize.py       # 3-format summarization
│       ├── sentiment.py       # Sentiment + confidence + emotions
│       ├── code_explain.py    # Code explanation + bug detection
│       └── cross_input.py     # Multi-file comparative reasoning
├── utils/
│   ├── __init__.py
│   ├── state.py               # Session memory (LangChain + Redis)
│   ├── cost_estimator.py      # Token counting + cost estimation
│   └── logger.py              # Rich logging + plan trace formatter
├── frontend/
│   └── index.html             # Full chat UI (no build step)
├── tests/
│   └── test_agent.py          # End-to-end test cases
├── sample_inputs/             # Demo files for evaluators
├── .env.example               # Environment variable template
├── render.yaml                # Render deployment config
├── requirements.txt           # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML + CSS + JS (no framework) |
| Backend | FastAPI + Uvicorn |
| AI Brain | Google Gemini 2.5 Flash Lite|
| OCR | Google Cloud Vision + Gemini Vision |
| Audio | Gemini Audio API + Google STT |
| PDF | LangChain PyPDFLoader + pdfplumber |
| Memory | LangChain ConversationMemory + Upstash Redis |
| Prompts | LangChain PromptTemplate |
---