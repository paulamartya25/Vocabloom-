---
title: Multilingual Translator
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# 🌐 Multilingual Translator – Phase 3

A production-ready multilingual translation and transcription API powered by **NLLB-200** (via CTranslate2 fast inference), a **custom hybrid model**, and **Whisper ONNX** for speech recognition — with a beautiful web UI and streaming WebSocket support.

---

## ✨ Features

| Feature | Details |
|---|---|
| **24 Languages** | English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu, Odia, Assamese, Nepali, Sinhala, French, German, Spanish, Arabic, Chinese, Japanese, Korean, Russian, Portuguese |
| **Fast Inference** | CTranslate2 int8-quantized NLLB-200, ~120ms per sentence |
| **Custom Hybrid** | Fine-tuned model optimised for Indian languages |
| **Speech-to-Text** | Whisper ONNX model tuned for Indian accents |
| **WebSocket Streaming** | Real-time word-by-word translation |
| **Web UI** | Dark theme, mobile-responsive, single HTML file |
| **REST API** | Fully documented FastAPI endpoints |

---

## 🚀 Quick Start

### Local Development with Docker Compose

```bash
# 1. Clone / navigate to this directory
git clone <your-repo-url>
cd phase3_deployment

# 2. Copy and fill in your environment variables
cp .env.example .env
# Edit .env: set HF_TOKEN and HF_USERNAME

# 3. Start with hot-reload
docker compose up --build

# 4. Open http://localhost:7860
```

### Local Development without Docker

```bash
# Install Python 3.11+ and ffmpeg
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_token
export HF_USERNAME=your_username

# Run
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

---

## 📡 REST API

### `GET /health`
Returns model loading status.
```json
{
  "status": "ok",
  "models": {"ct2": "loaded", "onnx_whisper": "loaded", "hybrid": "loaded"},
  "supported_languages": 24,
  "version": "3.0.0"
}
```

### `GET /languages`
Returns all 24 supported languages with ISO codes, names, flags, and NLLB codes.

### `POST /translate`
```json
// Request
{"text": "Hello world", "source_lang": "en", "target_lang": "hi", "model": "nllb"}

// Response
{"translated_text": "नमस्ते दुनिया", "model_used": "nllb-ct2", "latency_ms": 118.4, "source_lang": "en", "target_lang": "hi"}
```
- `source_lang`: ISO 639-1 code or `"auto"` for auto-detection
- `model`: `"nllb"` (default, fastest) or `"hybrid"` (custom fine-tuned)

### `POST /detect_language`
```json
// Request
{"text": "Bonjour le monde"}

// Response
{"detected_lang": "fr", "confidence": 0.9987, "language_name": "French"}
```

### `POST /transcribe`
Multipart form upload: `file` = WAV / MP3 / OGG / FLAC / WebM audio (max 50 MB).
```json
// Response
{"transcript": "Hello world", "detected_language": "en", "confidence": 0.95}
```

### `POST /translate_audio`
Audio → transcribe → translate pipeline.
```
file=<audio>  target_lang=hi  model=nllb
```
```json
// Response
{"transcript": "...", "translated_text": "...", "source_language": "en",
 "target_language": "hi", "model_used": "nllb-ct2", "total_latency_ms": 450.2}
```

---

## ⚡ WebSocket API

### `ws://host/ws/translate` — Streaming Translation

**Send:**
```json
{"type": "translate", "text": "Hello world", "source_lang": "en", "target_lang": "hi"}
```

**Receive (sequence):**
```json
{"type": "start", "source_lang": "en", "target_lang": "hi"}
{"type": "chunk", "text": "नमस्ते "}
{"type": "chunk", "text": "दुनिया "}
{"type": "done", "full_text": "नमस्ते दुनिया", "model_used": "nllb-ct2", "latency_ms": 120.5}
```

### `ws://host/ws/transcribe` — Streaming Transcription

**Send audio chunks:**
```json
{"type": "audio_chunk", "data": "<base64 PCM float32>", "sample_rate": 16000}
{"type": "end"}
```

**Receive:**
```json
{"type": "partial", "text": "…receiving audio…"}
{"type": "final", "text": "Hello world", "language": "en", "confidence": 0.95}
```

---

## 🏗️ Architecture

```
phase3_deployment/
├── app.py              # FastAPI application (REST + WebSocket)
├── model_loader.py     # ModelManager (CTranslate2, ONNX, HF pipeline)
├── static/
│   └── index.html      # Full web UI (embedded CSS + JS)
├── Dockerfile          # Production Docker image
├── docker-compose.yml  # Local development with hot-reload
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variable template
```

**Model Loading Priority:**
1. `{HF_USERNAME}/nllb-200-600m-qlora` (CTranslate2, int8) — fastest
2. `{HF_USERNAME}/custom-hybrid-v1` (HuggingFace pipeline) — highest quality for Indian langs
3. `facebook/nllb-200-distilled-600M` (HF pipeline) — public fallback

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes (private models) | HuggingFace API token |
| `HF_USERNAME` | Yes | Your HF username for private model repos |
| `MODEL_CACHE_DIR` | No | Model cache directory (default: `/tmp/models`) |
| `PORT` | No | Server port (default: `7860`) |
| `LOG_LEVEL` | No | Logging level: `debug`/`info`/`warning` (default: `info`) |

---

## 📊 Performance

| Model | Avg Latency | Languages | Notes |
|---|---|---|---|
| NLLB-200 CT2 (int8) | ~120ms | 24 | Recommended for production |
| Custom Hybrid | ~200ms | 24 | Best for Indian languages |
| Whisper ONNX | ~300ms | multilingual | Audio transcription |

---

*Built for Phase 3 of the Multilingual Translator project.*
