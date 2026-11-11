# 🌐 Multilingual Translator — Full Stack AI System

A production-grade multilingual translation system covering **24 languages** (Indian + European + CJK), built with custom-trained transformer models and deployed on Hugging Face Spaces.

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Kaggle Training                      │
│                                                                  │
│  Dataset Builder ──► XLM-RoBERTa Encoder                       │
│  (24 languages)       + 6-Layer Custom Decoder                  │
│                           = custom-hybrid-v1                    │
│                               │                                  │
│                               ▼ Push to HF Hub                  │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: Kaggle Fine-tuning                   │
│                                                                  │
│  NLLB-200-600M ──► QLoRA Fine-tune ──► ONNX + CTranslate2      │
│  Whisper-Small ──► Indian Accent LoRA ──► ONNX Export           │
│                               │                                  │
│                               ▼ Push to HF Hub                  │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                PHASE 3: HuggingFace Spaces Deployment            │
│                                                                  │
│  FastAPI Backend ──► Pulls models from HF Hub                   │
│  Docker Container     WebSocket streaming endpoint              │
│  Web UI               REST API                                  │
│  (mobile-ready)       Voice transcription + translation         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
LANGUAGE_TRANSLATER/
├── README.md                          ← You are here
├── SETUP_GUIDE.md                     ← Step-by-step setup instructions
│
├── phase1_kaggle/                     ← Phase 1: Dataset + Custom Model Training
│   ├── dataset_builder.py             ← Multilingual dataset builder
│   ├── train_hybrid_model.py          ← XLM-RoBERTa + Custom Decoder training
│   ├── phase1_notebook.ipynb          ← Kaggle notebook
│   └── requirements.txt
│
├── phase2_kaggle/                     ← Phase 2: Fine-tuning + Export
│   ├── finetune_nllb.py               ← NLLB-200 QLoRA fine-tuning
│   ├── finetune_whisper.py            ← Whisper Indian accent LoRA
│   ├── export_models.py               ← ONNX + CTranslate2 export
│   ├── phase2_notebook.ipynb          ← Kaggle notebook
│   └── requirements.txt
│
└── phase3_deployment/                 ← Phase 3: FastAPI + Docker deployment
    ├── app.py                         ← FastAPI application
    ├── model_loader.py                ← Model management module
    ├── static/
    │   └── index.html                 ← Web UI (mobile-ready)
    ├── Dockerfile                     ← Docker container config
    ├── docker-compose.yml             ← Local development
    ├── requirements.txt
    ├── .env.example
    └── README.md                      ← HF Spaces README with YAML frontmatter
```

---

## 🌍 Supported Languages

| Region | Languages |
|--------|-----------|
| 🇮🇳 Indian | Hindi (hi), Bengali (bn), Tamil (ta), Telugu (te), Marathi (mr), Gujarati (gu), Kannada (kn), Malayalam (ml), Punjabi (pa), Urdu (ur) |
| 🇪🇺 European | French (fr), German (de), Spanish (es), Italian (it), Portuguese (pt), Dutch (nl), Polish (pl), Romanian (ro), Swedish (sv), Czech (cs) |
| 🌏 CJK | Chinese (zh), Japanese (ja), Korean (ko) |
| 🌐 Pivot | English (en) |

**Total: 24 languages = 552 translation pairs**

---

## 🤖 Models

| Model | Purpose | Training | Inference |
|-------|---------|----------|-----------|
| `custom-hybrid-v1` | General multilingual translation | XLM-RoBERTa encoder + 6-layer custom decoder | PyTorch |
| `nllb-200-600m-qlora` | Conversational translation | QLoRA fine-tune of NLLB-200-600M | CTranslate2 (fast) |
| `whisper-small-indian-lora` | Indian accent ASR | LoRA fine-tune of Whisper-Small | ONNX Runtime |

---

## 🚀 Quick Start

### Prerequisites
- Kaggle account with GPU access
- Hugging Face account + token
- Docker (for local deployment)

### Step 1: Configure
```bash
# Copy and edit environment file
cp phase3_deployment/.env.example phase3_deployment/.env
# Edit HF_TOKEN and HF_USERNAME
```

### Step 2: Run Phase 1 on Kaggle
1. Upload `phase1_kaggle/` to Kaggle
2. Enable GPU (2x T4 or P100)
3. Add `HF_TOKEN` to Kaggle secrets
4. Run `phase1_notebook.ipynb`

### Step 3: Run Phase 2 on Kaggle
1. Upload `phase2_kaggle/` to Kaggle
2. Enable GPU
3. Run `phase2_notebook.ipynb`

### Step 4: Deploy to HF Spaces
```bash
# Clone your HF Space
git clone https://huggingface.co/spaces/{YOUR_USERNAME}/multilingual-translator
# Copy phase3_deployment/ files into it
cp -r phase3_deployment/* multilingual-translator/
cd multilingual-translator
git add . && git commit -m "Deploy multilingual translator"
git push
```

### Step 5: Local Development
```bash
cd phase3_deployment
cp .env.example .env  # Fill in your tokens
docker-compose up
# Open http://localhost:7860
```

---

## 📡 API Reference

### REST Endpoints

```bash
# Translate text
curl -X POST http://localhost:7860/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_lang": "en", "target_lang": "hi"}'

# Detect language
curl -X POST http://localhost:7860/detect_language \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते दुनिया"}'

# Transcribe audio
curl -X POST http://localhost:7860/transcribe \
  -F "audio=@speech.wav"
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:7860/ws/translate');
ws.send(JSON.stringify({
  type: 'translate',
  text: 'Hello world',
  source_lang: 'en',
  target_lang: 'hi'
}));
```

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Translation latency (CTranslate2) | ~50-200ms |
| Translation latency (Transformers) | ~300-800ms |
| Whisper transcription | ~1-3s per 30s audio |
| Supported concurrent WebSocket connections | 50+ |
| BLEU score (NLLB fine-tuned, en→hi) | ~35-45 |

---

## 🔧 Tech Stack

- **ML**: PyTorch, HuggingFace Transformers, PEFT, BitsAndBytes
- **Inference**: CTranslate2, ONNX Runtime, Optimum
- **Backend**: FastAPI, Uvicorn, WebSockets
- **Frontend**: Vanilla JS, CSS3 (glassmorphism)
- **Containerization**: Docker, Docker Compose
- **Model Hosting**: HuggingFace Hub
- **Training**: Kaggle GPUs (T4/P100)
- **Deployment**: HuggingFace Spaces

---

## 📄 License

MIT License — See LICENSE file for details.
