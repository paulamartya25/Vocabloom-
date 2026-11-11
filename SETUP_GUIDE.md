# 🛠️ Complete Setup Guide

This guide walks you through every step to run the full multilingual translator pipeline.

---

## Prerequisites

### Accounts Needed
1. **Hugging Face** account → https://huggingface.co/join
   - Generate token: Settings → Access Tokens → New Token (write access)
2. **Kaggle** account → https://www.kaggle.com
   - Enable GPU in notebook settings

### Local Tools Needed
- Python 3.10+
- Docker Desktop
- Git
- Git LFS (for large model files)

---

## Step 0: Initial Setup

```bash
# Clone/download this project
cd c:\Users\amart\Desktop\LANGUAGE_TRANSLATER

# Install git-lfs
git lfs install

# Login to HuggingFace CLI
pip install huggingface_hub
huggingface-cli login  # Paste your HF token
```

---

## Step 1: Kaggle Secrets Setup

Before running any Kaggle notebook, add these secrets:

1. Go to Kaggle → Your Account → Settings → API → Secrets
2. Add secret: `HF_TOKEN` = your HuggingFace token
3. Add secret: `HF_USERNAME` = your HuggingFace username

---

## Step 2: Phase 1 — Dataset + Custom Model Training (~6-8 hours on 2x T4)

### Upload to Kaggle:
1. Go to https://kaggle.com/code → New Notebook
2. Upload `phase1_kaggle/phase1_notebook.ipynb`
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Enable Internet: Settings → Internet → On
5. Add secrets: HF_TOKEN, HF_USERNAME
6. Click **Run All**

### What happens:
- Downloads multilingual corpora (~50GB compressed, ~15GB after filtering)
- Builds unified dataset with 24 languages
- Pushes dataset to: `{HF_USERNAME}/multilingual-translation-data`
- Trains XLM-RoBERTa + Custom 6-Layer Decoder
- Pushes best checkpoint to: `{HF_USERNAME}/custom-hybrid-v1`

### Expected output on HF Hub:
```
{HF_USERNAME}/multilingual-translation-data  ← Dataset
{HF_USERNAME}/custom-hybrid-v1               ← Model checkpoint
  ├── config.json
  ├── pytorch_model.bin (or model.safetensors)
  ├── tokenizer.json
  └── training_logs.json
```

---

## Step 3: Phase 2 — Fine-tuning + Export (~4-6 hours on 2x T4)

### Upload to Kaggle:
1. Create new Kaggle notebook
2. Upload `phase2_kaggle/phase2_notebook.ipynb`
3. Enable GPU T4 x2 + Internet
4. Add secrets
5. Click **Run All**

### What happens:
- Fine-tunes NLLB-200-600M with QLoRA on your dataset
- Fine-tunes Whisper-Small for Indian accents
- Exports both to ONNX and CTranslate2 formats
- Pushes to HF Hub

### Expected output on HF Hub:
```
{HF_USERNAME}/nllb-200-600m-qlora
  ├── adapter_config.json          ← LoRA weights
  ├── adapter_model.bin
  ├── ct2_model/                   ← CTranslate2 (fast inference)
  │   ├── model.bin
  │   └── vocabulary.json
  └── onnx/                        ← ONNX model
      └── model.onnx

{HF_USERNAME}/whisper-small-indian-lora
  ├── adapter_config.json
  ├── adapter_model.bin
  └── onnx/
      └── model.onnx
```

---

## Step 4: Phase 3 — Deploy to HuggingFace Spaces

### Option A: Deploy to HF Spaces (Public URL, free GPU)

```bash
# Install HF Hub CLI
pip install "huggingface_hub[cli]"

# Create a new Space
huggingface-cli repo create multilingual-translator --type space --space_sdk docker

# Clone the space
git clone https://huggingface.co/spaces/{YOUR_USERNAME}/multilingual-translator
cd multilingual-translator

# Copy deployment files
xcopy /E /I "c:\Users\amart\Desktop\LANGUAGE_TRANSLATER\phase3_deployment\*" .

# Set HF_TOKEN as Space secret (via HF UI):
# Space → Settings → Variables and Secrets → Add Secret
# Name: HF_TOKEN, Value: your_token

# Push
git add .
git commit -m "🚀 Deploy multilingual translator"
git push

# Wait 3-5 minutes for build
# Your app will be at: https://{username}-multilingual-translator.hf.space
```

### Option B: Run Locally with Docker

```bash
cd c:\Users\amart\Desktop\LANGUAGE_TRANSLATER\phase3_deployment

# Copy and fill in environment file
copy .env.example .env
# Edit .env with your HF_TOKEN and HF_USERNAME

# Build and run
docker-compose up --build

# Open browser
start http://localhost:7860
```

### Option C: Run Locally without Docker (Development)

```bash
cd c:\Users\amart\Desktop\LANGUAGE_TRANSLATER\phase3_deployment

# Install requirements
pip install -r requirements.txt

# Set environment variables
set HF_TOKEN=your_token_here
set HF_USERNAME=your_username_here

# Run
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Open browser
start http://localhost:7860
```

---

## Step 5: Access from Your Phone

### HF Spaces (Recommended):
- Just open `https://{username}-multilingual-translator.hf.space` on your phone browser!

### Local (same WiFi network):
1. Find your computer's local IP: `ipconfig` → IPv4 Address (e.g., `192.168.1.100`)
2. Open on phone: `http://192.168.1.100:7860`

---

## 🔥 Tips & Troubleshooting

### Kaggle out of disk space:
- Phase 1 uses ~40GB during training. Use Kaggle's 100GB scratch disk.
- Add to notebook: `!df -h /kaggle/working` to monitor

### HF Hub rate limits:
- Use `huggingface_hub.login(token=HF_TOKEN)` before pushing
- For large model files, use `upload_large_folder()` instead of `push_to_hub()`

### CUDA out of memory:
- Reduce `per_device_train_batch_size` from 8 → 4
- Increase `gradient_accumulation_steps` from 4 → 8

### Docker build fails:
- Ensure Docker Desktop is running
- Try: `docker system prune` to free space

### Models not loading on HF Spaces:
- Check Space logs: Space → Logs tab
- Ensure `HF_TOKEN` secret is set in Space settings
- Free tier Spaces have 16GB RAM; models need ~8GB

---

## 📊 Cost Estimate

| Resource | Cost |
|----------|------|
| Kaggle GPU (Phase 1) | FREE (30 hrs/week quota) |
| Kaggle GPU (Phase 2) | FREE |
| HuggingFace Spaces (CPU) | FREE |
| HuggingFace Spaces (GPU T4) | ~\$0.60/hr (optional upgrade) |
| HF Hub storage | FREE up to 10GB per repo |

**Total estimated cost: \$0 (all free tiers)** ✅

---

## 📞 Support

If you hit issues:
1. Check the HF Space logs
2. Check Kaggle notebook error output
3. Verify your HF_TOKEN has write permissions
