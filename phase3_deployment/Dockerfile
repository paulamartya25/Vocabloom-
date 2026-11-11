FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        build-essential \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies (install before copying source for better layer caching)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY app.py model_loader.py ./
COPY static/ ./static/

# ---------------------------------------------------------------------------
# Environment variables (overridable at runtime)
# ---------------------------------------------------------------------------
ENV HF_TOKEN=""
ENV HF_USERNAME=""
ENV MODEL_CACHE_DIR=/tmp/models
ENV PORT=7860
ENV LOG_LEVEL=info
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Pre-create model cache dir
RUN mkdir -p /tmp/models

# ---------------------------------------------------------------------------
# Optional: pre-download the fallback NLLB model at build time
# (comment out to keep image smaller; it will download on first request instead)
# ---------------------------------------------------------------------------
# ARG HF_TOKEN_BUILD=""
# RUN if [ -n "$HF_TOKEN_BUILD" ]; then \
#       python -c "from transformers import NllbTokenizerFast; \
#                  NllbTokenizerFast.from_pretrained('facebook/nllb-200-distilled-600M', cache_dir='/tmp/models')"; \
#     fi

# ---------------------------------------------------------------------------
# Expose port (HuggingFace Spaces default = 7860)
# ---------------------------------------------------------------------------
EXPOSE 7860

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--log-level", "info"]
