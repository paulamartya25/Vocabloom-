"""
app.py – Phase 3 FastAPI Multilingual Translator
=================================================
Endpoints:
  GET  /                  → Serve index.html
  GET  /health            → Model status
  GET  /languages         → Supported language list
  POST /translate         → Text translation
  POST /detect_language   → Language detection
  POST /transcribe        → Audio → transcript
  POST /translate_audio   → Audio → transcript → translation
  WS   /ws/translate      → Streaming text translation
  WS   /ws/transcribe     → Streaming audio transcription
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("translator.app")

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
HF_USERNAME = os.getenv("HF_USERNAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", None)
STATIC_DIR = Path(__file__).parent / "static"
PORT = int(os.getenv("PORT", 7860))

# ---------------------------------------------------------------------------
# Import model manager (lazy – actual loading happens in startup)
# ---------------------------------------------------------------------------
from model_loader import (  # noqa: E402
    model_manager,
    get_supported_languages,
    iso_to_nllb,
    LANGUAGE_MAP,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multilingual Translator API",
    description="Phase 3 – FastAPI backend with CTranslate2, ONNX Whisper, and WebSocket streaming",
    version="3.0.0",
)

# CORS – allow all origins for development / HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Rate limiter (sliding window, in-memory)
# ---------------------------------------------------------------------------
_rate_store: Dict[str, list] = defaultdict(list)
RATE_LIMIT = 10  # requests per second per IP
RATE_WINDOW = 1.0  # seconds


def _check_rate_limit(ip: str) -> bool:
    now = time.monotonic()
    window = _rate_store[ip]
    # Prune old timestamps
    _rate_store[ip] = [t for t in window if now - t < RATE_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        return False
    _rate_store[ip].append(now)
    return True


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/ws"):
        return await call_next(request)
    ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Max 10 requests/second per IP."},
        )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "%s %s → %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# Translation LRU cache (key = text + src + tgt + model)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1000)
def _cached_translate(text: str, src_lang: str, tgt_lang: str, model: str) -> tuple:
    """Cached wrapper around model_manager.translate(); returns (result, model_used, latency_ms)."""
    return model_manager.translate(text, src_lang, tgt_lang, model)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Load models in a background thread so the server starts fast."""
    asyncio.create_task(_load_models_async())


async def _load_models_async():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_models_sync)


def _load_models_sync():
    logger.info("=== Starting model loading ===")

    # 1. CTranslate2 NLLB model
    ct2_repo = f"{HF_USERNAME}/nllb-200-600m-qlora" if HF_USERNAME else ""
    loaded_ct2 = False
    if ct2_repo:
        loaded_ct2 = model_manager.load_ct2_model(ct2_repo)
    if not loaded_ct2:
        logger.info("Falling back to facebook/nllb-200-distilled-600M via HF pipeline")
        model_manager.load_hf_pipeline("facebook/nllb-200-distilled-600M")

    # 2. Custom Hybrid model
    hybrid_repo = f"{HF_USERNAME}/custom-hybrid-v1" if HF_USERNAME else ""
    if hybrid_repo:
        model_manager.load_hybrid_model(hybrid_repo)

    # 3. ONNX Whisper
    onnx_repo = f"{HF_USERNAME}/whisper-small-indian-lora" if HF_USERNAME else ""
    if onnx_repo:
        model_manager.load_onnx_model(onnx_repo)

    logger.info("=== Model loading complete: %s ===", model_manager.status)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    source_lang: str = Field(..., min_length=2, max_length=10)
    target_lang: str = Field(..., min_length=2, max_length=10)
    model: str = Field("nllb", pattern="^(nllb|hybrid|auto)$")


class TranslateResponse(BaseModel):
    translated_text: str
    model_used: str
    latency_ms: float
    source_lang: str
    target_lang: str


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class DetectResponse(BaseModel):
    detected_lang: str
    confidence: float
    language_name: str


class TranscribeResponse(BaseModel):
    transcript: str
    detected_language: str
    confidence: float


class TranslateAudioResponse(BaseModel):
    transcript: str
    translated_text: str
    source_language: str
    target_language: str
    model_used: str
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Helper: audio bytes → numpy array
# ---------------------------------------------------------------------------

def _bytes_to_audio(data: bytes, content_type: str = "") -> np.ndarray:
    """Convert uploaded audio bytes to a float32 numpy array at 16 kHz."""
    try:
        import soundfile as sf

        buf = io.BytesIO(data)
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)

        # Resample to 16 kHz if needed
        if sr != 16000:
            try:
                import resampy
                audio = resampy.resample(audio, sr, 16000)
            except ImportError:
                # Basic linear interpolation fallback
                target_len = int(len(audio) * 16000 / sr)
                audio = np.interp(
                    np.linspace(0, len(audio), target_len),
                    np.arange(len(audio)),
                    audio,
                )
        return audio.astype(np.float32)

    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> tuple[str, float]:
    """Return (iso_code, confidence) using langdetect."""
    try:
        from langdetect import detect_langs

        results = detect_langs(text)
        if results:
            top = results[0]
            lang = top.lang
            conf = float(top.prob)
            # Map 2-letter codes to our supported set
            if lang in LANGUAGE_MAP:
                return lang, conf
            # zh-cn, zh-tw → zh
            if lang.startswith("zh"):
                return "zh", conf
            # Try first 2 chars
            short = lang[:2]
            if short in LANGUAGE_MAP:
                return short, conf
        return "en", 0.5
    except Exception:
        return "en", 0.5


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        {"message": "Multilingual Translator API v3", "docs": "/docs"},
        status_code=200,
    )


@app.get("/health")
async def health():
    """Return current model loading status."""
    return {
        "status": "ok",
        "models": model_manager.status,
        "supported_languages": len(LANGUAGE_MAP),
        "version": "3.0.0",
    }


@app.get("/languages")
async def languages():
    """Return all supported languages with metadata."""
    return {"languages": get_supported_languages()}


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    """Translate text between supported languages."""
    # Validate language codes
    if req.source_lang not in LANGUAGE_MAP and req.source_lang != "auto":
        raise HTTPException(400, f"Unsupported source language: {req.source_lang}")
    if req.target_lang not in LANGUAGE_MAP:
        raise HTTPException(400, f"Unsupported target language: {req.target_lang}")

    src = req.source_lang
    if src == "auto":
        src, _ = _detect_language(req.text)

    if src == req.target_lang:
        return TranslateResponse(
            translated_text=req.text,
            model_used="passthrough",
            latency_ms=0.0,
            source_lang=src,
            target_lang=req.target_lang,
        )

    try:
        translated, model_used, latency_ms = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _cached_translate(req.text, src, req.target_lang, req.model),
        )
    except RuntimeError as exc:
        raise HTTPException(503, f"Translation service unavailable: {exc}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Translation error")
        raise HTTPException(500, f"Internal translation error: {exc}")

    return TranslateResponse(
        translated_text=translated,
        model_used=model_used,
        latency_ms=round(latency_ms, 2),
        source_lang=src,
        target_lang=req.target_lang,
    )


@app.post("/detect_language", response_model=DetectResponse)
async def detect_language(req: DetectRequest):
    """Detect the language of the given text."""
    lang, confidence = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _detect_language(req.text)
    )
    name = LANGUAGE_MAP.get(lang, {}).get("name", lang)
    return DetectResponse(detected_lang=lang, confidence=round(confidence, 4), language_name=name)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio (WAV/MP3/OGG/FLAC)."""
    allowed = {"audio/wav", "audio/mpeg", "audio/ogg", "audio/flac", "audio/x-wav",
               "audio/mp3", "audio/webm", "application/octet-stream"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(415, f"Unsupported audio format: {file.content_type}")

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:  # 50 MB
        raise HTTPException(413, "Audio file too large (max 50 MB)")

    try:
        audio = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _bytes_to_audio(data, file.content_type or "")
        )
        transcript, detected_lang, confidence = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model_manager.transcribe_onnx(audio)
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, f"Transcription service unavailable: {exc}")
    except Exception as exc:
        logger.exception("Transcription error")
        raise HTTPException(500, f"Internal transcription error: {exc}")

    return TranscribeResponse(
        transcript=transcript,
        detected_language=detected_lang,
        confidence=round(confidence, 4),
    )


@app.post("/translate_audio", response_model=TranslateAudioResponse)
async def translate_audio(
    file: UploadFile = File(...),
    target_lang: str = Form("en"),
    model: str = Form("nllb"),
):
    """Audio upload → transcribe → translate pipeline."""
    if target_lang not in LANGUAGE_MAP:
        raise HTTPException(400, f"Unsupported target language: {target_lang}")

    t_start = time.perf_counter()
    data = await file.read()

    # Step 1: Transcribe
    try:
        audio = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _bytes_to_audio(data, file.content_type or "")
        )
        transcript, detected_lang, _ = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model_manager.transcribe_onnx(audio)
        )
    except Exception as exc:
        raise HTTPException(503, f"Transcription failed: {exc}")

    # Step 2: Translate
    try:
        translated, model_used, _ = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model_manager.translate(transcript, detected_lang, target_lang, model)
        )
    except Exception as exc:
        raise HTTPException(503, f"Translation failed: {exc}")

    total_ms = (time.perf_counter() - t_start) * 1000
    return TranslateAudioResponse(
        transcript=transcript,
        translated_text=translated,
        source_language=detected_lang,
        target_language=target_lang,
        model_used=model_used,
        total_latency_ms=round(total_ms, 2),
    )


# ---------------------------------------------------------------------------
# WebSocket Endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket):
    """
    Streaming translation WebSocket.
    Client sends: {"type": "translate", "text": "...", "source_lang": "en", "target_lang": "hi"}
    Server sends: {"type": "chunk", "text": "..."} … {"type": "done", "full_text": "..."}
    """
    await websocket.accept()
    logger.info("WS /ws/translate connected: %s", websocket.client)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type != "translate":
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
                continue

            text = msg.get("text", "").strip()
            src = msg.get("source_lang", "auto")
            tgt = msg.get("target_lang", "en")
            model = msg.get("model", "nllb")

            if not text:
                await websocket.send_json({"type": "error", "message": "Empty text"})
                continue

            if tgt not in LANGUAGE_MAP:
                await websocket.send_json({"type": "error", "message": f"Unsupported target: {tgt}"})
                continue

            if src == "auto":
                src, _ = _detect_language(text)

            # Simulate streaming by splitting translation into word chunks
            await websocket.send_json({"type": "start", "source_lang": src, "target_lang": tgt})

            try:
                # Run translation in executor (CPU-bound)
                translated, model_used, latency_ms = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model_manager.translate(text, src, tgt, model),
                )
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            # Stream words back one-by-one for live feel
            words = translated.split()
            accumulated = []
            for i, word in enumerate(words):
                accumulated.append(word)
                await websocket.send_json({"type": "chunk", "text": word + " "})
                # Small delay between chunks for streaming effect
                if i % 3 == 2:
                    await asyncio.sleep(0.01)

            await websocket.send_json(
                {
                    "type": "done",
                    "full_text": translated,
                    "model_used": model_used,
                    "latency_ms": round(latency_ms, 2),
                }
            )

    except WebSocketDisconnect:
        logger.info("WS /ws/translate disconnected")
    except Exception as exc:
        logger.exception("WS /ws/translate error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    Streaming audio transcription WebSocket.
    Client sends base64-encoded PCM chunks (JSON):
      {"type": "audio_chunk", "data": "<base64>", "sample_rate": 16000}
      {"type": "end"}
    Server responds:
      {"type": "partial", "text": "..."}
      {"type": "final", "text": "...", "language": "en", "confidence": 0.9}
    """
    import base64

    await websocket.accept()
    logger.info("WS /ws/transcribe connected: %s", websocket.client)

    audio_buffer: list[np.ndarray] = []

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "audio_chunk":
                try:
                    chunk_bytes = base64.b64decode(msg["data"])
                    sample_rate = int(msg.get("sample_rate", 16000))
                    # Interpret raw bytes as float32 PCM
                    chunk_array = np.frombuffer(chunk_bytes, dtype=np.float32)
                    audio_buffer.append(chunk_array)
                    await websocket.send_json(
                        {"type": "partial", "text": "…receiving audio…"}
                    )
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": f"Chunk error: {exc}"})
                continue

            if msg_type == "end":
                if not audio_buffer:
                    await websocket.send_json({"type": "error", "message": "No audio received"})
                    audio_buffer = []
                    continue

                full_audio = np.concatenate(audio_buffer).astype(np.float32)
                audio_buffer = []

                try:
                    transcript, detected_lang, confidence = (
                        await asyncio.get_event_loop().run_in_executor(
                            None, lambda: model_manager.transcribe_onnx(full_audio)
                        )
                    )
                    await websocket.send_json(
                        {
                            "type": "final",
                            "text": transcript,
                            "language": detected_lang,
                            "confidence": round(confidence, 4),
                        }
                    )
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("WS /ws/transcribe disconnected")
    except Exception as exc:
        logger.exception("WS /ws/transcribe error: %s", exc)


# ---------------------------------------------------------------------------
# Entry-point for local development
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        workers=1,
        log_level=LOG_LEVEL.lower(),
    )
