"""
Phase 2 — Standalone Model Export Script
==========================================
Loads merged NLLB and Whisper models from disk (or the Hub) and:
  1. Exports to ONNX (optimum) with int8 quantization (onnxruntime)
  2. Converts NLLB to CTranslate2 format (int8)
  3. Pushes all artifacts to the Hugging Face Hub
  4. Benchmarks and reports inference times and file sizes

Usage (Kaggle):
    python export_models.py \
        --nllb_model_dir /kaggle/working/nllb_merged \
        --whisper_model_dir /kaggle/working/whisper_merged \
        --output_base /kaggle/working/exports \
        --hf_username YOUR_USERNAME \
        --hf_token YOUR_TOKEN
"""

import os
import gc
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import HfApi, login

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NLLB and Whisper to ONNX / CTranslate2")

    # Model sources
    parser.add_argument("--nllb_model_dir", type=str, default="/kaggle/working/nllb_merged",
                        help="Path to merged NLLB model (local) or Hub model ID")
    parser.add_argument("--whisper_model_dir", type=str, default="/kaggle/working/whisper_merged",
                        help="Path to merged Whisper model (local) or Hub model ID")

    # Output
    parser.add_argument("--output_base", type=str, default="/kaggle/working/exports",
                        help="Base directory for all exported artifacts")

    # Hub
    parser.add_argument("--hf_username", type=str, default=os.environ.get("HF_USERNAME", ""))
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--nllb_hub_repo", type=str, default="",
                        help="Hub repo for NLLB exports (default: {hf_username}/nllb-200-600m-qlora)")
    parser.add_argument("--whisper_hub_repo", type=str, default="",
                        help="Hub repo for Whisper exports (default: {hf_username}/whisper-small-indian-lora)")

    # Flags
    parser.add_argument("--export_nllb", action="store_true", default=True)
    parser.add_argument("--export_whisper", action="store_true", default=True)
    parser.add_argument("--quantize_onnx", action="store_true", default=True)
    parser.add_argument("--export_ct2", action="store_true", default=True)
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    parser.add_argument("--run_benchmark", action="store_true", default=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

def export_nllb_to_onnx(model_dir: str, onnx_dir: str) -> str:
    """Export NLLB model to ONNX using optimum CLI / API."""
    from optimum.exporters.onnx import main_export

    logger.info(f"Exporting NLLB to ONNX: {model_dir} → {onnx_dir}")
    Path(onnx_dir).mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=model_dir,
        output=onnx_dir,
        task="seq2seq-lm",
        opset=17,
        device="cpu",
        fp16=False,
        no_post_process=False,
    )
    logger.info("NLLB ONNX export complete.")
    return onnx_dir


def export_whisper_to_onnx(model_dir: str, onnx_dir: str) -> str:
    """Export Whisper model to ONNX using optimum."""
    from optimum.exporters.onnx import main_export

    logger.info(f"Exporting Whisper to ONNX: {model_dir} → {onnx_dir}")
    Path(onnx_dir).mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=model_dir,
        output=onnx_dir,
        task="automatic-speech-recognition",
        opset=17,
        device="cpu",
        fp16=False,
    )
    logger.info("Whisper ONNX export complete.")
    return onnx_dir


# ---------------------------------------------------------------------------
# ONNX int8 quantization
# ---------------------------------------------------------------------------

def quantize_onnx_model(onnx_path: str, quantized_path: str, weight_type: str = "INT8"):
    """
    Post-training static quantization of an ONNX model to int8
    using onnxruntime's quantization utilities.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    logger.info(f"Quantizing {onnx_path} → {quantized_path} (dynamic {weight_type}) …")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8 if weight_type == "INT8" else QuantType.QUInt8,
        per_channel=False,
        reduce_range=False,
        optimize_model=True,
    )
    logger.info("ONNX quantization complete.")


def quantize_all_onnx_in_dir(onnx_dir: str, quantized_dir: str):
    """Find all .onnx files in a directory and quantize each one."""
    Path(quantized_dir).mkdir(parents=True, exist_ok=True)
    onnx_files = list(Path(onnx_dir).rglob("*.onnx"))
    if not onnx_files:
        logger.warning(f"No .onnx files found in {onnx_dir}")
        return

    for onnx_file in onnx_files:
        relative = onnx_file.relative_to(onnx_dir)
        out_file = Path(quantized_dir) / relative.parent / (relative.stem + "_int8.onnx")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            quantize_onnx_model(str(onnx_file), str(out_file))
        except Exception as e:
            logger.warning(f"Could not quantize {onnx_file}: {e}")


# ---------------------------------------------------------------------------
# CTranslate2 export helpers
# ---------------------------------------------------------------------------

def export_nllb_to_ct2(model_dir: str, ct2_dir: str, quantization: str = "int8") -> str:
    """
    Convert merged NLLB model to CTranslate2 format.
    Tries OpusMTConverter first; falls back to HfConverter.
    """
    logger.info(f"Converting NLLB to CTranslate2: {model_dir} → {ct2_dir}")
    Path(ct2_dir).mkdir(parents=True, exist_ok=True)

    # --- Primary: CLI conversion (most reliable for NLLB) ---
    try:
        result = subprocess.run(
            [
                "ct2-opus-mt-converter",
                "--model", model_dir,
                "--output_dir", ct2_dir,
                "--quantization", quantization,
                "--force",
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            logger.info("CTranslate2 CLI conversion (ct2-opus-mt-converter) complete.")
            return ct2_dir
        logger.warning(f"CLI converter stderr: {result.stderr[:500]}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"CLI converter not available: {e}")

    # --- Fallback A: OpusMTConverter API ---
    try:
        import ctranslate2
        converter = ctranslate2.converters.OpusMTConverter(model_dir)
        converter.convert(ct2_dir, quantization=quantization, force=True)
        logger.info("CTranslate2 OpusMTConverter complete.")
        return ct2_dir
    except Exception as e:
        logger.warning(f"OpusMTConverter failed: {e}")

    # --- Fallback B: HfConverter (generic HF → CTranslate2) ---
    try:
        import ctranslate2
        converter = ctranslate2.converters.HfConverter(model_dir)
        converter.convert(ct2_dir, quantization=quantization, force=True)
        logger.info("CTranslate2 HfConverter complete.")
        return ct2_dir
    except Exception as e:
        logger.error(f"All CTranslate2 converters failed: {e}")
        raise

    return ct2_dir


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------

def push_to_hub(local_dir: str, repo_id: str, folder_in_repo: str, hf_token: str):
    """Upload a local directory to a Hub repository."""
    if not hf_token:
        logger.warning("No HF token — skipping Hub upload.")
        return

    api = HfApi()
    logger.info(f"Creating/confirming Hub repo: {repo_id}")
    api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, repo_type="model")

    logger.info(f"Uploading {local_dir} → {repo_id}/{folder_in_repo} …")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        path_in_repo=folder_in_repo,
        token=hf_token,
        commit_message=f"Phase 2 export: {folder_in_repo}",
    )
    logger.info("Upload complete.")


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def benchmark_nllb_onnx(onnx_dir: str, n_runs: int = 5) -> Optional[float]:
    """
    Run a quick NLLB ONNX inference benchmark.
    Returns average latency in milliseconds.
    """
    try:
        import onnxruntime as ort
        from transformers import NllbTokenizer

        encoder_path = str(Path(onnx_dir) / "encoder_model.onnx")
        decoder_path = str(Path(onnx_dir) / "decoder_model.onnx")

        if not (Path(encoder_path).exists() and Path(decoder_path).exists()):
            logger.warning("ONNX model files not found for NLLB benchmark.")
            return None

        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        tokenizer.src_lang = "eng_Latn"

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        enc_sess = ort.InferenceSession(encoder_path, sess_options=sess_opts,
                                        providers=["CPUExecutionProvider"])

        sample_text = "Hello, how are you today?"
        inputs = tokenizer(sample_text, return_tensors="np")

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            enc_sess.run(None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            })
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(latencies[1:])  # skip warm-up
        logger.info(f"NLLB ONNX encoder latency (avg): {avg_ms:.1f} ms")
        return avg_ms

    except Exception as e:
        logger.warning(f"NLLB ONNX benchmark failed: {e}")
        return None


def benchmark_whisper_onnx(onnx_dir: str, n_runs: int = 5) -> Optional[float]:
    """
    Run a quick Whisper ONNX encoder inference benchmark with a dummy audio input.
    Returns average latency in milliseconds.
    """
    try:
        import onnxruntime as ort

        encoder_path = str(Path(onnx_dir) / "encoder_model.onnx")
        if not Path(encoder_path).exists():
            logger.warning("Whisper ONNX encoder not found for benchmark.")
            return None

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        enc_sess = ort.InferenceSession(encoder_path, sess_options=sess_opts,
                                        providers=["CPUExecutionProvider"])

        # Whisper encoder expects [batch, 80, 3000] log-mel spectrogram
        dummy_input = np.random.randn(1, 80, 3000).astype(np.float32)
        input_name = enc_sess.get_inputs()[0].name

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            enc_sess.run(None, {input_name: dummy_input})
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(latencies[1:])
        logger.info(f"Whisper ONNX encoder latency (avg): {avg_ms:.1f} ms")
        return avg_ms

    except Exception as e:
        logger.warning(f"Whisper ONNX benchmark failed: {e}")
        return None


def benchmark_nllb_ct2(ct2_dir: str, n_runs: int = 5) -> Optional[float]:
    """
    Benchmark CTranslate2 NLLB inference latency.
    Returns average latency in milliseconds.
    """
    try:
        import ctranslate2
        from transformers import NllbTokenizer

        translator = ctranslate2.Translator(ct2_dir, device="cpu", inter_threads=2)
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        tokenizer.src_lang = "eng_Latn"

        sample = "Hello, how are you today?"
        source = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(sample, add_special_tokens=False)
        )

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            translator.translate_batch(
                [source],
                target_prefix=[["hin_Deva"]],
            )
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(latencies[1:])
        logger.info(f"NLLB CTranslate2 latency (avg): {avg_ms:.1f} ms")
        return avg_ms

    except Exception as e:
        logger.warning(f"CTranslate2 NLLB benchmark failed: {e}")
        return None


# ---------------------------------------------------------------------------
# File size reporting
# ---------------------------------------------------------------------------

def report_directory_size(label: str, directory: str) -> int:
    """Print and return total bytes in a directory."""
    p = Path(directory)
    if not p.exists():
        logger.info(f"{label}: directory does not exist ({directory})")
        return 0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    logger.info(f"{label}: {total / 1e6:.2f} MB ({total:,} bytes)")
    return total


def report_all_sizes(exports: dict):
    logger.info("\n" + "=" * 60)
    logger.info("EXPORT FILE SIZES")
    logger.info("=" * 60)
    for label, path in exports.items():
        if path:
            report_directory_size(label, path)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
        args.hf_token = os.environ["HF_TOKEN"]

    if not args.hf_username:
        args.hf_username = os.environ.get("HF_USERNAME", "")

    # Resolve Hub repo IDs
    nllb_hub_repo = args.nllb_hub_repo or (
        f"{args.hf_username}/nllb-200-600m-qlora" if args.hf_username else "nllb-200-600m-qlora"
    )
    whisper_hub_repo = args.whisper_hub_repo or (
        f"{args.hf_username}/whisper-small-indian-lora" if args.hf_username else "whisper-small-indian-lora"
    )

    base = Path(args.output_base)
    exports = {}

    # ── NLLB ─────────────────────────────────────────────────────────────────
    if args.export_nllb:
        # ONNX
        nllb_onnx_dir = str(base / "nllb_onnx")
        try:
            export_nllb_to_onnx(args.nllb_model_dir, nllb_onnx_dir)
            exports["NLLB ONNX (fp32)"] = nllb_onnx_dir

            if args.quantize_onnx:
                nllb_onnx_int8_dir = str(base / "nllb_onnx_int8")
                quantize_all_onnx_in_dir(nllb_onnx_dir, nllb_onnx_int8_dir)
                exports["NLLB ONNX (int8)"] = nllb_onnx_int8_dir

            if args.push_to_hub and args.hf_token:
                push_to_hub(nllb_onnx_dir, nllb_hub_repo, "onnx", args.hf_token)
                if args.quantize_onnx:
                    push_to_hub(nllb_onnx_int8_dir, nllb_hub_repo, "onnx_int8", args.hf_token)

        except Exception as e:
            logger.error(f"NLLB ONNX export failed: {e}")

        # CTranslate2
        if args.export_ct2:
            nllb_ct2_dir = str(base / "nllb_ct2")
            try:
                export_nllb_to_ct2(args.nllb_model_dir, nllb_ct2_dir)
                exports["NLLB CTranslate2 (int8)"] = nllb_ct2_dir

                if args.push_to_hub and args.hf_token:
                    push_to_hub(nllb_ct2_dir, nllb_hub_repo, "ct2_model", args.hf_token)

            except Exception as e:
                logger.error(f"NLLB CTranslate2 export failed: {e}")

    # ── Whisper ───────────────────────────────────────────────────────────────
    if args.export_whisper:
        whisper_onnx_dir = str(base / "whisper_onnx")
        try:
            export_whisper_to_onnx(args.whisper_model_dir, whisper_onnx_dir)
            exports["Whisper ONNX (fp32)"] = whisper_onnx_dir

            if args.quantize_onnx:
                whisper_onnx_int8_dir = str(base / "whisper_onnx_int8")
                quantize_all_onnx_in_dir(whisper_onnx_dir, whisper_onnx_int8_dir)
                exports["Whisper ONNX (int8)"] = whisper_onnx_int8_dir

            if args.push_to_hub and args.hf_token:
                push_to_hub(whisper_onnx_dir, whisper_hub_repo, "onnx", args.hf_token)
                if args.quantize_onnx:
                    push_to_hub(whisper_onnx_int8_dir, whisper_hub_repo, "onnx_int8", args.hf_token)

        except Exception as e:
            logger.error(f"Whisper ONNX export failed: {e}")

    # ── File sizes ────────────────────────────────────────────────────────────
    report_all_sizes(exports)

    # ── Benchmarks ────────────────────────────────────────────────────────────
    if args.run_benchmark:
        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE BENCHMARKS (CPU)")
        logger.info("=" * 60)

        if args.export_nllb:
            nllb_onnx_dir = str(base / "nllb_onnx")
            nllb_int8_dir = str(base / "nllb_onnx_int8")
            nllb_ct2_dir  = str(base / "nllb_ct2")

            fp32_lat = benchmark_nllb_onnx(nllb_onnx_dir)
            if args.quantize_onnx:
                # int8 files have _int8 suffix — benchmark fp32 dir still (encoder_model.onnx)
                int8_lat = benchmark_nllb_onnx(nllb_int8_dir)
                if fp32_lat and int8_lat:
                    logger.info(
                        f"NLLB ONNX speedup (int8 vs fp32): {fp32_lat / int8_lat:.2f}×"
                    )
            if args.export_ct2:
                benchmark_nllb_ct2(nllb_ct2_dir)

        if args.export_whisper:
            whisper_onnx_dir = str(base / "whisper_onnx")
            benchmark_whisper_onnx(whisper_onnx_dir)

        logger.info("=" * 60)

    logger.info("\nPhase 2 export pipeline complete ✓")


if __name__ == "__main__":
    main()
