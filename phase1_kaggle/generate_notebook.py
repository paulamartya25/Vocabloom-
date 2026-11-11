"""
Generates a fully self-contained phase1_notebook.ipynb for Kaggle.
Run this script LOCALLY to regenerate the notebook whenever the .py files change.
"""
import json, re, uuid, pathlib, sys

ROOT = pathlib.Path(__file__).parent
DS_PATH = ROOT / "dataset_builder.py"
TR_PATH = ROOT / "train_hybrid_model.py"
OUT_PATH = ROOT / "phase1_notebook.ipynb"

def cell_id():
    return uuid.uuid4().hex[:8]

def markdown(src: str):
    return {"cell_type": "markdown", "id": cell_id(), "metadata": {},
            "source": src}

def code(src: str):
    return {"cell_type": "code", "execution_count": None, "id": cell_id(),
            "metadata": {}, "outputs": [], "source": src}

def embed_script(path: pathlib.Path, dest: str) -> str:
    """Return a code cell source that writes `path` content to `dest` on disk."""
    content = path.read_text(encoding="utf-8")
    # Escape backslashes and triple-quotes safely
    content_escaped = content.replace("\\", "\\\\").replace('"""', r'\"\"\"')
    return f'''# ── Write {path.name} to Kaggle working dir ────────────────────────────
import pathlib, textwrap

_SRC = """{content_escaped}"""

dest = pathlib.Path("{dest}")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(_SRC, encoding="utf-8")
print(f"✅ Written {{len(_SRC)}} chars → {{dest}}")
'''

cells = [
    markdown("""# 🌐 Phase 1 — Multilingual Translator: Dataset Build + Hybrid Model Training

**Architecture:** XLM-RoBERTa encoder (first 6 layers frozen) + Custom 6-layer Transformer decoder built from scratch.

> **This notebook is fully self-contained** — no external dataset input is needed.
> All Python source files are written to `/kaggle/working/` by the cells below.

**Languages covered (24):**
- 🇮🇳 Indian: Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu
- 🇪🇺 European: French, German, Spanish, Italian, Portuguese, Dutch, Polish, Romanian, Swedish, Czech
- 🌏 CJK: Chinese (Simplified), Japanese, Korean
- 🌐 English (pivot)

**Corpus sources:** `Helsinki-NLP/opus-100`, `opus_books`, `Helsinki-NLP/tatoeba_mt`

**Hardware:** 2× NVIDIA T4 (16 GB each) — enable in Kaggle: *Settings → Accelerator → GPU T4 x2*

---
## Workflow
1. Install dependencies
2. Set HF secrets
3. GPU check
4. Write source scripts to disk  ← **fixes the "No such file" error**
5. Build & upload dataset
6. Train the hybrid model
7. Inspect metrics & checkpoints"""),

    markdown("## Step 1 — Install Dependencies"),
    code('''import subprocess, sys

def pip_install(packages, quiet=True):
    q = ["--quiet"] if quiet else []
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + q + packages
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  pip error:\\n{result.stderr[-2000:]}")
    else:
        print(f"✅ {' '.join(packages[:3])}{'...' if len(packages)>3 else ''}")

# Core ML
pip_install(["torch>=2.5.0", "torchvision>=0.20.0", "torchaudio>=2.5.0"])

# HuggingFace ecosystem
pip_install(["transformers>=4.45.0", "datasets>=3.0.0", "tokenizers>=0.20.0",
             "huggingface-hub>=0.25.0", "accelerate>=1.0.0", "evaluate>=0.4.3"])

# Data + metrics
pip_install(["pandas>=2.2.1", "pyarrow>=15.0.2", "sacrebleu>=2.4.0",
             "rouge-score>=0.1.2", "jiwer>=3.0.4"])

# Utilities
pip_install(["langdetect==1.0.9", "tqdm>=4.66.2", "scikit-learn>=1.4.0",
             "matplotlib>=3.8.3", "seaborn>=0.13.2", "pynvml>=11.5.0",
             "fastparquet>=2024.2.0"])

print("\\n🚀 All packages installed successfully!")'''),

    markdown("## Step 2 — Configure Secrets & Environment"),
    code('''import os

# ── Read from Kaggle Secrets ────────────────────────────────────────────────
# Add secrets in Kaggle: Notebook → Add-ons → Secrets
#   Name: HF_TOKEN    Value: hf_xxxxxxxxxxxx
#   Name: HF_USERNAME Value: your_hf_username
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    HF_TOKEN    = _s.get_secret("HF_TOKEN")
    HF_USERNAME = _s.get_secret("HF_USERNAME")
    print(f"✅ Secrets loaded from Kaggle vault — HF_USERNAME={HF_USERNAME}")
except Exception as e:
    print(f"⚠️  Kaggle secrets unavailable ({e})")
    # ── Fallback: paste manually ────────────────────────────────────────────
    HF_TOKEN    = ""   # 👈 paste your HF token here if secrets don\'t work
    HF_USERNAME = ""   # 👈 paste your HF username here

if not HF_TOKEN:
    print("❌  HF_TOKEN is empty — dataset/model push to Hub will be skipped.")
if not HF_USERNAME:
    print("❌  HF_USERNAME is empty — Hub repo names will be incomplete.")

# ── Set environment variables for child processes ───────────────────────────
os.environ["HF_TOKEN"]       = HF_TOKEN
os.environ["HF_USERNAME"]    = HF_USERNAME
os.environ["DATA_DIR"]       = "/kaggle/working/multilingual_dataset"
os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"
os.environ["OUTPUT_DIR"]     = "/kaggle/working/multilingual_dataset"

print(f"DATA_DIR       = {os.environ[\'DATA_DIR\']}")
print(f"CHECKPOINT_DIR = {os.environ[\'CHECKPOINT_DIR\']}")'''),

    markdown("## Step 3 — GPU & System Check"),
    code('''import torch, platform

print(f"Python  : {platform.python_version()}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"Device  : {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️ (enable GPU in Settings!)'}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB  ({p.multi_processor_count} SMs)")
else:
    print("\\n⚠️  No GPU found — go to Settings → Accelerator → GPU T4 x2 and re-run!")

# Disk space
import shutil
total, used, free = shutil.disk_usage("/kaggle/working")
print(f"\\nDisk /kaggle/working: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")'''),

    markdown("""## Step 4 — Write Source Scripts to Disk

> **This is the fix for the 'No such file or directory' error.**
> Both Python scripts are embedded directly in this notebook cell
> and written to `/kaggle/working/` before being executed."""),

    code(embed_script(DS_PATH, "/kaggle/working/dataset_builder.py")),
    code(embed_script(TR_PATH, "/kaggle/working/train_hybrid_model.py")),

    code('''# Verify both files exist and are readable
import pathlib
for fname in ["dataset_builder.py", "train_hybrid_model.py"]:
    p = pathlib.Path(f"/kaggle/working/{fname}")
    if p.exists():
        lines = len(p.read_text(encoding="utf-8").splitlines())
        print(f"✅  {fname:35s} {lines} lines  ({p.stat().st_size/1024:.1f} KB)")
    else:
        print(f"❌  {fname} NOT FOUND — something went wrong above!")'''),

    markdown("## Step 5 — Build the Multilingual Dataset"),
    code('''import subprocess, sys, time, os

print("=" * 60)
print("RUNNING: dataset_builder.py")
print("=" * 60)
print("This downloads ~15-50 GB of corpora from HuggingFace.")
print("Expected time: 30-90 min depending on network speed.\\n")

t0 = time.time()
result = subprocess.run(
    [sys.executable, "/kaggle/working/dataset_builder.py"],
    env=os.environ.copy(),
    text=True,
    capture_output=False,   # stream output directly to notebook
)
elapsed = time.time() - t0
print(f"\\n{'='*60}")
print(f"⏱️  Completed in {elapsed/60:.1f} min  (exit code {result.returncode})")
if result.returncode != 0:
    print("❌  dataset_builder.py FAILED — check output above.")
else:
    print("✅  Dataset build complete!")'''),

    markdown("## Step 5b — Dataset Statistics"),
    code('''import json, pathlib
import pandas as pd
import matplotlib.pyplot as plt

stats_path = pathlib.Path("/kaggle/working/multilingual_dataset/dataset_statistics.json")

if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)

    print("Dataset Statistics")
    def _n(v): return f"{v:,}" if isinstance(v, int) else str(v)
    print(f"  Total examples : {_n(stats.get('total_examples', 'N/A'))}")
    print(f"  Train split    : {_n(stats.get('train_examples', 'N/A'))}")
    print(f"  Val split      : {_n(stats.get('val_examples',   'N/A'))}")
    print(f"  Test split     : {_n(stats.get('test_examples',  'N/A'))}")
    print(f"  Language pairs : {stats.get('language_pairs', 'N/A')}")

    pairs = stats.get("pair_counts", {})
    if pairs:
        df_pairs = pd.Series(pairs).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 5))
        df_pairs.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Rows per Language Pair", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count"); ax.set_xlabel("Language Pair")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

    domains = stats.get("domain_distribution", {})
    if domains:
        df_dom = pd.Series(domains)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        df_dom.plot(kind="pie", ax=ax2, autopct="%1.1f%%")
        ax2.set_title("Domain Distribution"); ax2.set_ylabel("")
        plt.tight_layout(); plt.show()
else:
    print(f"⚠️  Statistics file not found at {stats_path}")
    print("    This is expected if dataset_builder.py hasn\'t finished yet.")
    # List what IS in the directory
    work_dir = pathlib.Path("/kaggle/working/multilingual_dataset")
    if work_dir.exists():
        files = list(work_dir.iterdir())
        print(f"\\n  Files in {work_dir}: {[f.name for f in files[:20]]}")
    else:
        print(f"\\n  Directory {work_dir} does not exist yet.")'''),

    markdown("## Step 6 — Train the Hybrid Model"),
    code('''import subprocess, sys, time, os

print("=" * 60)
print("RUNNING: train_hybrid_model.py")
print("=" * 60)
print("Training custom-hybrid-v1 (XLM-RoBERTa + 6-layer decoder).")
print("Expected time: 4-8 hours on 2×T4 GPUs.\\n")

t0 = time.time()
result = subprocess.run(
    [sys.executable, "/kaggle/working/train_hybrid_model.py"],
    env=os.environ.copy(),
    text=True,
    capture_output=False,
)
elapsed = time.time() - t0
print(f"\\n{'='*60}")
print(f"⏱️  Training finished in {elapsed/3600:.2f} hours  (exit code {result.returncode})")
if result.returncode != 0:
    print("❌  Training FAILED — check output above.")
else:
    print("✅  Training complete! Check Hub for pushed checkpoints.")'''),

    markdown("## Step 7 — Training Metrics Visualisation"),
    code('''import json, pathlib
import pandas as pd
import matplotlib.pyplot as plt

log_path = pathlib.Path("/kaggle/working/checkpoints/training_log.jsonl")

if log_path.exists():
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    df_log = pd.DataFrame(records)
    print(f"Total log entries: {len(df_log)}")
    print(df_log.tail(5).to_string(index=False))

    train_df = df_log[df_log.get("type", pd.NA) != "eval"] if "type" in df_log.columns else df_log
    eval_df  = df_log[df_log.get("type", "") == "eval"] if "type" in df_log.columns else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d0d14")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#9898b8"); ax.yaxis.label.set_color("#9898b8")
        ax.xaxis.label.set_color("#9898b8"); ax.title.set_color("#f0f0ff")

    if not train_df.empty and "train_loss" in train_df.columns:
        axes[0].plot(train_df["step"], train_df["train_loss"], color="#6c63ff", lw=1.5)
        axes[0].set_title("Training Loss"); axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.2)

    if not eval_df.empty and "val_loss" in eval_df.columns:
        axes[1].plot(eval_df["step"], eval_df["val_loss"], color="#ef4444", marker="o", lw=1.5, ms=4)
        axes[1].set_title("Validation Loss"); axes[1].set_xlabel("Step"); axes[1].set_ylabel("Val Loss")
        axes[1].grid(alpha=0.2)

    if not eval_df.empty and "bleu" in eval_df.columns:
        axes[2].plot(eval_df["step"], eval_df["bleu"], color="#10b981", marker="s", lw=1.5, ms=4)
        axes[2].set_title("BLEU Score (Val)"); axes[2].set_xlabel("Step"); axes[2].set_ylabel("BLEU")
        axes[2].grid(alpha=0.2)

    plt.suptitle("custom-hybrid-v1 Training Metrics", fontsize=14, fontweight="bold", color="#f0f0ff")
    plt.tight_layout()
    plt.savefig("/kaggle/working/training_metrics.png", dpi=150, facecolor="#0d0d14")
    plt.show()
    print("\\n📊 Plot saved → /kaggle/working/training_metrics.png")
else:
    print(f"⚠️  Log file not found at {log_path}")
    print("    This is expected if training hasn\'t run yet.")
    checkpoints = list(pathlib.Path("/kaggle/working/checkpoints").glob("*")) \\
        if pathlib.Path("/kaggle/working/checkpoints").exists() else []
    print(f"    Checkpoint dir contents: {[p.name for p in checkpoints[:10]]}")'''),

    markdown("## Step 8 — List Saved Checkpoints & Hub Links"),
    code('''import pathlib, os

ckpt_dir = pathlib.Path("/kaggle/working/checkpoints")
HF_USERNAME = os.environ.get("HF_USERNAME", "")

if ckpt_dir.exists():
    checkpoints = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s):\\n")
        for c in checkpoints:
            size_mb = c.stat().st_size / 1e6
            print(f"  ✅  {c.name:<40s}  {size_mb:.1f} MB")
    else:
        # Look for any checkpoint files
        all_files = list(ckpt_dir.iterdir())
        print(f"No checkpoint_step_*.pt files. Contents of {ckpt_dir}:")
        for f in sorted(all_files)[:20]:
            print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)" if f.is_file() else f"  {f.name}/")
else:
    print(f"⚠️  Checkpoint directory not found: {ckpt_dir}")
    print("    Training may not have run yet.")

print()
print("=" * 55)
print("📋 Phase 1 Deliverables Summary")
print("=" * 55)
print(f"  Dataset (Parquet)  : /kaggle/working/multilingual_dataset/")
print(f"  Statistics JSON    : /kaggle/working/multilingual_dataset/dataset_statistics.json")
print(f"  Checkpoints        : /kaggle/working/checkpoints/")
print(f"  Training log       : /kaggle/working/checkpoints/training_log.jsonl")
print(f"  Metrics plot       : /kaggle/working/training_metrics.png")
if HF_USERNAME:
    print(f"\\n  HF Dataset repo    : https://huggingface.co/datasets/{HF_USERNAME}/multilingual-translation-data")
    print(f"  HF Model repo      : https://huggingface.co/{HF_USERNAME}/custom-hybrid-v1")
else:
    print("\\n  ⚠️  Set HF_USERNAME to see Hub repo links")
print("\\n✅  Phase 1 complete — proceed to Phase 2 notebook!")'''),

    markdown("""---
## ✅ Phase 1 Complete

**Next:** Upload `phase2_notebook.ipynb` to a new Kaggle notebook for NLLB QLoRA + Whisper LoRA fine-tuning."""),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "cells": cells,
}

OUT_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[OK] Written: {OUT_PATH}")
print(f"     Cells : {len(cells)}")
print(f"     Size  : {OUT_PATH.stat().st_size/1024:.1f} KB")

