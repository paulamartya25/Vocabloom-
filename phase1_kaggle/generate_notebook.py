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

    markdown("## Step 5c — Classification Metrics: Precision, Recall, F1, ROC-AUC & Confusion Matrix"),
    code('''# ══════════════════════════════════════════════════════════════════
# Classification Metrics for Language Detection (24-class classifier)
# Metrics: Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR Curves
# ══════════════════════════════════════════════════════════════════
import subprocess, sys

# Install extra deps if needed
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "scikit-learn>=1.4.0", "langdetect==1.0.9"], check=False)

import json, pathlib, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")

from langdetect import detect, LangDetectException
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#0d0d14", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#2a2a4a", "text.color": "#f0f0ff",
    "axes.labelcolor": "#f0f0ff", "xtick.color": "#9898b8",
    "ytick.color": "#9898b8", "grid.color": "#2a2a4a",
})

# ── 24-language test corpus ───────────────────────────────────────────────────
LANGUAGES = {
    "en":"English","hi":"Hindi","bn":"Bengali","ta":"Tamil","te":"Telugu",
    "mr":"Marathi","gu":"Gujarati","kn":"Kannada","ml":"Malayalam","pa":"Punjabi",
    "ur":"Urdu","fr":"French","de":"German","es":"Spanish","it":"Italian",
    "pt":"Portuguese","nl":"Dutch","pl":"Polish","ro":"Romanian","sv":"Swedish",
    "cs":"Czech","zh":"Chinese","ja":"Japanese","ko":"Korean",
}

CORPUS = [
    ("The quick brown fox jumps over the lazy dog.", "en"),
    ("Artificial intelligence is transforming the world.", "en"),
    ("She walked into the library and picked up a book.", "en"),
    ("The government announced new economic policies today.", "en"),
    ("नमस्ते, आप कैसे हैं? मुझे आशा है आप ठीक हैं।", "hi"),
    ("भारत एक विविधताओं से भरा देश है।", "hi"),
    ("मैं मुंबई में रहता हूं और वहां काम करता हूं।", "hi"),
    ("আমি বাংলা বলি এবং বাংলাদেশে থাকি।", "bn"),
    ("আজকের আবহাওয়া খুব সুন্দর।", "bn"),
    ("নান் தமிழ் பேசுகிறேன் மற்றும் சென்னையில் வசிக்கிறேன்.", "ta"),
    ("நேடு வானிலை மிகவும் அழகாக உள்ளது.", "ta"),
    ("నేను తెలుగు మాట్లాడతాను మరియు హైదరాబాద్‌లో నివసిస్తున్నాను.", "te"),
    ("మీ మరాఠీ బోలతో ఆణి పుణ్యాత్ రాహతో.", "mr"),
    ("हुं ગુজরાতી બોलું છું અને અમદાવાદમાં રહું છું.", "gu"),
    ("ನಾನು ಕನ್ನಡ ಮಾತನಾಡುತ್ತೇನೆ ಮತ್ತು ಬೆಂಗಳೂರಿನಲ್ಲಿ ವಾಸಿಸುತ್ತೇನೆ.", "kn"),
    ("ഞാൻ മലയാളം സംസാരിക്കുന്നു, കേരളത്തിൽ താമസിക്കുന്നു.", "ml"),
    ("ਮੈਂ ਪੰਜਾਬੀ ਬੋਲਦਾ ਹਾਂ ਅਤੇ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਰਹਿੰਦਾ ਹਾਂ।", "pa"),
    ("میں اردو بولتا ہوں اور لاہور میں رہتا ہوں۔", "ur"),
    ("Bonjour, comment allez-vous? J\'espère que vous allez bien.", "fr"),
    ("La France est un pays magnifique avec une riche histoire.", "fr"),
    ("L\'intelligence artificielle transforme notre monde.", "fr"),
    ("Guten Tag, wie geht es Ihnen? Ich hoffe, es geht Ihnen gut.", "de"),
    ("Künstliche Intelligenz verändert unsere Welt.", "de"),
    ("Hola, ¿cómo estás? Espero que estés bien hoy.", "es"),
    ("La inteligencia artificial está transformando el mundo.", "es"),
    ("Ciao, come stai? Spero che tu stia bene oggi.", "it"),
    ("L\'intelligenza artificiale sta trasformando il mondo.", "it"),
    ("Olá, como você está? Espero que esteja bem.", "pt"),
    ("Hallo, hoe gaat het met u? Kunstmatige intelligentie.", "nl"),
    ("Cześć, jak się masz? Sztuczna inteligencja zmienia świat.", "pl"),
    ("Bună ziua, cum vă mai simțiți? România este frumoasă.", "ro"),
    ("Hej, hur mår du? Artificiell intelligens förändrar världen.", "sv"),
    ("Dobrý den, jak se máte? Česká republika je krásná.", "cs"),
    ("你好，你好吗？人工智能正在改变我们的世界。", "zh"),
    ("中国是一个拥有悠久历史的美丽国家。", "zh"),
    ("こんにちは、お元気ですか？人工知能は世界を変えています。", "ja"),
    ("日本は豊かな歴史を持つ美しい国です。", "ja"),
    ("안녕하세요, 어떻게 지내세요? 인공지능이 세계를 바꾸고 있습니다.", "ko"),
    ("한국은 풍부한 역사를 가진 아름다운 나라입니다.", "ko"),
]

# ── Run detection ─────────────────────────────────────────────────────────────
print("Running language detection on", len(CORPUS), "samples...")
lang_set = set(LANGUAGES.keys())
y_true, y_pred = [], []
for text, true_lang in CORPUS:
    try:
        pred = detect(text)
        if pred.startswith("zh"): pred = "zh"
        if pred not in lang_set: pred = pred[:2] if pred[:2] in lang_set else "unknown"
    except LangDetectException:
        pred = "unknown"
    y_true.append(true_lang)
    y_pred.append(pred)

labels = sorted(set(y_true + y_pred) & lang_set)
print(f"Labels detected: {len(labels)} languages")

# ── Core metrics ──────────────────────────────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
mp   = precision_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
mr   = recall_score   (y_true, y_pred, average="macro",    labels=labels, zero_division=0)
mf1  = f1_score       (y_true, y_pred, average="macro",    labels=labels, zero_division=0)
wp   = precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
wr   = recall_score   (y_true, y_pred, average="weighted", labels=labels, zero_division=0)
wf1  = f1_score       (y_true, y_pred, average="weighted", labels=labels, zero_division=0)

report = classification_report(y_true, y_pred, labels=labels,
                                output_dict=True, zero_division=0)

print(f"""
╔══════════════════════════════════════════════╗
║   LANGUAGE DETECTION — CLASSIFICATION REPORT ║
╠══════════════════════════════════════════════╣
║  Accuracy          : {acc:.4f}               ║
║  Macro  Precision  : {mp:.4f}               ║
║  Macro  Recall     : {mr:.4f}               ║
║  Macro  F1-Score   : {mf1:.4f}               ║
║  Weighted Precision: {wp:.4f}               ║
║  Weighted Recall   : {wr:.4f}               ║
║  Weighted F1-Score : {wf1:.4f}               ║
╚══════════════════════════════════════════════╝
""")

# Per-language table
print(f"{'Lang':<6} {'Name':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
print("-" * 58)
for code in labels:
    if code not in report: continue
    m = report[code]
    name = LANGUAGES.get(code, code)
    print(f"{code:<6} {name:<14} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1-score']:>8.4f} {int(m['support']):>8}")

# Save JSON
out = pathlib.Path("/kaggle/working/metrics/classification")
out.mkdir(parents=True, exist_ok=True)
with open(out / "lang_detection_report.json", "w") as f:
    safe = {k:v for k,v in report.items()}
    json.dump({"accuracy":acc,"macro_f1":mf1,"weighted_f1":wf1,"per_class":safe}, f, indent=2)
print(f"\\nSaved JSON → {out / 'lang_detection_report.json'}")'''),

    code('''# ── PLOT 1: Confusion Matrix ─────────────────────────────────────────────────
import pathlib, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

out = pathlib.Path("/kaggle/working/metrics/classification")
n   = len(labels)
cm  = confusion_matrix(y_true, y_pred, labels=labels)
with np.errstate(divide="ignore", invalid="ignore"):
    cm_norm = np.where(cm.sum(1,keepdims=True)==0, 0, cm/cm.sum(1,keepdims=True))

fig, ax = plt.subplots(figsize=(max(10,n*0.65), max(8,n*0.55)))
fig.patch.set_facecolor("#0d0d14"); ax.set_facecolor("#0d0d14")
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

names = [f"{c}\\n{LANGUAGES.get(c,c)}" for c in labels]
ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=7, color="#9898b8")
ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7, color="#9898b8")
ax.set_xlabel("Predicted Language", color="#f0f0ff", fontsize=11)
ax.set_ylabel("True Language",      color="#f0f0ff", fontsize=11)
ax.set_title("Language Detection — Confusion Matrix (Row-normalised Recall)",
             color="#f0f0ff", fontsize=12, fontweight="bold")

for i in range(n):
    for j in range(n):
        v = cm[i,j]
        if v > 0:
            col = "white" if cm_norm[i,j] > 0.5 else "#aaaacc"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=max(5, 8-n//6), color=col, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Recall (row-normalised)", color="#9898b8", fontsize=9)
cbar.ax.tick_params(colors="#9898b8")

plt.tight_layout()
plt.savefig(out/"lang_confusion_matrix.png", dpi=150, facecolor="#0d0d14", bbox_inches="tight")
plt.show()
print("Saved → lang_confusion_matrix.png")'''),

    code('''# ── PLOT 2: Per-Language Precision / Recall / F1 Bar Chart ──────────────────
import numpy as np
import matplotlib.pyplot as plt
import pathlib

out = pathlib.Path("/kaggle/working/metrics/classification")
langs_present = [l for l in labels if l in LANGUAGES]
precs = [report[l]["precision"] for l in langs_present]
recs  = [report[l]["recall"]    for l in langs_present]
f1s   = [report[l]["f1-score"]  for l in langs_present]
names = [f"{l}\\n{LANGUAGES[l]}" for l in langs_present]

x = np.arange(len(langs_present)); w = 0.26
fig, ax = plt.subplots(figsize=(max(12,len(langs_present)*0.75), 5))
fig.patch.set_facecolor("#0d0d14"); ax.set_facecolor("#1a1a2e")
ax.bar(x-w, precs, w, label="Precision", color="#6c63ff", edgecolor="none")
ax.bar(x,   recs,  w, label="Recall",    color="#a855f7", edgecolor="none")
ax.bar(x+w, f1s,   w, label="F1-Score",  color="#06b6d4", edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, color="#9898b8")
ax.set_ylim(0, 1.15); ax.set_ylabel("Score", color="#f0f0ff")
ax.set_title("Per-Language Precision / Recall / F1-Score",
             color="#f0f0ff", fontsize=13, fontweight="bold")
ax.tick_params(colors="#9898b8"); ax.legend(framealpha=0.3, labelcolor="#f0f0ff")
ax.grid(axis="y", alpha=0.2)
for xi, f1 in zip(x+w, f1s):
    if f1 > 0.05:
        ax.text(xi, f1+0.02, f"{f1:.2f}", ha="center", va="bottom",
                fontsize=7, color="#f0f0ff")
plt.tight_layout()
plt.savefig(out/"lang_per_class_metrics.png", dpi=150, facecolor="#0d0d14", bbox_inches="tight")
plt.show()
print("Saved → lang_per_class_metrics.png")'''),

    code('''# ── PLOT 3: ROC-AUC Curves (One-vs-Rest per language) ────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

out = pathlib.Path("/kaggle/working/metrics/classification")

# Build soft-probability matrix from hard predictions
def build_proba(y_t, y_p, lbls):
    idx = {l:i for i,l in enumerate(lbls)}
    P = np.full((len(y_t), len(lbls)), 0.01/(max(len(lbls)-1,1)))
    for i,p in enumerate(y_p):
        if p in idx: P[i,idx[p]] = 0.99
    return P

proba  = build_proba(y_true, y_pred, labels)
y_bin  = label_binarize(y_true, classes=labels)
COLORS = plt.cm.tab20(np.linspace(0,1,len(labels)))

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#0d0d14"); ax.set_facecolor("#1a1a2e")

roc_aucs = {}
for i, lang in enumerate(labels):
    if y_bin.shape[1] <= i: continue
    try:
        fpr, tpr, _ = roc_curve(y_bin[:,i], proba[:,i])
        roc_auc = auc(fpr, tpr)
        roc_aucs[lang] = round(roc_auc, 4)
        ax.plot(fpr, tpr, color=COLORS[i], lw=1.8,
                label=f"{lang} ({LANGUAGES.get(lang,'')}): AUC={roc_auc:.3f}")
    except Exception: pass

ax.plot([0,1],[0,1],"w--",lw=0.8,alpha=0.4,label="Random (AUC=0.50)")
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("False Positive Rate", color="#f0f0ff", fontsize=11)
ax.set_ylabel("True Positive Rate",  color="#f0f0ff", fontsize=11)
ax.set_title("ROC Curves — Language Detection (One-vs-Rest)",
             color="#f0f0ff", fontsize=13, fontweight="bold")
ax.tick_params(colors="#9898b8"); ax.grid(alpha=0.15)
ax.legend(loc="lower right", fontsize=7, framealpha=0.3, labelcolor="#f0f0ff", ncol=2)

mean_auc = np.mean(list(roc_aucs.values())) if roc_aucs else 0
ax.text(0.55, 0.08, f"Mean AUC = {mean_auc:.4f}", transform=ax.transAxes,
        color="#10b981", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(out/"lang_roc_curves.png", dpi=150, facecolor="#0d0d14", bbox_inches="tight")
plt.show()
print(f"Mean ROC-AUC across all languages: {mean_auc:.4f}")
print("Saved → lang_roc_curves.png")'''),

    code('''# ── PLOT 4: Precision-Recall Curves ──────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics import precision_recall_curve, average_precision_score

out = pathlib.Path("/kaggle/working/metrics/classification")
COLORS = plt.cm.tab20(np.linspace(0,1,len(labels)))

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("#0d0d14"); ax.set_facecolor("#1a1a2e")

ap_scores = {}
for i, lang in enumerate(labels):
    if y_bin.shape[1] <= i: continue
    try:
        prec, rec, _ = precision_recall_curve(y_bin[:,i], proba[:,i])
        ap = average_precision_score(y_bin[:,i], proba[:,i])
        ap_scores[lang] = round(ap, 4)
        ax.plot(rec, prec, color=COLORS[i], lw=1.8,
                label=f"{lang} ({LANGUAGES.get(lang,'')}): AP={ap:.3f}")
    except Exception: pass

ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
ax.set_xlabel("Recall",    color="#f0f0ff", fontsize=11)
ax.set_ylabel("Precision", color="#f0f0ff", fontsize=11)
ax.set_title("Precision-Recall Curves — Language Detection (One-vs-Rest)",
             color="#f0f0ff", fontsize=13, fontweight="bold")
ax.tick_params(colors="#9898b8"); ax.grid(alpha=0.15)
ax.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="#f0f0ff", ncol=2)
plt.tight_layout()
plt.savefig(out/"lang_pr_curves.png", dpi=150, facecolor="#0d0d14", bbox_inches="tight")
plt.show()

mean_ap = np.mean(list(ap_scores.values())) if ap_scores else 0
print(f"Mean Average Precision: {mean_ap:.4f}")
print("Saved → lang_pr_curves.png")
print("\\nAll classification metric charts saved to /kaggle/working/metrics/classification/")
print("Files:", [f.name for f in sorted(pathlib.Path("/kaggle/working/metrics/classification").iterdir())])'''),

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

