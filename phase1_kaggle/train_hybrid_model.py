"""
Phase 1 — Train Hybrid Model: Custom XLM-RoBERTa Encoder + 6-Layer Scratch Decoder
====================================================================================
Model name: custom-hybrid-v1

Architecture
------------
  Encoder : FacebookAI/xlm-roberta-base  (first 6 layers frozen, last 6 fine-tuned)
  Decoder : Custom 6-layer Transformer built from scratch in PyTorch
              d_model=768  nhead=12  dim_feedforward=3072  dropout=0.1
              Language embedding (50 languages, learned)
              Sinusoidal positional encoding
              Output projection → vocab_size

Training
--------
  Mixed precision (fp16)        Gradient checkpointing
  AdamW + cosine LR schedule    Label-smoothing cross-entropy (0.1)
  Gradient accumulation 4 steps Batch size 32 / GPU
  Checkpoint every 1000 steps   Hub push every 5000 steps
  Early stopping (patience=3)   BLEU evaluation on val set

Usage:
    python train_hybrid_model.py

Environment variables:
    HF_TOKEN       — HuggingFace write token
    HF_USERNAME    — HuggingFace username
    DATA_DIR       — Parquet dataset dir (default: /kaggle/working/multilingual_dataset)
    CHECKPOINT_DIR — Where to save checkpoints (default: /kaggle/working/checkpoints)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset as TorchDataset

import pandas as pd
import pyarrow.parquet as pq
import sacrebleu
from tqdm import tqdm

from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaModel,
    get_cosine_schedule_with_warmup,
)
from huggingface_hub import HfApi, login

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train_hybrid")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # Paths
    DATA_DIR        = Path(os.environ.get("DATA_DIR",       "/kaggle/working/multilingual_dataset"))
    CHECKPOINT_DIR  = Path(os.environ.get("CHECKPOINT_DIR", "/kaggle/working/checkpoints"))
    HF_TOKEN        = os.environ.get("HF_TOKEN",    "")
    HF_USERNAME     = os.environ.get("HF_USERNAME", "")
    HF_REPO_ID      = f"{HF_USERNAME}/custom-hybrid-v1" if os.environ.get("HF_USERNAME") else ""

    # Model
    ENCODER_MODEL   = "FacebookAI/xlm-roberta-base"
    D_MODEL         = 768
    NHEAD           = 12
    NUM_DEC_LAYERS  = 6
    DIM_FEEDFORWARD = 3072
    DROPOUT         = 0.1
    MAX_SEQ_LEN     = 128   # tokens (keep GPU memory in check on 2×T4)
    NUM_LANGUAGES   = 50    # language embedding slots

    # Training
    BATCH_SIZE          = 32
    GRAD_ACCUM_STEPS    = 4
    LR                  = 3e-4
    ENCODER_LR          = 5e-5   # lower LR for fine-tuned encoder layers
    WARMUP_STEPS        = 1000
    MAX_STEPS           = 100_000
    LABEL_SMOOTHING     = 0.1
    EVAL_EVERY          = 1000
    SAVE_EVERY          = 1000
    HUB_PUSH_EVERY      = 5000
    EARLY_STOP_PATIENCE = 3
    NUM_WORKERS         = 2
    SEED                = 42

    # Encoder layer freezing
    FREEZE_ENCODER_LAYERS = 6   # freeze first N layers (0-indexed)

    LOG_FILE = Path(os.environ.get("CHECKPOINT_DIR", "/kaggle/working/checkpoints")) / "training_log.jsonl"


CFG = Config()


# ── Language-code → index mapping ─────────────────────────────────────────────
LANG_CODES = [
    "en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur",
    "fr", "de", "es", "it", "pt", "nl", "pl", "ro", "sv", "cs",
    "zh", "ja", "ko",
    # padding slots (allow room for more languages)
    *[f"xx{i}" for i in range(26)],
]
LANG2IDX: Dict[str, int] = {lang: idx for idx, lang in enumerate(LANG_CODES)}


def lang_to_idx(lang: str) -> int:
    return LANG2IDX.get(lang, len(LANG_CODES) - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Positional Encoding (sinusoidal)
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Multilingual Decoder (6 layers, built from scratch)
# ═══════════════════════════════════════════════════════════════════════════════

class MultilingualDecoder(nn.Module):
    """
    6-layer Transformer decoder that cross-attends to XLM-RoBERTa encoder
    hidden states.  A learned language embedding is summed to every decoder
    input token so the model knows which target language to generate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int              = CFG.D_MODEL,
        nhead: int                = CFG.NHEAD,
        num_layers: int           = CFG.NUM_DEC_LAYERS,
        dim_feedforward: int      = CFG.DIM_FEEDFORWARD,
        dropout: float            = CFG.DROPOUT,
        num_languages: int        = CFG.NUM_LANGUAGES,
        max_seq_len: int          = 1024,
    ):
        super().__init__()
        self.d_model = d_model

        # Token + language + position embeddings
        self.token_embedding    = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.language_embedding = nn.Embedding(num_languages, d_model)
        self.pos_encoding       = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # 6× TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-LN for training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers     = num_layers,
            norm           = nn.LayerNorm(d_model),
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between token embedding and output projection
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.language_embedding.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias if self.output_projection.bias is not None else
                       torch.zeros(1))   # no-op guard

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for auto-regressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        tgt_ids:          torch.Tensor,     # (B, T)
        encoder_out:      torch.Tensor,     # (B, S, d_model)
        tgt_lang_ids:     torch.Tensor,     # (B,)
        encoder_pad_mask: Optional[torch.Tensor] = None,  # (B, S) True = padding
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:
        B, T = tgt_ids.shape
        device = tgt_ids.device

        # Build target embeddings
        tok_emb  = self.token_embedding(tgt_ids)             # (B, T, d)
        lang_emb = self.language_embedding(tgt_lang_ids)     # (B, d)
        lang_emb = lang_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, d)

        x = self.pos_encoding(tok_emb + lang_emb)           # (B, T, d)

        # Causal mask
        causal_mask = self._make_causal_mask(T, device)      # (T, T)

        # Transformer decoder
        out = self.transformer_decoder(
            tgt                  = x,
            memory               = encoder_out,
            tgt_mask             = causal_mask,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_key_padding_mask = encoder_pad_mask,
        )   # (B, T, d)

        logits = self.output_projection(out)                 # (B, T, vocab)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Full Hybrid Model
# ═══════════════════════════════════════════════════════════════════════════════

class CustomHybridTranslator(nn.Module):
    """
    XLM-RoBERTa encoder (first 6 layers frozen) + Custom 6-layer decoder.
    """

    def __init__(self, tokenizer: XLMRobertaTokenizerFast):
        super().__init__()
        self.tokenizer = tokenizer
        vocab_size = len(tokenizer)

        # ── Encoder ───────────────────────────────────────────────────────────
        log.info(f"Loading encoder: {CFG.ENCODER_MODEL}")
        self.encoder: XLMRobertaModel = XLMRobertaModel.from_pretrained(
            CFG.ENCODER_MODEL,
            add_pooling_layer=False,
        )

        # Enable gradient checkpointing on encoder to save VRAM
        self.encoder.gradient_checkpointing_enable()

        # Freeze first N encoder layers
        self._freeze_encoder_layers(CFG.FREEZE_ENCODER_LAYERS)

        # ── Decoder ───────────────────────────────────────────────────────────
        log.info(f"Building decoder: d_model={CFG.D_MODEL} layers={CFG.NUM_DEC_LAYERS}")
        self.decoder = MultilingualDecoder(vocab_size=vocab_size)

    def _freeze_encoder_layers(self, n: int) -> None:
        # Always freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        # Freeze first n transformer layers
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False
        frozen = sum(1 for p in self.encoder.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.encoder.parameters())
        log.info(f"Encoder: {frozen}/{total} parameter groups frozen (first {n} layers + embeddings)")

    def encode(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state   # (B, S, d_model)

    def forward(
        self,
        src_input_ids:   torch.Tensor,   # (B, S)
        src_attn_mask:   torch.Tensor,   # (B, S)
        tgt_input_ids:   torch.Tensor,   # (B, T)
        tgt_lang_ids:    torch.Tensor,   # (B,)
        tgt_attn_mask:   Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:
        encoder_out = self.encode(src_input_ids, src_attn_mask)
        # Encoder pad mask: True where padding
        encoder_pad_mask = (src_attn_mask == 0)
        # Decoder pad mask: True where padding
        tgt_pad_mask = (tgt_attn_mask == 0) if tgt_attn_mask is not None else None

        logits = self.decoder(
            tgt_ids              = tgt_input_ids,
            encoder_out          = encoder_out,
            tgt_lang_ids         = tgt_lang_ids,
            encoder_pad_mask     = encoder_pad_mask,
            tgt_key_padding_mask = tgt_pad_mask,
        )
        return logits   # (B, T, vocab)

    @torch.no_grad()
    def generate(
        self,
        src_input_ids: torch.Tensor,
        src_attn_mask: torch.Tensor,
        tgt_lang_id:   int,
        max_new_tokens: int = 128,
        bos_token_id:   int = 0,
        eos_token_id:   int = 2,
    ) -> List[List[int]]:
        """Greedy decoding for BLEU evaluation."""
        device = src_input_ids.device
        B = src_input_ids.size(0)
        encoder_out = self.encode(src_input_ids, src_attn_mask)
        encoder_pad_mask = (src_attn_mask == 0)

        lang_ids = torch.full((B,), tgt_lang_id, dtype=torch.long, device=device)
        tgt = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        sequences = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            logits = self.decoder(
                tgt_ids          = tgt,
                encoder_out      = encoder_out,
                tgt_lang_ids     = lang_ids,
                encoder_pad_mask = encoder_pad_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1)   # (B,)
            for i in range(B):
                if not finished[i]:
                    sequences[i].append(next_token[i].item())
                    if next_token[i].item() == eos_token_id:
                        finished[i] = True
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        return sequences


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class TranslationDataset(TorchDataset):
    """
    Reads sharded Parquet files from a split directory.
    Tokenises on-the-fly (CPU) to keep GPU busy.
    """

    def __init__(
        self,
        split_dir:   Path,
        tokenizer:   XLMRobertaTokenizerFast,
        max_seq_len: int = CFG.MAX_SEQ_LEN,
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.rows: List[Dict] = []

        shards = sorted(split_dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards in {split_dir}")

        log.info(f"Loading {len(shards)} shards from {split_dir} …")
        for shard in tqdm(shards, desc=f"load {split_dir.name}"):
            tbl = pq.read_table(shard, columns=["source_text", "target_text", "source_lang", "target_lang"])
            df  = tbl.to_pandas()
            for _, row in df.iterrows():
                self.rows.append({
                    "src":      row["source_text"],
                    "tgt":      row["target_text"],
                    "src_lang": row["source_lang"],
                    "tgt_lang": row["target_lang"],
                })

        log.info(f"  → {len(self.rows):,} examples loaded")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.rows[idx]

        src_enc = self.tokenizer(
            item["src"],
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            item["tgt"],
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        src_ids  = src_enc["input_ids"].squeeze(0)        # (S,)
        src_mask = src_enc["attention_mask"].squeeze(0)   # (S,)
        tgt_ids  = tgt_enc["input_ids"].squeeze(0)        # (T,)
        tgt_mask = tgt_enc["attention_mask"].squeeze(0)   # (T,)

        # Decoder input: shift right (prepend BOS, drop last token)
        dec_input = torch.cat([tgt_ids.new_tensor([self.tokenizer.bos_token_id]),
                               tgt_ids[:-1]])
        dec_mask  = torch.cat([tgt_mask.new_tensor([1]), tgt_mask[:-1]])

        return {
            "src_input_ids":  src_ids,
            "src_attn_mask":  src_mask,
            "tgt_input_ids":  dec_input,       # decoder input
            "tgt_attn_mask":  dec_mask,
            "labels":         tgt_ids,         # ground truth (shifted)
            "tgt_lang_id":    torch.tensor(lang_to_idx(item["tgt_lang"]), dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 1):
        super().__init__()
        self.smoothing     = smoothing
        self.ignore_index  = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, T, V)   targets: (B, T)
        B, T, V = logits.shape
        logits  = logits.reshape(-1, V)
        targets = targets.reshape(-1)

        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss  = -log_probs.gather(dim=-1, index=targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        # Mask padding
        mask = targets != self.ignore_index
        loss = loss[mask]
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# BLEU evaluation helper
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_bleu(
    model:     CustomHybridTranslator,
    loader:    DataLoader,
    tokenizer: XLMRobertaTokenizerFast,
    device:    torch.device,
    max_batches: int = 50,
) -> float:
    model.eval()
    hypotheses: List[str] = []
    references: List[str] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        src_ids  = batch["src_input_ids"].to(device)
        src_mask = batch["src_attn_mask"].to(device)
        labels   = batch["labels"]                         # (B, T) on CPU
        tgt_lang = batch["tgt_lang_id"][0].item()         # use first item (batch may be mixed)

        seqs = model.generate(
            src_input_ids  = src_ids,
            src_attn_mask  = src_mask,
            tgt_lang_id    = tgt_lang,
            max_new_tokens = 64,
            bos_token_id   = tokenizer.bos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

        for i, seq in enumerate(seqs):
            hyp = tokenizer.decode(seq, skip_special_tokens=True)
            ref_ids = labels[i].tolist()
            ref = tokenizer.decode(ref_ids, skip_special_tokens=True)
            hypotheses.append(hyp)
            references.append(ref)

    if not hypotheses:
        return 0.0

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:     CustomHybridTranslator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    step:      int,
    val_loss:  float,
    ckpt_dir:  Path,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_step_{step:07d}.pt"
    # If model is wrapped in DataParallel
    state_dict = (model.module if hasattr(model, "module") else model).state_dict()
    torch.save(
        {
            "step":               step,
            "best_val_loss":      val_loss,
            "model_state_dict":   state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":  scaler.state_dict(),
        },
        ckpt_path,
    )
    log.info(f"Checkpoint saved → {ckpt_path}")
    # Keep only the 2 most recent checkpoints to save disk space
    old_ckpts = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))[:-2]
    for old in old_ckpts:
        try:
            old.unlink()
            log.info(f"Deleted old checkpoint: {old.name}")
        except Exception:
            pass
    return ckpt_path


def push_checkpoint_to_hub(ckpt_path: Path, repo_id: str, token: str) -> None:
    if not token or not repo_id:
        log.warning("HF_TOKEN / HF_USERNAME not set — skipping hub push.")
        return
    login(token=token)
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj = str(ckpt_path),
            path_in_repo    = ckpt_path.name,
            repo_id         = repo_id,
            repo_type       = "model",
            token           = token,
        )
        log.info(f"Checkpoint pushed to hub: {repo_id}/{ckpt_path.name}")
    except Exception as exc:
        log.warning(f"Hub push failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training log
# ═══════════════════════════════════════════════════════════════════════════════

def log_metrics(metrics: Dict, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def build_optimizer_and_scheduler(model: nn.Module, total_steps: int):
    # Separate param groups: encoder (low LR) vs decoder (higher LR)
    base = model.module if hasattr(model, "module") else model
    encoder_params = [p for p in base.encoder.parameters() if p.requires_grad]
    decoder_params = list(base.decoder.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": CFG.ENCODER_LR},
            {"params": decoder_params, "lr": CFG.LR},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps    = CFG.WARMUP_STEPS,
        num_training_steps  = total_steps,
    )
    return optimizer, scheduler


def train() -> None:
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    CFG.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device setup ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda")
        log.info(f"Using {n_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
    else:
        device = torch.device("cpu")
        n_gpus = 0
        log.warning("No GPU found — training on CPU (will be slow).")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {CFG.ENCODER_MODEL}")
    tokenizer: XLMRobertaTokenizerFast = XLMRobertaTokenizerFast.from_pretrained(CFG.ENCODER_MODEL)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = TranslationDataset(CFG.DATA_DIR / "train", tokenizer)
    val_dataset   = TranslationDataset(CFG.DATA_DIR / "val",   tokenizer)

    effective_batch = CFG.BATCH_SIZE * max(1, n_gpus) * CFG.GRAD_ACCUM_STEPS
    log.info(f"Effective batch size: {effective_batch}")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = CFG.BATCH_SIZE * max(1, n_gpus),
        shuffle     = True,
        num_workers = CFG.NUM_WORKERS,
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = CFG.BATCH_SIZE * max(1, n_gpus),
        shuffle     = False,
        num_workers = CFG.NUM_WORKERS,
        pin_memory  = device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CustomHybridTranslator(tokenizer).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        log.info(f"Wrapped model in DataParallel across {n_gpus} GPUs.")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params — total: {total_params/1e6:.1f}M  trainable: {trainable_params/1e6:.1f}M")

    # ── Loss / optim / scheduler / scaler ────────────────────────────────────
    criterion = LabelSmoothingCrossEntropy(
        smoothing    = CFG.LABEL_SMOOTHING,
        ignore_index = tokenizer.pad_token_id,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, CFG.MAX_STEPS)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Resume from checkpoint if RESUME_FROM is set ─────────────────────────
    global_step      = 0
    best_val_loss    = float("inf")
    no_improve_count = 0

    resume_path = os.environ.get("RESUME_FROM", "")
    if resume_path and Path(resume_path).exists():
        log.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        # Load model weights
        base_model = model.module if hasattr(model, "module") else model
        base_model.load_state_dict(ckpt["model_state_dict"])
        # Load optimizer + scheduler + scaler
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step   = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log.info(f"Resumed at step {global_step:,}  best_val_loss={best_val_loss:.4f}")
    else:
        log.info("Starting training from scratch …")

    t_start = time.time()
    log.info("Starting training …")

    while global_step < CFG.MAX_STEPS:
        model.train()

        for batch in train_loader:
            if global_step >= CFG.MAX_STEPS:
                break

            src_ids  = batch["src_input_ids"].to(device)
            src_mask = batch["src_attn_mask"].to(device)
            tgt_ids  = batch["tgt_input_ids"].to(device)
            tgt_mask = batch["tgt_attn_mask"].to(device)
            labels   = batch["labels"].to(device)
            lang_ids = batch["tgt_lang_id"].to(device)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(
                    src_input_ids = src_ids,
                    src_attn_mask = src_mask,
                    tgt_input_ids = tgt_ids,
                    tgt_lang_ids  = lang_ids,
                    tgt_attn_mask = tgt_mask,
                )
                loss = criterion(logits, labels) / CFG.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (global_step + 1) % CFG.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss = loss.item() * CFG.GRAD_ACCUM_STEPS
            train_ppl  = math.exp(min(train_loss, 20.0))

            if global_step % 100 == 0:
                elapsed = time.time() - t_start
                lr_enc  = optimizer.param_groups[0]["lr"]
                lr_dec  = optimizer.param_groups[1]["lr"]
                log.info(
                    f"step={global_step:>7d}  loss={train_loss:.4f}  ppl={train_ppl:.2f}  "
                    f"lr_enc={lr_enc:.2e}  lr_dec={lr_dec:.2e}  elapsed={elapsed:.0f}s"
                )
                log_metrics(
                    {
                        "step":       global_step,
                        "train_loss": train_loss,
                        "train_ppl":  train_ppl,
                        "lr_enc":     lr_enc,
                        "lr_dec":     lr_dec,
                        "elapsed_s":  round(elapsed, 2),
                    },
                    CFG.LOG_FILE,
                )

            # ── Evaluation ────────────────────────────────────────────────────
            if global_step > 0 and global_step % CFG.EVAL_EVERY == 0:
                model.eval()
                val_loss_total = 0.0
                val_batches    = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        v_src_ids  = val_batch["src_input_ids"].to(device)
                        v_src_mask = val_batch["src_attn_mask"].to(device)
                        v_tgt_ids  = val_batch["tgt_input_ids"].to(device)
                        v_tgt_mask = val_batch["tgt_attn_mask"].to(device)
                        v_labels   = val_batch["labels"].to(device)
                        v_lang_ids = val_batch["tgt_lang_id"].to(device)

                        with autocast(enabled=(device.type == "cuda")):
                            v_logits = model(
                                src_input_ids = v_src_ids,
                                src_attn_mask = v_src_mask,
                                tgt_input_ids = v_tgt_ids,
                                tgt_lang_ids  = v_lang_ids,
                                tgt_attn_mask = v_tgt_mask,
                            )
                            v_loss = criterion(v_logits, v_labels)
                        val_loss_total += v_loss.item()
                        val_batches    += 1
                        if val_batches >= 200:
                            break

                val_loss = val_loss_total / max(val_batches, 1)
                val_ppl  = math.exp(min(val_loss, 20.0))

                # BLEU
                base_model = model.module if hasattr(model, "module") else model
                bleu_score = evaluate_bleu(base_model, val_loader, tokenizer, device)

                log.info(
                    f"[EVAL step={global_step}]  val_loss={val_loss:.4f}  "
                    f"val_ppl={val_ppl:.2f}  BLEU={bleu_score:.2f}"
                )
                log_metrics(
                    {
                        "step":      global_step,
                        "val_loss":  val_loss,
                        "val_ppl":   val_ppl,
                        "bleu":      bleu_score,
                        "type":      "eval",
                    },
                    CFG.LOG_FILE,
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    log.info(f"No improvement ({no_improve_count}/{CFG.EARLY_STOP_PATIENCE})")
                    if no_improve_count >= CFG.EARLY_STOP_PATIENCE:
                        log.info("Early stopping triggered.")
                        return

                model.train()

            # ── Checkpoint ───────────────────────────────────────────────────
            if global_step > 0 and global_step % CFG.SAVE_EVERY == 0:
                base_model = model.module if hasattr(model, "module") else model
                ckpt = save_checkpoint(
                    base_model, optimizer, scheduler, scaler,
                    global_step, best_val_loss, CFG.CHECKPOINT_DIR
                )
                if global_step % CFG.HUB_PUSH_EVERY == 0:
                    push_checkpoint_to_hub(ckpt, CFG.HF_REPO_ID, CFG.HF_TOKEN)

            global_step += 1

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
