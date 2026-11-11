"""
Phase 2 — NLLB-200-distilled-600M QLoRA Fine-tuning
======================================================
Fine-tunes facebook/nllb-200-distilled-600M on a multilingual translation
dataset using 4-bit QLoRA (BitsAndBytes + PEFT), evaluates with BLEU,
pushes adapters to the Hub, then merges weights and exports to ONNX and
CTranslate2.

Designed for Kaggle with 2× T4 GPUs (16 GB VRAM each).
"""

import os
import gc
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
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
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "facebook/nllb-200-distilled-600M"

# NLLB language codes — required for tokenizer src/tgt lang settings
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pt": "por_Latn",
}

LANGUAGE_PAIRS = [
    ("en", "hi"), ("en", "ta"), ("en", "bn"), ("en", "te"),
    ("en", "mr"), ("en", "gu"), ("en", "kn"), ("en", "ml"),
    ("hi", "en"), ("ta", "en"), ("bn", "en"),
]

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
SEED = 42

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune NLLB-200-600M")

    # Hub / paths
    parser.add_argument("--hf_username", type=str, default=os.environ.get("HF_USERNAME", ""),
                        help="Hugging Face username")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""),
                        help="Hugging Face API token")
    parser.add_argument("--hub_dataset", type=str, default="",
                        help="HF Hub dataset path (e.g. username/multilingual-translation-data)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/nllb_qlora",
                        help="Directory to save checkpoints")
    parser.add_argument("--onnx_dir", type=str, default="/kaggle/working/nllb_onnx",
                        help="Directory to save ONNX export")
    parser.add_argument("--ct2_dir", type=str, default="/kaggle/working/nllb_ct2",
                        help="Directory to save CTranslate2 export")
    parser.add_argument("--merged_dir", type=str, default="/kaggle/working/nllb_merged",
                        help="Directory for merged weights")

    # Training hyper-parameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training set size for quick smoke-tests")
    parser.add_argument("--max_eval_samples", type=int, default=2000)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Export flags
    parser.add_argument("--export_onnx", action="store_true", default=True)
    parser.add_argument("--export_ct2", action="store_true", default=True)
    parser.add_argument("--push_to_hub", action="store_true", default=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# BitsAndBytes config
# ---------------------------------------------------------------------------

def get_bnb_config():
    """Return 4-bit NF4 BitsAndBytesConfig."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


# ---------------------------------------------------------------------------
# Model & tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(hf_token: str):
    """Load NLLB model in 4-bit and prepare for k-bit training."""
    logger.info("Loading NllbTokenizer …")
    tokenizer = NllbTokenizer.from_pretrained(
        MODEL_ID,
        token=hf_token or None,
    )

    logger.info("Loading NLLB model in 4-bit QLoRA mode …")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=hf_token or None,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # required for gradient checkpointing

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset loading & preprocessing
# ---------------------------------------------------------------------------

def load_hf_hub_dataset(hub_path: str, hf_token: str) -> Optional[DatasetDict]:
    """Try loading the custom multilingual dataset from the Hub."""
    try:
        logger.info(f"Attempting to load dataset from Hub: {hub_path}")
        ds = load_dataset(hub_path, token=hf_token or None)
        logger.info("Successfully loaded Hub dataset.")
        return ds
    except Exception as e:
        logger.warning(f"Could not load Hub dataset ({e}). Falling back to opus-100.")
        return None


def load_fallback_dataset() -> DatasetDict:
    """
    Fallback: Helsinki-NLP/opus-100 en-hi.
    Returns a DatasetDict with train/validation splits.
    """
    logger.info("Loading fallback dataset: Helsinki-NLP/opus-100 (en-hi) …")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-hi")
    # opus-100 may not have a validation split — create one
    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=0.02, seed=SEED)
        ds = DatasetDict({"train": split["train"], "validation": split["test"]})
    return ds


def preprocess_function_hub(examples, tokenizer, src_lang_code, tgt_lang_code,
                             max_input=MAX_INPUT_LENGTH, max_target=MAX_TARGET_LENGTH):
    """
    Pre-process examples from the custom Hub dataset.
    Expects columns: 'source_text', 'target_text', 'source_lang', 'target_lang'
    """
    tokenizer.src_lang = src_lang_code
    inputs = tokenizer(
        examples["source_text"],
        max_length=max_input,
        truncation=True,
        padding=False,
    )
    tokenizer.tgt_lang = tgt_lang_code
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            examples["target_text"],
            max_length=max_target,
            truncation=True,
            padding=False,
        )
    inputs["labels"] = targets["input_ids"]
    return inputs


def preprocess_function_opus(examples, tokenizer,
                              max_input=MAX_INPUT_LENGTH, max_target=MAX_TARGET_LENGTH):
    """
    Pre-process opus-100 en-hi examples.
    Columns: 'translation' dict with keys 'en' and 'hi'.
    """
    src_texts = [ex["en"] for ex in examples["translation"]]
    tgt_texts = [ex["hi"] for ex in examples["translation"]]

    tokenizer.src_lang = NLLB_LANG_CODES["en"]
    inputs = tokenizer(
        src_texts,
        max_length=max_input,
        truncation=True,
        padding=False,
    )
    tokenizer.tgt_lang = NLLB_LANG_CODES["hi"]
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            tgt_texts,
            max_length=max_target,
            truncation=True,
            padding=False,
        )
    inputs["labels"] = targets["input_ids"]
    return inputs


def prepare_datasets(args, tokenizer) -> Tuple[Any, Any]:
    """Load and preprocess train/validation datasets."""
    hub_path = args.hub_dataset or (f"{args.hf_username}/multilingual-translation-data"
                                    if args.hf_username else "")
    raw_ds = None
    if hub_path:
        raw_ds = load_hf_hub_dataset(hub_path, args.hf_token)

    if raw_ds is not None:
        # Hub dataset — preprocess using source/target lang columns
        # We'll use English→Hindi subset for primary fine-tuning
        def preprocess_hub(examples):
            # Filter to en→hi pairs if lang columns exist
            return preprocess_function_hub(
                examples, tokenizer,
                NLLB_LANG_CODES["en"], NLLB_LANG_CODES["hi"]
            )

        logger.info("Tokenizing Hub dataset …")
        tokenized = raw_ds.map(
            preprocess_hub,
            batched=True,
            remove_columns=raw_ds["train"].column_names,
            num_proc=2,
            desc="Tokenizing",
        )
    else:
        raw_ds = load_fallback_dataset()
        logger.info("Tokenizing opus-100 fallback dataset …")
        tokenized = raw_ds.map(
            lambda ex: preprocess_function_opus(ex, tokenizer),
            batched=True,
            remove_columns=raw_ds["train"].column_names,
            num_proc=2,
            desc="Tokenizing",
        )

    train_ds = tokenized["train"]
    eval_ds = tokenized["validation"]

    # Optional size caps
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        train_ds = train_ds.select(range(args.max_train_samples))
    if args.max_eval_samples and args.max_eval_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(args.max_eval_samples))

    logger.info(f"Train samples: {len(train_ds):,} | Eval samples: {len(eval_ds):,}")
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# BLEU metric
# ---------------------------------------------------------------------------

def build_compute_metrics(tokenizer):
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds may be a tuple when predict_with_generate=True
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in labels with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip whitespace
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]  # sacrebleu expects list of refs

        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": round(result["score"], 4)}

        # Prediction length
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args, model, tokenizer, train_ds, eval_ds) -> str:
    """Run Seq2SeqTrainer and return path to best checkpoint."""
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    hub_model_id = (f"{args.hf_username}/nllb-200-600m-qlora"
                    if args.hf_username else "nllb-200-600m-qlora")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=256,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        push_to_hub=args.push_to_hub and bool(args.hf_username),
        hub_model_id=hub_model_id if args.push_to_hub else None,
        hub_token=args.hf_token or None,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
        sortish_sampler=True,
        seed=SEED,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Resume from checkpoint if available
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    logger.info("Starting training …")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate
    logger.info("Evaluating …")
    eval_metrics = trainer.evaluate(max_length=256)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if args.push_to_hub and args.hf_username:
        logger.info(f"Pushing adapter to Hub: {hub_model_id}")
        trainer.push_to_hub(commit_message="QLoRA adapter — Phase 2 training complete")

    return output_dir


# ---------------------------------------------------------------------------
# Merge LoRA → full model
# ---------------------------------------------------------------------------

def merge_and_save(adapter_dir: str, merged_dir: str, hf_token: str) -> str:
    """Merge LoRA adapter into base model and save."""
    logger.info("Merging LoRA adapter into base model …")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    # Reload base model in full precision for merging
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=hf_token or None,
    )
    tokenizer = NllbTokenizer.from_pretrained(adapter_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    logger.info(f"Merged model saved to: {merged_dir}")
    return merged_dir


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(merged_dir: str, onnx_dir: str):
    """Export merged model to ONNX using optimum."""
    try:
        from optimum.exporters.onnx import main_export
        logger.info(f"Exporting to ONNX → {onnx_dir} …")
        Path(onnx_dir).mkdir(parents=True, exist_ok=True)
        main_export(
            model_name_or_path=merged_dir,
            output=onnx_dir,
            task="seq2seq-lm",
            opset=17,
            device="cpu",
            fp16=False,
        )
        logger.info("ONNX export complete.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise


# ---------------------------------------------------------------------------
# CTranslate2 export
# ---------------------------------------------------------------------------

def export_to_ct2(merged_dir: str, ct2_dir: str):
    """Convert merged model to CTranslate2 format."""
    try:
        import ctranslate2
        logger.info(f"Converting to CTranslate2 → {ct2_dir} …")
        Path(ct2_dir).mkdir(parents=True, exist_ok=True)
        converter = ctranslate2.converters.OpusMTConverter(merged_dir)
        converter.convert(
            ct2_dir,
            quantization="int8",
            force=True,
        )
        logger.info("CTranslate2 conversion complete.")
    except Exception as e:
        logger.warning(f"OpusMTConverter failed ({e}), trying HfConverter …")
        try:
            import ctranslate2
            converter = ctranslate2.converters.HfConverter(merged_dir)
            converter.convert(
                ct2_dir,
                quantization="int8",
                force=True,
            )
            logger.info("CTranslate2 conversion (HfConverter) complete.")
        except Exception as e2:
            logger.error(f"CTranslate2 export failed: {e2}")
            raise


# ---------------------------------------------------------------------------
# Hub upload helpers
# ---------------------------------------------------------------------------

def push_directory_to_hub(local_dir: str, repo_id: str, folder_path: str, hf_token: str):
    """Upload all files in local_dir to a Hub repo under folder_path."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, repo_type="model")
    logger.info(f"Uploading {local_dir} → {repo_id}/{folder_path} …")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        path_in_repo=folder_path,
        token=hf_token,
        commit_message=f"Upload {folder_path}",
    )
    logger.info("Upload complete.")


# ---------------------------------------------------------------------------
# File size reporting
# ---------------------------------------------------------------------------

def report_sizes(directories: List[str]):
    """Print total sizes of output directories."""
    for d in directories:
        p = Path(d)
        if not p.exists():
            logger.warning(f"Directory not found: {d}")
            continue
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        logger.info(f"{d}: {total / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(SEED)

    # Authenticate
    if args.hf_token:
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
        args.hf_token = os.environ["HF_TOKEN"]

    if not args.hf_username and os.environ.get("HF_USERNAME"):
        args.hf_username = os.environ["HF_USERNAME"]

    hub_model_id = (f"{args.hf_username}/nllb-200-600m-qlora"
                    if args.hf_username else "nllb-200-600m-qlora")

    # ── 1. Load model & tokenizer ────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.hf_token)

    # ── 2. Prepare datasets ──────────────────────────────────────────────────
    train_ds, eval_ds = prepare_datasets(args, tokenizer)

    # ── 3. Train ─────────────────────────────────────────────────────────────
    adapter_dir = train(args, model, tokenizer, train_ds, eval_ds)

    # Free GPU memory before merging
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 4. Merge ─────────────────────────────────────────────────────────────
    merged_dir = merge_and_save(adapter_dir, args.merged_dir, args.hf_token)

    # ── 5. ONNX export ───────────────────────────────────────────────────────
    if args.export_onnx:
        export_to_onnx(merged_dir, args.onnx_dir)

    # ── 6. CTranslate2 export ────────────────────────────────────────────────
    if args.export_ct2:
        export_to_ct2(merged_dir, args.ct2_dir)

    # ── 7. Push exports to Hub ───────────────────────────────────────────────
    if args.push_to_hub and args.hf_username and args.hf_token:
        push_directory_to_hub(args.onnx_dir, hub_model_id, "onnx", args.hf_token)
        push_directory_to_hub(args.ct2_dir, hub_model_id, "ct2_model", args.hf_token)
        push_directory_to_hub(merged_dir, hub_model_id, "merged", args.hf_token)

    # ── 8. Report sizes ──────────────────────────────────────────────────────
    report_sizes([adapter_dir, merged_dir, args.onnx_dir, args.ct2_dir])

    logger.info("Phase 2 NLLB fine-tuning pipeline complete ✓")


if __name__ == "__main__":
    main()
