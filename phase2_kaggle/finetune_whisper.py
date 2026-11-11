"""
Phase 2 — Whisper-Small LoRA Fine-tuning for Indian-Accent ASR
===============================================================
Fine-tunes openai/whisper-small on Common Voice 13 (hi, ta, bn, te, mr)
and FLEURS Indian languages using 4-bit QLoRA + PEFT LoRA.
Evaluates with WER, pushes adapter to Hub, merges weights, and exports
to ONNX via optimum.

Designed for Kaggle with 2× T4 GPUs (16 GB VRAM each).
"""

import os
import gc
import json
import time
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    concatenate_datasets,
    Audio,
)
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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
MODEL_ID = "openai/whisper-small"
SAMPLE_RATE = 16_000
SEED = 42

# Common Voice 13 locale → Whisper language token
CV_LOCALES = {
    "hi": "hi",   # Hindi
    "ta": "ta",   # Tamil
    "bn": "bn",   # Bengali
    "te": "te",   # Telugu
    "mr": "mr",   # Marathi
}

# FLEURS dataset language codes (BCP-47)
FLEURS_LANG_CODES = {
    "hi": "hi_in",
    "ta": "ta_in",
    "bn": "bn_in",
    "te": "te_in",
    "mr": "mr_in",
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Whisper-Small for Indian ASR")

    # Hub / paths
    parser.add_argument("--hf_username", type=str, default=os.environ.get("HF_USERNAME", ""))
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/whisper_lora")
    parser.add_argument("--onnx_dir", type=str, default="/kaggle/working/whisper_onnx")
    parser.add_argument("--merged_dir", type=str, default="/kaggle/working/whisper_merged")

    # Training hyper-parameters
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1000)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Dataset options
    parser.add_argument("--use_fleurs", action="store_true", default=True,
                        help="Include FLEURS as supplementary dataset")
    parser.add_argument("--max_cv_samples_per_lang", type=int, default=5000,
                        help="Cap Common Voice samples per language (saves memory)")

    # Export
    parser.add_argument("--export_onnx", action="store_true", default=True)
    parser.add_argument("--push_to_hub", action="store_true", default=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# BitsAndBytes config
# ---------------------------------------------------------------------------

def get_bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


# ---------------------------------------------------------------------------
# Model & processor loading
# ---------------------------------------------------------------------------

def load_model_and_processor(hf_token: str):
    """Load Whisper model in 4-bit QLoRA mode."""
    logger.info("Loading WhisperProcessor and WhisperFeatureExtractor …")
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        language="Hindi",   # default language; overridden per sample at inference
        task="transcribe",
        token=hf_token or None,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        MODEL_ID,
        token=hf_token or None,
    )

    logger.info("Loading WhisperForConditionalGeneration in 4-bit …")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=hf_token or None,
    )

    # Required Whisper fine-tuning settings
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Apply LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor, feature_extractor


# ---------------------------------------------------------------------------
# Dataset loading & preprocessing
# ---------------------------------------------------------------------------

def _cast_audio(ds: Dataset) -> Dataset:
    """Cast audio column to 16 kHz mono."""
    return ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


def load_common_voice_split(
    locale: str,
    split: str,
    hf_token: str,
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load a single Common Voice 13 locale split."""
    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            locale,
            split=split,
            token=hf_token or None,
            trust_remote_code=True,
        )
        ds = _cast_audio(ds)
        # Keep only relevant columns
        keep = [c for c in ["audio", "sentence", "locale"] if c in ds.column_names]
        ds = ds.select_columns(keep)
        # Add locale as language tag if not present
        if "locale" not in ds.column_names:
            ds = ds.add_column("locale", [locale] * len(ds))
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        logger.info(f"  CV13 {locale}/{split}: {len(ds):,} samples")
        return ds
    except Exception as e:
        logger.warning(f"Failed to load CV13 {locale}/{split}: {e}")
        return None


def load_fleurs_split(
    lang_code: str,
    split: str,
    hf_token: str,
    max_samples: Optional[int] = None,
) -> Optional[Dataset]:
    """Load a FLEURS split for a given language."""
    try:
        ds = load_dataset(
            "google/fleurs",
            lang_code,
            split=split,
            token=hf_token or None,
            trust_remote_code=True,
        )
        ds = _cast_audio(ds)
        # Normalize columns: keep audio + transcription text
        # FLEURS uses 'transcription' column
        rename_map = {}
        if "transcription" in ds.column_names and "sentence" not in ds.column_names:
            rename_map["transcription"] = "sentence"
        if rename_map:
            ds = ds.rename_columns(rename_map)
        keep = [c for c in ["audio", "sentence", "language"] if c in ds.column_names]
        ds = ds.select_columns(keep)
        # Harmonise locale column name
        if "language" in ds.column_names and "locale" not in ds.column_names:
            ds = ds.rename_columns({"language": "locale"})
        elif "locale" not in ds.column_names:
            locale = lang_code.split("_")[0]
            ds = ds.add_column("locale", [locale] * len(ds))
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        logger.info(f"  FLEURS {lang_code}/{split}: {len(ds):,} samples")
        return ds
    except Exception as e:
        logger.warning(f"Failed to load FLEURS {lang_code}/{split}: {e}")
        return None


def build_combined_dataset(args) -> Tuple[Dataset, Dataset]:
    """
    Build combined train/validation datasets from Common Voice 13
    and optionally FLEURS for all configured Indian language locales.
    """
    train_parts: List[Dataset] = []
    eval_parts: List[Dataset] = []

    for locale, _lang in CV_LOCALES.items():
        train_ds = load_common_voice_split(
            locale, "train", args.hf_token, args.max_cv_samples_per_lang
        )
        val_ds = load_common_voice_split(
            locale, "validation", args.hf_token, min(500, args.max_cv_samples_per_lang or 500)
        )
        if train_ds:
            train_parts.append(train_ds)
        if val_ds:
            eval_parts.append(val_ds)

    if args.use_fleurs:
        for locale, fleurs_code in FLEURS_LANG_CODES.items():
            train_ds = load_fleurs_split(
                fleurs_code, "train", args.hf_token, 2000
            )
            val_ds = load_fleurs_split(
                fleurs_code, "validation", args.hf_token, 200
            )
            if train_ds:
                train_parts.append(train_ds)
            if val_ds:
                eval_parts.append(val_ds)

    if not train_parts:
        raise RuntimeError("No training data could be loaded! Check dataset access.")

    train_dataset = concatenate_datasets(train_parts).shuffle(seed=SEED)
    eval_dataset = concatenate_datasets(eval_parts).shuffle(seed=SEED)

    if args.max_train_samples and args.max_train_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples and args.max_eval_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    logger.info(f"Combined train: {len(train_dataset):,} | eval: {len(eval_dataset):,}")
    return train_dataset, eval_dataset


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Map function: convert raw audio → log-mel spectrogram + tokenized labels.
    """
    audio = batch["audio"]
    # Compute log-mel spectrogram
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]

    # Tokenize transcription
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for speech-to-text seq2seq models.
    Pads input features and label sequences independently.
    """
    processor: WhisperProcessor
    decoder_start_token_id: int
    pad_to_multiple_of: int = 8
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── Audio features (always fixed length 3000 for Whisper) ───────────
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # ── Label sequences ──────────────────────────────────────────────────
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Replace pad token id in labels with -100 so loss ignores padding
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Drop BOS token if it was prepended by the tokenizer
        if (
            (labels[:, 0] == self.decoder_start_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# WER metric
# ---------------------------------------------------------------------------

def build_compute_metrics(tokenizer):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id for decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def preprocess_and_cache(
    raw_train: Dataset,
    raw_eval: Dataset,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    cache_dir: str,
) -> Tuple[Dataset, Dataset]:
    """Tokenize and cache datasets."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def _prepare(batch):
        return prepare_dataset(batch, feature_extractor, tokenizer)

    logger.info("Preprocessing train split …")
    train_ds = raw_train.map(
        _prepare,
        remove_columns=raw_train.column_names,
        num_proc=1,  # audio processing is best single-threaded
        desc="Processing train audio",
        cache_file_name=str(Path(cache_dir) / "train_cache"),
    )

    logger.info("Preprocessing eval split …")
    eval_ds = raw_eval.map(
        _prepare,
        remove_columns=raw_eval.column_names,
        num_proc=1,
        desc="Processing eval audio",
        cache_file_name=str(Path(cache_dir) / "eval_cache"),
    )

    return train_ds, eval_ds


def train(args, model, processor, feature_extractor, train_ds, eval_ds) -> str:
    """Run Seq2SeqTrainer and return adapter output directory."""
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    hub_model_id = (f"{args.hf_username}/whisper-small-indian-lora"
                    if args.hf_username else "whisper-small-indian-lora")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.push_to_hub and bool(args.hf_username),
        hub_model_id=hub_model_id if args.push_to_hub else None,
        hub_token=args.hf_token or None,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=SEED,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor.tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    logger.info("Starting Whisper LoRA training …")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Evaluating …")
    eval_metrics = trainer.evaluate(max_length=225)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if args.push_to_hub and args.hf_username:
        logger.info(f"Pushing adapter to Hub: {hub_model_id}")
        trainer.push_to_hub(commit_message="Whisper LoRA — Phase 2 training complete")

    return output_dir


# ---------------------------------------------------------------------------
# Merge LoRA → full model
# ---------------------------------------------------------------------------

def merge_and_save(adapter_dir: str, merged_dir: str, hf_token: str) -> str:
    """Merge LoRA adapter into base Whisper model and save."""
    logger.info("Merging LoRA adapter into base Whisper model …")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=hf_token or None,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)

    # Save processor
    processor = WhisperProcessor.from_pretrained(adapter_dir)
    processor.save_pretrained(merged_dir)

    logger.info(f"Merged Whisper model saved to: {merged_dir}")
    return merged_dir


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(merged_dir: str, onnx_dir: str):
    """Export merged Whisper model to ONNX using optimum."""
    try:
        from optimum.exporters.onnx import main_export
        logger.info(f"Exporting Whisper to ONNX → {onnx_dir} …")
        Path(onnx_dir).mkdir(parents=True, exist_ok=True)
        main_export(
            model_name_or_path=merged_dir,
            output=onnx_dir,
            task="automatic-speech-recognition",
            opset=17,
            device="cpu",
            fp16=False,
        )
        logger.info("Whisper ONNX export complete.")
    except Exception as e:
        logger.error(f"Whisper ONNX export failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------

def push_directory_to_hub(local_dir: str, repo_id: str, folder_path: str, hf_token: str):
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


# ---------------------------------------------------------------------------
# File size reporting
# ---------------------------------------------------------------------------

def report_sizes(directories: List[str]):
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

    if args.hf_token:
        login(token=args.hf_token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
        args.hf_token = os.environ["HF_TOKEN"]

    if not args.hf_username and os.environ.get("HF_USERNAME"):
        args.hf_username = os.environ["HF_USERNAME"]

    hub_model_id = (f"{args.hf_username}/whisper-small-indian-lora"
                    if args.hf_username else "whisper-small-indian-lora")

    # ── 1. Load model & processor ────────────────────────────────────────────
    model, processor, feature_extractor = load_model_and_processor(args.hf_token)

    # ── 2. Build combined dataset ────────────────────────────────────────────
    raw_train, raw_eval = build_combined_dataset(args)

    # ── 3. Preprocess & cache ─────────────────────────────────────────────────
    cache_dir = str(Path(args.output_dir) / "dataset_cache")
    train_ds, eval_ds = preprocess_and_cache(
        raw_train, raw_eval, feature_extractor, processor.tokenizer, cache_dir
    )

    # ── 4. Train ─────────────────────────────────────────────────────────────
    adapter_dir = train(args, model, processor, feature_extractor, train_ds, eval_ds)

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 5. Merge ─────────────────────────────────────────────────────────────
    merged_dir = merge_and_save(adapter_dir, args.merged_dir, args.hf_token)

    # ── 6. ONNX export ───────────────────────────────────────────────────────
    if args.export_onnx:
        export_to_onnx(merged_dir, args.onnx_dir)

    # ── 7. Push exports to Hub ───────────────────────────────────────────────
    if args.push_to_hub and args.hf_username and args.hf_token:
        push_directory_to_hub(args.onnx_dir, hub_model_id, "onnx", args.hf_token)
        push_directory_to_hub(merged_dir, hub_model_id, "merged", args.hf_token)

    # ── 8. Report sizes ──────────────────────────────────────────────────────
    report_sizes([adapter_dir, merged_dir, args.onnx_dir])

    logger.info("Phase 2 Whisper fine-tuning pipeline complete ✓")


if __name__ == "__main__":
    main()
