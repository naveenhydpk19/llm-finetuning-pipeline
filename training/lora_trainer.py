"""
LoRA/PEFT fine-tuning pipeline for domain-adapted LLMs on AWS SageMaker.
Uses HuggingFace Transformers + PEFT library with PyTorch backend.
"""

import os
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class LoRATrainingConfig:
    # Model
    base_model_id: str = "meta-llama/Llama-2-7b-hf"
    model_max_length: int = 2048
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 25

    # Paths (SageMaker SM_* env vars)
    output_dir: str = field(default_factory=lambda: os.environ.get("SM_OUTPUT_DATA_DIR", "/tmp/output"))
    train_data_dir: str = field(default_factory=lambda: os.environ.get("SM_CHANNEL_TRAIN", "/tmp/train"))
    eval_data_dir: str = field(default_factory=lambda: os.environ.get("SM_CHANNEL_EVAL", "/tmp/eval"))


def load_model_and_tokenizer(config: LoRATrainingConfig):
    logger.info(f"Loading base model: {config.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(config.base_model_id, **load_kwargs)

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(model, config: LoRATrainingConfig):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int):
    def tokenize(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize, remove_columns=dataset["train"].column_names, batched=False)


def train(config: LoRATrainingConfig):
    model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(model, config)

    logger.info(f"Loading dataset from {config.train_data_dir}")
    dataset = DatasetDict({
        "train": load_from_disk(config.train_data_dir),
        "eval": load_from_disk(config.eval_data_dir),
    })
    tokenized = tokenize_dataset(dataset, tokenizer, config.model_max_length)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Model saved to {config.output_dir}")

    # Save training metrics
    metrics = trainer.evaluate()
    metrics_path = os.path.join(config.output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Eval metrics saved: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    args = parser.parse_args()

    config = LoRATrainingConfig(
        base_model_id=args.base_model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        load_in_4bit=args.load_in_4bit,
    )
    train(config)
