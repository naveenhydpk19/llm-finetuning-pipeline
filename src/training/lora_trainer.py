"""
LoRA/PEFT trainer — wraps HuggingFace PEFT + TRL SFTTrainer
for use inside a SageMaker training job container.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class LoRATrainingConfig:
    base_model: str
    output_dir: str
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")
    # Training
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    # QLoRA
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"


class LoRATrainer:
    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Training on device: %s", self.device)

    def load_model(self):
        cfg = self.config

        bnb_config = None
        if cfg.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        if cfg.use_4bit:
            model = prepare_model_for_kbit_training(model)

        if cfg.gradient_checkpointing:
            model.enable_input_require_grads()

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.target_modules),
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def train(self, dataset: DatasetDict) -> str:
        model, tokenizer = self.load_model()
        cfg = self.config

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=cfg.bf16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            report_to="none",   # LangSmith handles observability
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            args=training_args,
        )

        logger.info("Starting LoRA fine-tuning: %s", cfg.base_model)
        trainer.train()

        adapter_path = os.path.join(cfg.output_dir, "adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        logger.info("Adapter saved to %s", adapter_path)
        return adapter_path

    def merge_and_save(self, adapter_path: str, output_path: str) -> None:
        """Merge LoRA adapter into base model weights for deployment."""
        from peft import PeftModel

        logger.info("Merging adapter into base model...")
        base = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        merged = PeftModel.from_pretrained(base, adapter_path)
        merged = merged.merge_and_unload()
        merged.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.save_pretrained(output_path)
        logger.info("Merged model saved to %s", output_path)
