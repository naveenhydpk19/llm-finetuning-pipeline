"""
Local demo — runs the full LoRA fine-tuning pipeline with zero GPU/AWS dependencies.

Usage:
    python demo_local.py
    python demo_local.py --model-name my-domain-model --epochs 2

What it does:
  1. Generates a synthetic instruction dataset (no S3 needed)
  2. Runs data preprocessing (dedup, quality filter, tokenization)
  3. Simulates LoRA training with mock metrics
  4. Runs mock Ragas evaluation and checks promotion thresholds
  5. Simulates model registration in SageMaker Model Registry
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessing import (
    deduplicate,
    format_instruction,
    quality_filter,
)

# ── Synthetic training data ───────────────────────────────────────
SYNTHETIC_INSTRUCTIONS = [
    {"instruction": "Explain what Retrieval-Augmented Generation (RAG) is.", "input": "", "output": "RAG is an AI framework that combines a retrieval system with a language model. It fetches relevant documents at inference time and uses them to ground the LLM's response, reducing hallucination."},
    {"instruction": "What is LoRA in the context of LLM fine-tuning?", "input": "", "output": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds small trainable rank-decomposition matrices to frozen model weights, training only ~0.1-1% of parameters."},
    {"instruction": "Summarize the following text.", "input": "OpenSearch is a distributed search and analytics engine based on Elasticsearch. It supports full-text search, vector search, and aggregations.", "output": "OpenSearch is a distributed search/analytics engine supporting full-text search, vector search, and data aggregations."},
    {"instruction": "What is the difference between SFT and DPO?", "input": "", "output": "SFT (Supervised Fine-Tuning) trains a model to mimic desired outputs. DPO (Direct Preference Optimization) trains the model to prefer chosen responses over rejected ones, aligning it with human preferences without a separate reward model."},
    {"instruction": "Explain gradient checkpointing.", "input": "", "output": "Gradient checkpointing reduces GPU memory usage during training by not storing all intermediate activations. Instead, they are recomputed on the backward pass, trading compute for memory."},
    {"instruction": "What is pgvector?", "input": "", "output": "pgvector is a PostgreSQL extension that adds vector similarity search capabilities, enabling storage and querying of embeddings directly in a relational database."},
    {"instruction": "How does LangGraph differ from LangChain?", "input": "", "output": "LangChain provides components for LLM pipelines. LangGraph extends it with stateful, graph-based agent orchestration — nodes are functions that read/write shared state, with conditional routing and checkpointing."},
    {"instruction": "What is QLoRA?", "input": "", "output": "QLoRA combines LoRA with 4-bit quantization of base model weights using NF4 format. This enables fine-tuning of very large models (70B+) on a single GPU by drastically reducing VRAM requirements."},
    {"instruction": "Explain the Ragas faithfulness metric.", "input": "", "output": "Ragas faithfulness measures whether the generated answer is factually consistent with the retrieved context. A score of 1.0 means every claim in the answer can be attributed to the context."},
    {"instruction": "What is AWS SageMaker Pipelines?", "input": "", "output": "SageMaker Pipelines is a CI/CD service for ML workflows. It enables reproducible, parameterised pipeline runs with step caching, experiment tracking, and integration with the SageMaker Model Registry."},
    # Deliberately add some near-duplicates for dedup testing
    {"instruction": "What is LoRA?", "input": "", "output": "LoRA adds trainable low-rank matrices to frozen model layers for parameter-efficient fine-tuning."},
    {"instruction": "What is RAG?", "input": "", "output": "RAG combines retrieval with generation to reduce hallucinations and ground responses in real documents."},
]


def generate_dataset(n: int = 50) -> list:
    """Expand synthetic data to n examples by repeating with minor variation."""
    base = SYNTHETIC_INSTRUCTIONS.copy()
    while len(base) < n:
        sample = random.choice(SYNTHETIC_INSTRUCTIONS).copy()
        # Add minor variation so dedup doesn't remove everything
        sample = {k: v + " " * (len(base) % 3) for k, v in sample.items()}
        base.append(sample)
    return base[:n]


# ── Mock trainer ──────────────────────────────────────────────────
class MockLoRATrainer:
    def __init__(self, model_name: str, num_epochs: int = 3, lora_r: int = 16):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.lora_r = lora_r

    def train(self, num_samples: int):
        print(f"\n  Base model:   meta-llama/Meta-Llama-3-8B-Instruct (mock)")
        print(f"  LoRA rank:    r={self.lora_r} | alpha={self.lora_r * 2}")
        print(f"  Target mods:  q_proj, v_proj, k_proj, o_proj")
        print(f"  Trainable:    0.42% of parameters (~33M / 8B)")
        print(f"  Dataset:      {num_samples} examples")
        print(f"  Epochs:       {self.num_epochs}\n")

        steps_per_epoch = max(1, num_samples // 32)
        history = []
        for epoch in range(1, self.num_epochs + 1):
            for step in range(1, steps_per_epoch + 1):
                loss = 2.1 * (0.72 ** (epoch * steps_per_epoch + step))
                if step % max(1, steps_per_epoch // 3) == 0 or step == steps_per_epoch:
                    print(f"  Epoch {epoch}/{self.num_epochs} | Step {step}/{steps_per_epoch} | loss: {loss:.4f}")
                    time.sleep(0.05)
            val_loss = loss * 1.08
            history.append({"epoch": epoch, "train_loss": loss, "val_loss": val_loss})

        return {
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "adapter_path": f"outputs/{self.model_name}/adapter",
            "history": history,
        }


# ── Mock evaluator ────────────────────────────────────────────────
class MockEvaluator:
    THRESHOLDS = {"domain_accuracy": 0.97, "hallucination_rate": 0.03, "faithfulness": 0.85}

    def evaluate(self):
        print("\n  Running domain accuracy benchmark...")
        time.sleep(0.3)
        scores = {
            "domain_accuracy": 0.971,
            "hallucination_rate": 0.021,
            "faithfulness": 0.912,
            "answer_relevancy": 0.887,
        }
        return scores

    def check_promotion(self, scores: dict) -> tuple[bool, list]:
        failures = []
        for metric, threshold in self.THRESHOLDS.items():
            val = scores.get(metric, 0)
            if metric == "hallucination_rate":
                if val > threshold:
                    failures.append(f"{metric}: {val:.3f} > max {threshold}")
            else:
                if val < threshold:
                    failures.append(f"{metric}: {val:.3f} < min {threshold}")
        return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Pipeline — Local Demo")
    parser.add_argument("--model-name", default="enterprise-llm-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--dataset-size", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("  LLM Fine-Tuning Pipeline (LoRA/PEFT) — Local Demo")
    print("  (No GPU / AWS / HuggingFace token needed)")
    print("=" * 60)

    # ── Step 1: Data prep ────────────────────────────────────────
    print(f"\n[1/4] Preparing dataset ({args.dataset_size} examples)...")
    raw_data = generate_dataset(args.dataset_size)

    formatted = [format_instruction(ex) for ex in raw_data]
    print(f"  Raw examples:      {len(formatted)}")

    filtered = [ex for ex in formatted if quality_filter(ex)]
    print(f"  After quality filter: {len(filtered)}")

    deduped = deduplicate(filtered)
    print(f"  After deduplication:  {len(deduped)}")

    train_size = int(len(deduped) * 0.95)
    val_size = len(deduped) - train_size
    print(f"  Train / Val split: {train_size} / {val_size}")

    # ── Step 2: Training ─────────────────────────────────────────
    print(f"\n[2/4] Fine-tuning with LoRA (r={args.lora_r})...")
    trainer = MockLoRATrainer(args.model_name, args.epochs, args.lora_r)
    train_result = trainer.train(train_size)
    print(f"\n  ✅ Training complete")
    print(f"  Final train loss: {train_result['final_train_loss']:.4f}")
    print(f"  Final val loss:   {train_result['final_val_loss']:.4f}")
    print(f"  Adapter saved to: {train_result['adapter_path']}")

    # ── Step 3: Evaluation ───────────────────────────────────────
    print(f"\n[3/4] Running evaluation harness (Ragas + domain benchmarks)...")
    evaluator = MockEvaluator()
    scores = evaluator.evaluate()

    print(f"\n  Evaluation scores:")
    for metric, val in scores.items():
        threshold = MockEvaluator.THRESHOLDS.get(metric)
        if threshold:
            is_rate = "rate" in metric
            ok = val <= threshold if is_rate else val >= threshold
            sym = "✅" if ok else "❌"
            direction = f"≤ {threshold}" if is_rate else f"≥ {threshold}"
            print(f"  {sym} {metric}: {val:.3f} (threshold: {direction})")
        else:
            print(f"  📊 {metric}: {val:.3f}")

    passed, failures = evaluator.check_promotion(scores)

    # ── Step 4: Promotion ────────────────────────────────────────
    print(f"\n[4/4] Promotion gate check...")
    if passed:
        print(f"  ✅ All thresholds passed — model approved for registry")
        print(f"  [MockRegistry] Registering {args.model_name} v1.0 in SageMaker Model Registry...")
        time.sleep(0.2)
        print(f"  [MockRegistry] Registered ✅ — Status: Approved")
    else:
        print(f"  ❌ Promotion REJECTED — threshold failures:")
        for f in failures:
            print(f"     • {f}")

    print("\n" + "=" * 60)
    print(f"  {'✅ PIPELINE COMPLETE' if passed else '❌ PIPELINE FAILED — model not promoted'}")
    print("=" * 60)
    print("\nTo run with real GPU/AWS:")
    print("  • Set HF_TOKEN and AWS credentials")
    print("  • Choose config: configs/lora_llama3_8b.yaml")
    print("  • Run: python scripts/train.py --config configs/lora_llama3_8b.yaml")


if __name__ == "__main__":
    main()
