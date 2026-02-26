"""
Automated evaluation harness for fine-tuned LLMs.
Uses Ragas metrics + custom domain benchmarks before model promotion.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    # Thresholds for production promotion gate
    faithfulness_threshold: float = 0.82
    answer_relevancy_threshold: float = 0.78
    domain_accuracy_threshold: float = 0.97
    max_new_tokens: int = 512
    batch_size: int = 8


@dataclass
class EvalResult:
    model_id: str
    ragas_scores: Dict[str, float] = field(default_factory=dict)
    domain_accuracy: float = 0.0
    passed_gate: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "ragas_scores": self.ragas_scores,
            "domain_accuracy": self.domain_accuracy,
            "passed_gate": self.passed_gate,
            "failure_reasons": self.failure_reasons,
        }


class FineTunedModelEvaluator:
    """
    Evaluation harness for LoRA/PEFT fine-tuned models.
    Runs Ragas + domain-specific benchmarks and enforces promotion gates.
    """

    def __init__(
        self,
        base_model_id: str,
        adapter_path: str,
        config: Optional[EvalConfig] = None,
    ):
        self.config = config or EvalConfig()
        logger.info(f"Loading model: base={base_model_id}, adapter={adapter_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.max_new_tokens,
        )

    def generate_answers(self, questions: List[str]) -> List[str]:
        answers = []
        for q in questions:
            prompt = f"### Instruction:\n{q}\n\n### Response:\n"
            output = self.pipe(prompt, do_sample=False)[0]["generated_text"]
            answer = output.split("### Response:\n")[-1].strip()
            answers.append(answer)
        return answers

    def run_ragas_eval(
        self, questions: List[str], contexts: List[List[str]], ground_truths: List[str]
    ) -> Dict[str, float]:
        answers = self.generate_answers(questions)
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        scores = {
            "faithfulness": float(results["faithfulness"]),
            "answer_relevancy": float(results["answer_relevancy"]),
            "context_precision": float(results["context_precision"]),
        }
        logger.info(f"Ragas scores: {scores}")
        return scores

    def run_domain_benchmark(self, benchmark_path: str) -> float:
        """
        Run domain-specific benchmark (instruction → expected output matching).
        Returns accuracy as fraction of correct responses.
        """
        with open(benchmark_path) as f:
            benchmark = json.load(f)

        correct = 0
        for item in benchmark:
            answer = self.generate_answers([item["instruction"]])[0]
            if self._check_answer(answer, item["expected"], item.get("match_type", "contains")):
                correct += 1

        accuracy = correct / len(benchmark) if benchmark else 0.0
        logger.info(f"Domain benchmark accuracy: {accuracy:.4f} ({correct}/{len(benchmark)})")
        return accuracy

    def _check_answer(self, answer: str, expected: str, match_type: str) -> bool:
        if match_type == "exact":
            return answer.strip().lower() == expected.strip().lower()
        elif match_type == "contains":
            return expected.lower() in answer.lower()
        return False

    def evaluate_and_gate(
        self,
        questions: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
        benchmark_path: Optional[str] = None,
        model_id: str = "unknown",
    ) -> EvalResult:
        result = EvalResult(model_id=model_id)

        result.ragas_scores = self.run_ragas_eval(questions, contexts, ground_truths)

        if benchmark_path and Path(benchmark_path).exists():
            result.domain_accuracy = self.run_domain_benchmark(benchmark_path)

        # Apply promotion gate
        failures = []
        if result.ragas_scores.get("faithfulness", 0) < self.config.faithfulness_threshold:
            failures.append(f"faithfulness {result.ragas_scores['faithfulness']:.3f} < {self.config.faithfulness_threshold}")
        if result.ragas_scores.get("answer_relevancy", 0) < self.config.answer_relevancy_threshold:
            failures.append(f"answer_relevancy {result.ragas_scores['answer_relevancy']:.3f} < {self.config.answer_relevancy_threshold}")
        if benchmark_path and result.domain_accuracy < self.config.domain_accuracy_threshold:
            failures.append(f"domain_accuracy {result.domain_accuracy:.3f} < {self.config.domain_accuracy_threshold}")

        result.failure_reasons = failures
        result.passed_gate = len(failures) == 0

        status = "PASSED" if result.passed_gate else "FAILED"
        logger.info(f"Promotion gate: {status}. Failures: {failures or 'none'}")

        output_path = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/tmp")) / "eval_result.json"
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result
