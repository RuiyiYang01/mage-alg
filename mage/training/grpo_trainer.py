"""GRPO Trainer for the MAGE Learner agent.

Group Relative Policy Optimization:
- For each question, sample K completions from the current policy
- Score each with reward function (F1 for QA)
- Compute advantage: (reward_i - mean) / std per group
- Policy gradient loss: -advantage * log_prob(completion)

No reference model needed — the group mean serves as baseline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    load_in_4bit: bool = True
    device_map: str = "auto"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # GRPO
    num_samples: int = 4
    temperature: float = 0.7
    max_new_tokens: int = 512
    normalize_advantages: bool = True

    # Training
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_steps: int = 200
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    lora_strategy: str = "continue"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GRPOConfig:
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        float_fields = {"learning_rate", "weight_decay", "lora_dropout",
                        "warmup_ratio", "temperature"}
        int_fields = {"lora_rank", "lora_alpha", "max_steps", "batch_size",
                      "gradient_accumulation_steps", "max_seq_length",
                      "num_samples", "max_new_tokens"}
        for k in float_fields:
            if k in filtered and isinstance(filtered[k], str):
                filtered[k] = float(filtered[k])
        for k in int_fields:
            if k in filtered and isinstance(filtered[k], str):
                filtered[k] = int(filtered[k])
        return cls(**filtered)


class GRPOCompletionDataset(Dataset):
    """Dataset of (question, completion, advantage) for GRPO training."""

    def __init__(
        self,
        questions: list[dict[str, Any]],
        completions: list[list[str]],
        advantages: list[list[float]],
        tokenizer: Any,
        max_seq_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: list[dict[str, Any]] = []

        for q_idx, question in enumerate(questions):
            for c_idx in range(len(completions[q_idx])):
                messages = [
                    {"role": "system", "content": "You are an intelligent reasoning agent. Think step by step to answer the question."},
                    {"role": "user", "content": question["instruction"]},
                    {"role": "assistant", "content": completions[q_idx][c_idx]},
                ]
                self.examples.append({
                    "messages": messages,
                    "advantage": advantages[q_idx][c_idx],
                })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        messages = example["messages"]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        encodings = self.tokenizer(
            text, truncation=True, max_length=self.max_seq_length,
            padding="max_length", return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Mask everything before assistant response
        assistant_content = messages[-1]["content"]
        assistant_tokens = self.tokenizer.encode(assistant_content, add_special_tokens=False)
        if assistant_tokens:
            num_assistant_tokens = min(len(assistant_tokens) + 5, len(labels))
            mask_end = (attention_mask == 1).sum().item()
            mask_start = max(0, mask_end - num_assistant_tokens)
            labels[:mask_start] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantage": torch.tensor(example["advantage"], dtype=torch.float32),
        }


class GRPOTrainer:
    """GRPO Trainer — samples completions, computes advantages, trains via policy gradient."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    # -- Model management (same pattern as SFTTrainer) --

    def load_model(self) -> None:
        if self._is_loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.config.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"device_map": self.config.device_map}
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **load_kwargs,
        )
        self._apply_lora()
        self._is_loaded = True
        logger.info(f"Model loaded with LoRA (rank={self.config.lora_rank})")

    def _apply_lora(self) -> None:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
        self._model = get_peft_model(self._model, lora_config)
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    def reset_lora(self) -> None:
        if self._model is None:
            return
        from peft import PeftModel
        if isinstance(self._model, PeftModel):
            self._model = self._model.unload()
        self._apply_lora()
        logger.info("LoRA adapter reset to fresh weights")

    # -- GRPO-specific methods --

    def sample_completions(
        self,
        instruction: str,
        context: str = "",
        num_samples: int = 4,
    ) -> list[str]:
        """Sample K completions from current policy."""
        self.load_model()
        self._model.eval()

        if context:
            user_content = (
                f"Answer the following question based on the provided context. "
                f"Think step by step.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {instruction}\n\n"
                f"Provide your reasoning, then give your final answer on a new line "
                f"starting with 'Answer:'"
            )
        else:
            user_content = instruction

        messages = [
            {"role": "system", "content": "You are an intelligent reasoning agent. Think step by step to answer the question."},
            {"role": "user", "content": user_content},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        completions = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
                generated = outputs[0][inputs["input_ids"].shape[1]:]
                completion = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
                completions.append(completion)

        return completions

    def train(
        self,
        questions: list[dict[str, Any]],
        completions: list[list[str]],
        rewards: list[list[float]],
        iteration: int,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run GRPO training."""
        self.load_model()

        # LoRA strategy
        if self.config.lora_strategy == "reset" or iteration == 0:
            self.reset_lora()
            logger.info(f"LoRA reset (strategy={self.config.lora_strategy}, iter={iteration})")
        else:
            logger.info("LoRA continued from previous iteration")

        # Compute advantages per group
        advantages = []
        for reward_group in rewards:
            mean_r = sum(reward_group) / len(reward_group) if reward_group else 0.0
            if self.config.normalize_advantages and len(reward_group) > 1:
                std_r = (sum((r - mean_r) ** 2 for r in reward_group) / len(reward_group)) ** 0.5
                std_r = max(std_r, 1e-8)
                group_adv = [(r - mean_r) / std_r for r in reward_group]
            else:
                group_adv = [r - mean_r for r in reward_group]
            advantages.append(group_adv)

        # Build dataset
        train_dataset = GRPOCompletionDataset(
            questions=questions,
            completions=completions,
            advantages=advantages,
            tokenizer=self._tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        if len(train_dataset) == 0:
            logger.warning("Empty training dataset — skipping")
            return {"iteration": iteration, "status": "skipped", "reason": "empty_dataset"}

        logger.info(
            f"GRPO training: {len(questions)} questions x K={self.config.num_samples} "
            f"= {len(train_dataset)} examples (iteration {iteration})"
        )

        # DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size * self.config.num_samples,
            shuffle=True,
            drop_last=False,
        )

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = min(self.config.max_steps, len(dataloader))
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

        # Training loop
        self._model.train()
        global_step = 0
        total_loss = 0.0
        log_losses: list[float] = []

        for batch in dataloader:
            if global_step >= self.config.max_steps:
                break

            adv = batch.pop("advantage").to(self._model.device)
            batch = {k: v.to(self._model.device) for k, v in batch.items()}

            # Compute per-example log probs
            outputs = self._model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)

            valid_mask = (shift_labels != -100).float()
            per_example_log_prob = (token_log_probs * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

            # GRPO loss: -advantage * log_prob
            loss = -(adv * per_example_log_prob).mean() / self.config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.config.gradient_accumulation_steps

            if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            log_losses.append(loss.item() * self.config.gradient_accumulation_steps)

            if global_step % 50 == 0:
                avg = sum(log_losses[-50:]) / len(log_losses[-50:])
                logger.info(f"  Step {global_step}/{self.config.max_steps}: grpo_loss={avg:.4f}")

        self._model.eval()
        avg_loss = total_loss / max(global_step, 1)

        if output_dir:
            self.save_checkpoint(output_dir)

        all_rewards = [r for group in rewards for r in group]
        metrics = {
            "iteration": iteration,
            "status": "completed",
            "num_questions": len(questions),
            "num_completions": len(train_dataset),
            "num_steps": global_step,
            "avg_grpo_loss": avg_loss,
            "final_loss": log_losses[-1] if log_losses else 0.0,
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        }
        logger.info(f"GRPO complete: {global_step} steps, loss={avg_loss:.4f}, mean_reward={metrics['mean_reward']:.3f}")
        return metrics

    # -- Inference & checkpointing --

    def generate(self, prompt: str, instruction: str = "", max_new_tokens: int = 512) -> str:
        self.load_model()
        self._model.eval()

        messages = [
            {"role": "system", "content": "You are an intelligent reasoning agent. Think step by step to answer the question."},
            {"role": "user", "content": prompt if not instruction else f"Task: {instruction}\n\n{prompt}"},
        ]

        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.3, do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        from peft import PeftModel
        if isinstance(self._model, PeftModel):
            self._model.save_pretrained(path / "lora_adapter")
            logger.info(f"Saved LoRA checkpoint to {path / 'lora_adapter'}")
        with open(path / "training_config.json", "w") as f:
            json.dump({"model_name": self.config.model_name, "trainer_type": "grpo",
                        "lora_rank": self.config.lora_rank, "num_samples": self.config.num_samples}, f, indent=2)

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        adapter_path = path / "lora_adapter"
        if not adapter_path.exists():
            logger.warning(f"No LoRA checkpoint at {adapter_path}")
            return
        self.load_model()
        from peft import PeftModel
        if isinstance(self._model, PeftModel):
            self._model = self._model.unload()
        self._model = PeftModel.from_pretrained(self._model, str(adapter_path))
        logger.info(f"Loaded LoRA checkpoint from {adapter_path}")

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer
