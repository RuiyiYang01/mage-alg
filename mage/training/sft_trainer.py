"""SFT Trainer for the MAGE Learner agent.

Handles:
- Converting Trajectory/SkillDataset into tokenized training data
- LoRA adapter setup
- SFT training loop
- Checkpoint saving/loading

Following EXIF's approach:
- Train from scratch each iteration (cumulative data, fresh LoRA)
- LoRA rank 16-64
- AdamW optimizer with cosine schedule
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from mage.agents.base import SkillDataset, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    device_map: str = "auto"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 200
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1

    # Data
    cumulative: bool = True  # use data from all previous iterations
    lora_strategy: str = "continue"  # "reset" (EXIF-style) or "continue" (prevent forgetting)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SFTConfig:
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        # Coerce numeric fields that YAML may read as strings
        float_fields = {"learning_rate", "weight_decay", "lora_dropout", "warmup_ratio"}
        int_fields = {"lora_rank", "lora_alpha", "max_steps", "batch_size",
                       "gradient_accumulation_steps", "max_seq_length"}
        for k in float_fields:
            if k in filtered and isinstance(filtered[k], str):
                filtered[k] = float(filtered[k])
        for k in int_fields:
            if k in filtered and isinstance(filtered[k], str):
                filtered[k] = int(filtered[k])
        return cls(**filtered)


class TrajectoryDataset(Dataset):
    """Converts MAGE Trajectories into tokenized training examples.

    Each trajectory becomes a conversation:
      System: You are an agent performing tasks...
      User: [instruction] + [observation]
      Assistant: [action]

    For multi-step trajectories, we create one training example per
    (observation, action) pair, with the instruction and history as context.
    """

    def __init__(
        self,
        trajectories: list[Trajectory],
        tokenizer: Any,
        max_seq_length: int = 2048,
        interaction_mode: str = "sequential",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: list[dict[str, Any]] = []

        for traj in trajectories:
            if interaction_mode == "sequential":
                self._process_sequential(traj)
            else:
                self._process_reasoning(traj)

    def _process_sequential(self, traj: Trajectory) -> None:
        """Process a sequential trajectory into training examples.

        Each (observation_t, action_t) pair becomes one training example.
        """
        for t in range(len(traj.actions)):
            # Build context from history
            history_actions = traj.actions[:t]
            current_obs = traj.observations[t] if t < len(traj.observations) else ""

            # Build the conversation
            messages = [
                {
                    "role": "system",
                    "content": "You are an intelligent agent. Follow the instruction and choose the best action based on the current observation.",
                },
                {
                    "role": "user",
                    "content": self._build_user_message(
                        traj.instruction, current_obs, history_actions,
                    ),
                },
                {
                    "role": "assistant",
                    "content": traj.actions[t],
                },
            ]

            self.examples.append({
                "messages": messages,
                "skill_id": traj.skill_id,
                "reward": traj.reward,
            })

    def _process_reasoning(self, traj: Trajectory) -> None:
        """Process a reasoning trajectory into training examples.

        The full reasoning chain becomes one training example.
        """
        if not traj.actions:
            return

        messages = [
            {
                "role": "system",
                "content": "You are an intelligent reasoning agent. Think step by step to answer the question.",
            },
            {
                "role": "user",
                "content": traj.instruction,
            },
            {
                "role": "assistant",
                "content": "\n".join(traj.actions),
            },
        ]

        self.examples.append({
            "messages": messages,
            "skill_id": traj.skill_id,
            "reward": traj.reward,
        })

    def _build_user_message(
        self, instruction: str, observation: str, history: list[str],
    ) -> str:
        parts = [f"Task: {instruction}"]
        if history:
            recent = history[-3:]  # keep last 3 actions for context
            parts.append(f"Previous actions: {', '.join(recent)}")
        if observation:
            parts.append(f"\n{observation}")
        parts.append("\nChoose your next action:")
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        messages = example["messages"]

        # Tokenize using chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # For SFT: labels = input_ids, with padding tokens masked to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Mask everything before the assistant response
        # Find where the assistant content starts
        assistant_content = messages[-1]["content"]
        assistant_tokens = self.tokenizer.encode(assistant_content, add_special_tokens=False)
        if len(assistant_tokens) > 0:
            # Find the position of assistant tokens in the full sequence
            # Simple heuristic: mask all tokens except the last N where N = len(assistant_tokens)
            num_assistant_tokens = min(len(assistant_tokens) + 5, len(labels))  # +5 for template tokens
            mask_end = (attention_mask == 1).sum().item()
            mask_start = max(0, mask_end - num_assistant_tokens)
            labels[:mask_start] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "reward": torch.tensor(example.get("reward", 1.0), dtype=torch.float32),
        }


class SFTTrainer:
    """SFT Trainer for MAGE Learner.

    Handles LoRA setup, training loop, and checkpointing.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Load base model with LoRA adapter."""
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

        # Apply LoRA
        self._apply_lora()

        self._is_loaded = True
        logger.info(f"Model loaded with LoRA (rank={self.config.lora_rank})")

    def _apply_lora(self) -> None:
        """Apply fresh LoRA adapter to the model."""
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
        logger.info(f"LoRA: {trainable:,} trainable params / {total:,} total ({100*trainable/total:.2f}%)")

    def reset_lora(self) -> None:
        """Reset LoRA adapter to fresh weights (for train-from-scratch-each-iteration)."""
        if self._model is None:
            return

        # Remove existing adapter and apply fresh one
        from peft import PeftModel
        if isinstance(self._model, PeftModel):
            self._model = self._model.unload()
        self._apply_lora()
        logger.info("LoRA adapter reset to fresh weights")

    def train(
        self,
        dataset: SkillDataset,
        iteration: int,
        output_dir: str | Path | None = None,
        interaction_mode: str = "sequential",
        reward_weighted: bool = False,
    ) -> dict[str, Any]:
        """Run SFT training on the skill dataset.

        Args:
            dataset: Trajectories to train on
            iteration: Current evolution iteration
            output_dir: Where to save checkpoint
            interaction_mode: "sequential" or "reasoning"
            reward_weighted: If True, weight each example's loss by its reward

        Returns:
            Training metrics dict
        """
        self.load_model()

        # LoRA reset strategy:
        # - "reset": EXIF-style, reset LoRA each iteration (train from scratch with cumulative data)
        # - "continue": keep previous LoRA, continue training (prevents forgetting)
        lora_strategy = self.config.lora_strategy if hasattr(self.config, 'lora_strategy') else "continue"
        if lora_strategy == "reset" or iteration == 0:
            self.reset_lora()
            logger.info(f"LoRA reset (strategy={lora_strategy}, iter={iteration})")
        else:
            logger.info(f"LoRA continued from previous iteration (strategy={lora_strategy})")

        # Build training dataset
        train_dataset = TrajectoryDataset(
            dataset.trajectories,
            self._tokenizer,
            max_seq_length=self.config.max_seq_length,
            interaction_mode=interaction_mode,
        )

        if len(train_dataset) == 0:
            logger.warning("Empty training dataset — skipping training")
            return {"iteration": iteration, "status": "skipped", "reason": "empty_dataset"}

        logger.info(f"Training on {len(train_dataset)} examples (iteration {iteration})")

        # DataLoader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        total_steps = min(
            self.config.max_steps,
            len(dataloader) * self.config.gradient_accumulation_steps,
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # Training loop
        self._model.train()
        global_step = 0
        total_loss = 0.0
        log_losses: list[float] = []

        for batch in dataloader:
            if global_step >= self.config.max_steps:
                break

            # Extract reward weights before sending to model
            rewards = batch.pop("reward").to(self._model.device)

            # Move to device
            batch = {k: v.to(self._model.device) for k, v in batch.items()}

            if reward_weighted:
                # Compute per-example loss manually for reward weighting
                outputs = self._model(**batch)
                logits = outputs.logits
                labels = batch["labels"]
                # Shift for causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                per_token_loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                # Reshape to (batch, seq_len) and mean per example
                per_token_loss = per_token_loss.view(shift_labels.size())
                valid_tokens = (shift_labels != -100).float()
                per_example_loss = (per_token_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1)
                # Weight by reward
                loss = (per_example_loss * rewards).mean() / self.config.gradient_accumulation_steps
            else:
                outputs = self._model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

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
                avg_loss = sum(log_losses[-50:]) / len(log_losses[-50:])
                logger.info(f"  Step {global_step}/{self.config.max_steps}: loss={avg_loss:.4f}")

        self._model.eval()
        avg_loss = total_loss / max(global_step, 1)

        # Save checkpoint
        if output_dir:
            self.save_checkpoint(output_dir)

        metrics = {
            "iteration": iteration,
            "status": "completed",
            "num_examples": len(train_dataset),
            "num_steps": global_step,
            "avg_loss": avg_loss,
            "final_loss": log_losses[-1] if log_losses else 0.0,
        }
        logger.info(f"Training complete: {global_step} steps, avg_loss={avg_loss:.4f}")
        return metrics

    def generate(self, prompt: str, instruction: str = "", max_new_tokens: int = 64) -> str:
        """Generate a response using the current model."""
        self.load_model()
        self._model.eval()

        messages = [
            {"role": "system", "content": "You are an intelligent agent. Follow the instruction and choose the best action."},
            {"role": "user", "content": prompt if not instruction else f"Task: {instruction}\n\n{prompt}\n\nChoose your next action:"},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save LoRA adapter checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        from peft import PeftModel
        if isinstance(self._model, PeftModel):
            self._model.save_pretrained(path / "lora_adapter")
            logger.info(f"Saved LoRA checkpoint to {path / 'lora_adapter'}")

        # Save config
        with open(path / "training_config.json", "w") as f:
            json.dump({
                "model_name": self.config.model_name,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
            }, f, indent=2)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load LoRA adapter from checkpoint."""
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
