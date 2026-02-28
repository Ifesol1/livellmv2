"""
Quick demo training to open gates for live signal response on Qwen3-0.6B.

This keeps VRAM low and runs a short synthetic finetune so that the model
will visibly react to injected signals during generation.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from dual_stream import DualStreamTransformer
from dual_stream.training import (
    DualStreamTrainer,
    DynamicLiveDataset,
    create_synthetic_training_data,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    logger.info(f"Using device={device}, dtype={torch_dtype}")

    # 1) Load small Qwen3 backbone with dual-stream surgery
    model = DualStreamTransformer.from_pretrained_with_surgery(
        model_path="Qwen/Qwen3-0.6B",
        device=device,
        torch_dtype=torch_dtype,
        load_in_4bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Build synthetic training data with guaranteed live injections
    base_samples, signal_pool = create_synthetic_training_data(num_samples=200, signal_pool_size=32)
    train_dataset = DynamicLiveDataset(
        base_samples=base_samples,
        live_signals=signal_pool,
        tokenizer=tokenizer,
        max_length=256,
        injection_probability=1.0,  # always inject for this demo
        max_injections_per_sample=2,
    )

    # 3) Trainer (freeze base weights implicitly via smaller LR groups)
    trainer = DualStreamTrainer(
        model=model,
        train_dataset=train_dataset,
        learning_rate=2e-5,
        max_steps=300,
        batch_size=2,
        gradient_accumulation_steps=4,
        device=device,
        mixed_precision=True,
    )

    # 4) Train briefly and save
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True, parents=True)
    trainer.train(log_every=10, save_every=150, save_path=str(save_dir))

    # Final checkpoint and gate snapshot
    trainer.save_checkpoint(str(save_dir), is_best=True)
    gate_info = model.get_layer_gate_info()
    logger.info(f"Final gate snapshot: {gate_info}")

    logger.info("Quick demo training complete. Checkpoints saved in ./checkpoints")


if __name__ == "__main__":
    main()


