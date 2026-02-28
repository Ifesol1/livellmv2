"""
Training Script for Dual-Stream Qwen3-4B

This script trains the cross-attention gates to respond to live signals.
The base model weights are frozen, only the new dual-stream components are trained.
"""

import torch
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Stream Qwen3")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model to use (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--data-path", type=str, default="data/training_data_converted.json",
                        help="Path to training data JSON")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save-every", type=int, default=200,
                        help="Save checkpoint every N steps")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit quantization (saves VRAM)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--freeze-base", action="store_true", default=True,
                        help="Freeze base model weights (only train dual-stream layers)")
    parser.add_argument("--injection-probability", type=float, default=0.8,
                        help="Probability of injecting a live signal into a sample")
    parser.add_argument("--max-injections-per-sample", type=int, default=2,
                        help="Maximum number of live injections per sample")
    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Training requires GPU.")
        return 1

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Import after CUDA check
    from dual_stream import DualStreamTransformer
    from dual_stream.training import (
        DualStreamTrainer,
        DynamicLiveDataset,
        load_grok_generated_data,
    )
    from transformers import AutoTokenizer

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load model with surgery
    logger.info("="*60)
    logger.info(f"Step 1: Loading {args.model} with Dual-Stream Surgery")
    logger.info("="*60)

    # Use bfloat16 for stability
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using dtype: {torch_dtype}")

    model = DualStreamTransformer.from_pretrained_with_surgery(
        model_path=args.model,
        device="cuda",
        torch_dtype=torch_dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    logger.info(f"Model loaded. Live injection layers: {sorted(model.live_layers)}")

    # Step 2: Freeze base model if requested
    if args.freeze_base:
        logger.info("Freezing base model weights (only training dual-stream layers)")
        trainable_params = 0
        frozen_params = 0
        
        for name, param in model.named_parameters():
            # Only train: cross-attention layers, gates, and live encoder
            if any(x in name.lower() for x in ['cross', 'gate', 'live']):
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        logger.info(f"Trainable params: {trainable_params/1e6:.2f}M")
        logger.info(f"Frozen params: {frozen_params/1e6:.2f}M")

    # Step 3: Load tokenizer
    logger.info("="*60)
    logger.info("Step 2: Loading Tokenizer")
    logger.info("="*60)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Step 4: Load training data
    logger.info("="*60)
    logger.info("Step 3: Loading Training Data")
    logger.info("="*60)

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        return 1

    base_samples, signal_pool = load_grok_generated_data(str(data_path))
    logger.info(f"Loaded {len(base_samples)} samples, {len(signal_pool)} signals")

    # Step 5: Create dataset
    dataset = DynamicLiveDataset(
        base_samples=base_samples,
        live_signals=signal_pool,
        tokenizer=tokenizer,
        max_length=256,  # Short sequences to save memory
        injection_probability=args.injection_probability,
        max_injections_per_sample=args.max_injections_per_sample,
    )

    # Step 6: Create trainer
    logger.info("="*60)
    logger.info("Step 4: Creating Trainer")
    logger.info("="*60)

    # Clear memory before creating trainer
    torch.cuda.empty_cache()
    
    # Disable mixed precision - model is already in float16, GradScaler doesn't work
    use_mixed_precision = False
    
    trainer = DualStreamTrainer(
        model=model,
        train_dataset=dataset,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        device="cuda",
        mixed_precision=use_mixed_precision,
    )

    logger.info(f"Trainer ready:")
    logger.info(f"  - Max steps: {args.max_steps}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation}")
    logger.info(f"  - Learning rate: {args.learning_rate}")

    # Step 7: Train!
    logger.info("="*60)
    logger.info("Step 5: Starting Training")
    logger.info("="*60)

    try:
        trainer.train(
            log_every=args.log_every,
            save_every=args.save_every,
            save_path=str(output_dir),
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(str(output_dir))
        logger.info("Checkpoint saved")

    # Step 8: Final save and summary
    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)

    trainer.save_checkpoint(str(output_dir))

    # Show final gate values
    gate_info = model.get_layer_gate_info()
    logger.info("Final gate values:")
    for layer_idx, info in sorted(gate_info.items()):
        logger.info(f"  Layer {layer_idx}: {info['effective_gate']:.4f} ({info['influence_percent']:.1f}%)")

    logger.info(f"\nCheckpoints saved to: {output_dir}")
    logger.info("To load the trained model:")
    logger.info(f"  checkpoint = torch.load('{output_dir}/checkpoint_best.pt')")
    logger.info("  model.load_state_dict(checkpoint['model_state_dict'])")

    return 0


if __name__ == "__main__":
    exit(main())
