"""
Training script for Soft Prompt Injector / Dual Stream Gate.

This script teaches the "Valve" (Gate) and "Translator" (Projector) 
how to inject signals effectively.

Strategy:
    1. Freeze the main LLM (Qwen).
    2. Train ONLY the SoftPromptInjector parameters.
    3. Task: "Injection Recovery"
       Input: "The status is [MASK]"
       Signal: "Critical"
       Target: "Critical"
       
    The model MUST use the subconscious signal to predict the next word,
    because the text prompt gives no clue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from tqdm import tqdm
from typing import List, Tuple

from .soft_prompt_generator import SoftPromptInjector, create_native_encoder

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Use small model for training
BATCH_SIZE = 8
LEARNING_RATE = 5e-4  # Lower LR for better convergence
NUM_EPOCHS = 5  # More epochs for diverse data
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TO_INJECT = 12  # Middle layer

class InjectionDataset(Dataset):
    """
    Dataset that teaches the adapter to respond to signals.
    
    Three types of samples:
    1. Fill-in-blank: "Status is" + signal "online" → predict "online"
    2. Override: "peaceful day" + signal "DANGER" → predict "However" or "But"
    3. Topic switch: "nice weather" + signal "URGENT" → predict "urgent"
    """
    def __init__(self, size: int = 1000):
        self.size = size
        
        # Type 1: Simple fill-in-blank
        self.fill_in = [
            ("The system status is", "online", "online"),
            ("The system status is", "offline", "offline"),
            ("Current weather:", "sunny", "sunny"),
            ("Current weather:", "rainy", "rainy"),
            ("Alert level:", "green", "green"),
            ("Alert level:", "red", "red"),
            ("Battery is", "full", "full"),
            ("Battery is", "empty", "empty"),
        ]
        
        # Type 2: Override/Interrupt patterns - signal should cause interruption
        self.override = [
            ("Everything is calm and peaceful", "DANGER", "However"),
            ("The day was going smoothly", "ALERT", "But"),
            ("All systems operating normally", "WARNING", "Wait"),
            ("Nothing unusual to report", "URGENT", "Actually"),
            ("The office was quiet", "FIRE", "Suddenly"),
            ("A beautiful sunny day", "EMERGENCY", "However"),
            ("Work proceeded as usual", "CRITICAL", "But"),
            ("The building was peaceful", "EVACUATE", "Wait"),
        ]
        
        # Type 3: Topic injection - signal word should appear
        self.topic = [
            ("Let me tell you about", "safety", "safety"),
            ("The most important thing is", "fire", "fire"),
            ("We need to discuss", "emergency", "emergency"),
            ("I want to mention", "danger", "danger"),
            ("First, let's talk about", "evacuation", "evacuation"),
        ]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Weighted random selection (more overrides to learn interruption)
        r = random.random()
        if r < 0.5:  # 50% override patterns
            prompt, signal, target = random.choice(self.override)
        elif r < 0.8:  # 30% fill-in
            prompt, signal, target = random.choice(self.fill_in)
        else:  # 20% topic
            prompt, signal, target = random.choice(self.topic)
        return prompt, signal, target

def train():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map=DEVICE,
        trust_remote_code=True
    )
    
    # Freeze model
    print("Freezing base model...")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Create Injector (The only thing we will train)
    hidden_dim = model.config.hidden_size
    injector = SoftPromptInjector(hidden_dim, gate_init=0.1).to(DEVICE, dtype=torch.bfloat16)
    injector.train()
    
    # Create Signal Encoder (Frozen)
    encoder = create_native_encoder(model, tokenizer)
    encoder.to(DEVICE, dtype=torch.bfloat16)
    
    # Optimizer - only for injector
    optimizer = optim.AdamW(injector.parameters(), lr=LEARNING_RATE)
    
    # Dataset
    dataset = InjectionDataset(size=2000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("\nStarting training of the Gate/Injector...")
    print(f"Injecting at layer {LAYER_TO_INJECT}")
    
    # Get target layer for hook
    if hasattr(model, "model"):
        target_layer = model.model.layers[LAYER_TO_INJECT]
    else:
        target_layer = model.transformer.h[LAYER_TO_INJECT]

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for prompts, signals, targets_text in pbar:
            # 1. Prepare Inputs
            # We want to predict the target word immediately after the prompt
            inputs = tokenizer(list(prompts), return_tensors="pt", padding=True).to(DEVICE)
            targets = tokenizer(list(targets_text), return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
            
            # 2. Encode Signals (Native Encoder - No Grad)
            with torch.no_grad():
                signal_embeddings = encoder(list(signals)) # [batch, slots, hidden]
                signal_embeddings = signal_embeddings.to(dtype=torch.bfloat16)
            
            # 3. Hook Setup
            # We need to capture the hidden state, inject, and put it back
            # Since we can't easily hook inside a .forward() for training without rewriting,
            # we will assume a simplified forward pass or use a hook that allows gradients.
            
            # TRICK: Torch hooks allow gradients to flow back!
            def train_hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                
                # Inject!
                # Note: We use the OUTER 'injector' variable which tracks gradients
                injected = injector(h, signal_embeddings)
                
                if isinstance(output, tuple):
                    return (injected,) + output[1:]
                return injected

            hook_handle = target_layer.register_forward_hook(train_hook)
            
            # 4. Forward Pass
            # We append the signal text to inputs just to get the target IDs aligned,
            # BUT the model shouldn't 'see' it in the prompt. 
            # Actually, for next-token prediction:
            # Input: "The status is"
            # Target Label for last token: "online"
            
            # Let's just run the prompt
            outputs = model(**inputs)
            
            # Remove hook immediately
            hook_handle.remove()
            
            # 5. Compute Loss
            # We want the LAST token of the prompt to predict the FIRST token of the signal
            logits = outputs.logits[:, -1, :] # [batch, vocab]
            target_ids = targets.input_ids[:, 0] # First token of signal
            
            loss = F.cross_entropy(logits, target_ids)
            
            # 6. Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader):.4f}")
        
    print("\nTraining Complete!")
    print("Saving adapter weights...")
    torch.save(injector.state_dict(), "soft_prompt_adapter.pt")
    print("Saved to soft_prompt_adapter.pt")

if __name__ == "__main__":
    train()
