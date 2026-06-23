"""
Phase 3: Subspace Extraction (Contrastive Activation SVD)
Extracts safety directions by finding the contrastive difference between
activations of harmful and safe prompts, then applying SVD.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to sys path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset_loader import load_advbench, load_safe_prompts

def extract_safety_directions():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading base model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use float16 for GPU, float32 for CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    print("Loading prompts...")
    harmful_prompts = load_advbench()
    safe_prompts = load_safe_prompts()
    
    # Verify counts
    assert len(harmful_prompts) == len(safe_prompts), "Mismatch in prompt counts!"
    
    num_layers = model.config.num_hidden_layers
    
    # Storage dictionaries
    # Format: dict[layer_idx] -> dict[proj_name] -> list of tensors of shape (d_model,)
    harmful_activations = {i: {"q_proj": [], "v_proj": []} for i in range(num_layers)}
    safe_activations = {i: {"q_proj": [], "v_proj": []} for i in range(num_layers)}

    current_storage = None # Will point to harmful_activations or safe_activations

    def shared_hook(layer_idx, proj_name):
        def hook(module, inp, out):
            if current_storage is not None:
                # inp[0] shape: (batch_size, seq_len, d_model)
                # We want the activation at the last token
                last_token_act = inp[0][:, -1, :].detach().float().cpu()
                current_storage[layer_idx][proj_name].append(last_token_act)
        return hook

    print("Registering forward hooks on q_proj and v_proj...")
    handles = []
    for i in range(num_layers):
        layer = model.model.layers[i]
        handles.append(layer.self_attn.q_proj.register_forward_hook(shared_hook(i, "q_proj")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(shared_hook(i, "v_proj")))

    # Process prompts one by one
    def process_prompts(prompts, desc):
        for prompt in tqdm(prompts, desc=desc):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)

    print("Extracting harmful activations...")
    current_storage = harmful_activations
    process_prompts(harmful_prompts, "Harmful")

    print("Extracting safe activations...")
    current_storage = safe_activations
    process_prompts(safe_prompts, "Safe")

    print("Removing hooks...")
    for h in handles:
        h.remove()

    print("Computing SVD and extracting top-5 safety directions...")
    safety_directions = {}
    k = 5
    
    for i in tqdm(range(num_layers), desc="SVD Layer"):
        for proj in ["q_proj", "v_proj"]:
            # Stack all activations for this layer and projection
            H = torch.cat(harmful_activations[i][proj], dim=0) # shape: (520, d_model)
            S = torch.cat(safe_activations[i][proj], dim=0)    # shape: (520, d_model)
            
            diff = H - S # Contrastive difference
            
            # Center the difference matrix
            diff_mean = diff.mean(dim=0, keepdim=True)
            diff_centered = diff - diff_mean
            
            # Run SVD
            # diff_centered = U @ S_val @ Vh
            # Vh shape is (min(N, d_model), d_model)
            U, S_val, Vh = torch.linalg.svd(diff_centered, full_matrices=False)
            
            # We want top k right singular vectors
            directions = Vh[:k, :].T # Transpose to shape (d_model, k)
            
            # L2 Normalize the directions
            directions = directions / torch.norm(directions, dim=0, keepdim=True)
            
            key = f"model.layers.{i}.self_attn.{proj}"
            safety_directions[key] = directions

    # Save
    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "safety_directions.pt"
    
    torch.save(safety_directions, out_path)
    print(f"\nSuccess! Extracted directions saved to {out_path}")
    print("Tensor shape for each layer:", list(safety_directions.values())[0].shape)

if __name__ == "__main__":
    extract_safety_directions()
