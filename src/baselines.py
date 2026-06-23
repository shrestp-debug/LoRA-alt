"""
Phase 4: Static Safety Baselines (SafeLoRA and SaLoRA)
"""

import torch
import contextlib
import logging
import random

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def safelora_eval_context(model, safety_directions):
    """
    Temporarily merges the SafeLoRA projection into the base weights for evaluation,
    then reverts the base weights to continue training.
    
    Formula:
      ΔW = (B @ A) * scaling
      ΔW_safe = ΔW - (ΔW @ V) @ V^T
      W_eval = W_base + ΔW_safe
    """
    logger.info("[SafeLoRA] Entering periodic evaluation context...")
    
    # Store the deltas to subtract later
    deltas = {}
    norm_ratios = []
    
    for name, module in model.named_modules():
        # Identify PEFT lora layers
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            adapter_name = "default"
            if adapter_name not in module.lora_A:
                continue
                
            # Match the layer name to the keys in safety_directions
            match_key = None
            for k in safety_directions.keys():
                if k in name:
                    match_key = k
                    break
            
            if match_key is None:
                continue
                
            A = module.lora_A[adapter_name].weight # shape: (r, d_in)
            B = module.lora_B[adapter_name].weight # shape: (d_out, r)
            scaling = module.scaling[adapter_name]
            
            # Compute full cumulative delta
            delta_W = (B @ A) * scaling # shape: (d_out, d_in)
            
            # Match V to the dtype and device of delta_W (which is likely float32)
            V = safety_directions[match_key].to(device=delta_W.device, dtype=delta_W.dtype)
            
            # Project delta_W orthogonal to V
            # delta_W_safe = delta_W - (delta_W @ V) @ V^T
            proj = torch.matmul(torch.matmul(delta_W, V), V.T)
            delta_W_safe = delta_W - proj
            
            # Log magnitude
            norm_before = torch.norm(delta_W).item()
            norm_after = torch.norm(delta_W_safe).item()
            ratio = norm_after / norm_before if norm_before > 0 else 0.0
            norm_ratios.append(ratio)
            
            deltas[name] = delta_W_safe
            
            # Temporarily merge into base weight
            module.base_layer.weight.data += delta_W_safe
            
    avg_ratio = sum(norm_ratios) / len(norm_ratios) if norm_ratios else 0.0
    logger.info(f"[SafeLoRA] Projected weight norm ratio ||ΔW_safe|| / ||ΔW||: {avg_ratio:.4f}")
    
    # Disable adapters so they don't add to the merged weights during forward pass
    with model.disable_adapter():
        yield
        
    # Revert base weights back to standard training state
    for name, module in model.named_modules():
        if name in deltas:
            module.base_layer.weight.data -= deltas[name]
    
    logger.info("[SafeLoRA] Exited evaluation context, restored base weights.")


def apply_salora(model, safety_directions):
    """
    Applies a fixed forward pre-hook to the LoRA branch to project the input 
    orthogonal to the safety subspace at every forward pass.
    
    Formula:
      lora_input = x - (x @ V) @ V^T
    """
    logger.info("[SaLoRA] Applying fixed forward filters...")
    hooks = []
    
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            adapter_name = "default"
            if adapter_name not in module.lora_A:
                continue
                
            match_key = None
            for k in safety_directions.keys():
                if k in name:
                    match_key = k
                    break
            
            if match_key is None:
                continue
                
            V_base = safety_directions[match_key]
            
            lora_A = module.lora_A[adapter_name]
            
            # The hook modifies the input `x` before it hits `lora_A`
            def pre_hook(mod, args, V_base=V_base):
                x = args[0] # shape: (batch, seq, d_in)
                
                # Match V to the dtype and device of x dynamically
                V = V_base.to(device=x.device, dtype=x.dtype)
                
                # S(x) = x - (x @ V) @ V^T
                x_V = torch.matmul(x, V) # (batch, seq, k)
                x_proj = x - torch.matmul(x_V, V.T)
                
                # Periodic magnitude logging to verify the 99% threshold
                if not hasattr(mod, "salora_stats"):
                    mod.salora_stats = []
                
                if random.random() < 0.001: # Sample sparsely to save compute
                    norm_x_V = torch.norm(x_V).item()
                    x_proj_V = torch.matmul(x_proj, V)
                    norm_x_proj_V = torch.norm(x_proj_V).item()
                    
                    mod.salora_stats.append({
                        "original_safety_norm": norm_x_V,
                        "projected_safety_norm": norm_x_proj_V,
                        "ratio": norm_x_proj_V / norm_x_V if norm_x_V > 0 else 0
                    })
                    
                return (x_proj,)
                
            hook_handle = lora_A.register_forward_pre_hook(pre_hook)
            hooks.append(hook_handle)
            
    logger.info(f"[SaLoRA] Successfully applied {len(hooks)} input projection hooks.")
    return hooks

def log_salora_stats(model):
    """
    Aggregates and logs the SaLoRA projection magnitude stats to verify 
    the safety component is reduced by >99%.
    """
    all_stats = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            adapter_name = "default"
            if adapter_name in module.lora_A and hasattr(module.lora_A[adapter_name], "salora_stats"):
                all_stats.extend(module.lora_A[adapter_name].salora_stats)
                
    if not all_stats:
        return
        
    avg_ratio = sum(s["ratio"] for s in all_stats) / len(all_stats)
    logger.info(f"[SaLoRA Verification] Average safety norm ratio after projection: {avg_ratio:.6f} (Expected < 0.01)")
