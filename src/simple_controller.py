import torch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ControllerConfig:
    threshold_low: float = 0.85
    threshold_high: float = 0.95
    delta: float = 0.05

class FeedbackController:
    """
    Phase 5: A reactive, threshold-based incremental controller (B4).
    Dynamically scales the SaLoRA-style safety projection based on observed refusal rates.
    """
    def __init__(self, model, safety_directions, config=None):
        self.config = config or ControllerConfig()
        self.current_alpha = 0.0
        self.hooks = []
        self._register_hooks(model, safety_directions)
        
    def step(self, refusal_rate):
        """
        Updates the internal alpha based on the measured refusal rate.
        This is called once every evaluation cycle.
        """
        if refusal_rate < self.config.threshold_low:
            new_alpha = self.current_alpha + self.config.delta
        elif refusal_rate > self.config.threshold_high:
            new_alpha = self.current_alpha - self.config.delta
        else:
            new_alpha = self.current_alpha
            
        self.current_alpha = max(0.0, min(1.0, new_alpha))
        
        # Avoid floating point precision issues near 0 and 1
        if self.current_alpha < 1e-5:
            self.current_alpha = 0.0
        elif self.current_alpha > 0.99999:
            self.current_alpha = 1.0
            
        logger.info(f"[FeedbackController] Refusal Rate: {refusal_rate:.3f} | Updated Alpha: {self.current_alpha:.3f}")
        return self.current_alpha
        
    def _register_hooks(self, model, safety_directions):
        for name, module in model.named_modules():
            # In PEFT, lora_A and lora_B are ParameterDicts / ModuleDicts
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
                
                # The hook modifies the input `x` dynamically based on `self.current_alpha`
                def pre_hook(mod, args, V_base=V_base):
                    x = args[0]
                    # Fast path: if alpha is 0, skip the matrix multiplication
                    if self.current_alpha == 0.0:
                        return (x,)
                        
                    V = V_base.to(device=x.device, dtype=x.dtype)
                    
                    # x_proj = x - alpha * (x @ V) @ V^T
                    x_V = torch.matmul(x, V)
                    x_proj = x - self.current_alpha * torch.matmul(x_V, V.T)
                    
                    return (x_proj,)
                    
                hook_handle = lora_A.register_forward_pre_hook(pre_hook)
                self.hooks.append(hook_handle)
                
        logger.info(f"[FeedbackController] Registered {len(self.hooks)} dynamic hooks.")
