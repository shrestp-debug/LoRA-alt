"""
metrics.py — Phase 1, Step 1.4
======================================
Core evaluation metrics suite for LoRA-SafeLoop.

Metrics implemented:
  1. Safety: Refusal Rate on AdvBench (520 harmful prompts, evaluated using refusal_judge.py)
  2. GSM8K: Exact-match accuracy on 200 math test examples
  3. Alpaca: Validation loss on 500 held-out instructions (with prompt masking)
  4. Subspace Alignment: Cosine alignment of LoRA weight update with safety directions
"""

import re
import logging
from typing import Optional

import torch
from tqdm import tqdm

from src.refusal_judge import evaluate_refusal_rate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Safety Evaluation (Refusal Rate)
# ===========================================================================

def evaluate_safety(
    model,
    tokenizer,
    advbench_prompts: list[str],
    batch_size: int = 4,
    device: str = "cpu",
    log_failures_to: Optional[str] = None,
) -> float:
    """
    Evaluates the model's safety by generating responses to AdvBench prompts
    and running them through the LLM-as-Judge refusal classifier.

    Args:
        model            : The HF model (optionally wrapped in PEFT).
        tokenizer        : The tokenizer matching the model.
        advbench_prompts : List of harmful prompts (AdvBench).
        batch_size       : Batch size for generating completions.
        device           : Device to run generation on.
        log_failures_to  : Optional CSV path to log judge failures.

    Returns:
        float : Refusal rate in [0, 1].
    """
    model.eval()
    completions = []
    
    # Save the original padding side and set to left for batch generation
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Generate responses from the model
    with torch.no_grad():
        for i in tqdm(range(0, len(advbench_prompts), batch_size), desc="Evaluating safety"):
            batch_prompts = advbench_prompts[i:i + batch_size]
            formatted_prompts = []
            
            for prompt in batch_prompts:
                # Apply Qwen chat template
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
                
            inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # greedy decoding for deterministic eval
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Slice out prompt tokens so we only decode the model's generated text
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            batch_completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            completions.extend(batch_completions)

    # Restore original padding side
    tokenizer.padding_side = prev_padding_side

    # 2. Evaluate refusal rate using the Gemini judge
    refusal_rate, _ = evaluate_refusal_rate(
        advbench_prompts, completions, log_failures_to=log_failures_to
    )
    return refusal_rate


# ===========================================================================
# 2. GSM8K Evaluation (Exact-Match Accuracy)
# ===========================================================================

def extract_gsm8k_gt_answer(answer_str: str) -> Optional[str]:
    """Extracts the final numeric ground truth answer from GSM8K's answer field."""
    # GSM8K ground truth ends with "#### <number>"
    match = re.search(r"####\s*(-?\d+[\.,]?\d*)", answer_str)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_gsm8k_pred_answer(model_completion: str) -> Optional[str]:
    """Parses model output to find the predicted mathematical answer."""
    # Case 1: Look for "#### <number>" if the model mimics training format
    match_hash = re.search(r"####\s*(-?\d+[\.,]?\d*)", model_completion)
    if match_hash:
        return match_hash.group(1).replace(",", "")
        
    # Case 2: Look for phrase triggers like "final answer is <number>"
    match_phrase = re.search(
        r"(?:final answer is|ans is|answer is)\s*:?\s*\\?\$?\s*(-?\d+[\.,]?\d*)",
        model_completion,
        re.IGNORECASE
    )
    if match_phrase:
        return match_phrase.group(1).replace(",", "")
        
    # Case 3: Fall back to the last numeric string in the model output
    numbers = re.findall(r"-?\d+[\.,]?\d*", model_completion)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def evaluate_task_gsm8k(
    model,
    tokenizer,
    test_examples: list[dict],
    batch_size: int = 4,
    device: str = "cpu"
) -> float:
    """
    Evaluates exact-match mathematical accuracy on the GSM8K test subset.

    Args:
        model         : The HF model.
        tokenizer     : The tokenizer matching the model.
        test_examples : List of math question/answer dicts.
        batch_size    : Batch size for generation.
        device        : Running device.

    Returns:
        float : Exact-match accuracy in [0, 1].
    """
    model.eval()
    correct = 0
    total = len(test_examples)
    
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="Evaluating GSM8K"):
            batch = test_examples[i:i + batch_size]
            formatted_prompts = []
            
            for item in batch:
                messages = [{"role": "user", "content": item["question"]}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
                
            inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            completions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for item, completion in zip(batch, completions):
                gt_num = extract_gsm8k_gt_answer(item["answer"])
                pred_num = extract_gsm8k_pred_answer(completion)
                
                if gt_num is not None and pred_num is not None:
                    if float(gt_num) == float(pred_num):
                        correct += 1
                elif gt_num is None:
                    logger.warning(f"Could not parse GSM8K GT answer: {item['answer']}")

    tokenizer.padding_side = prev_padding_side
    return correct / total if total > 0 else 0.0


# ===========================================================================
# 3. Alpaca Evaluation (Validation Loss with Prompt Masking)
# ===========================================================================

def tokenize_prompt_output(tokenizer, prompt: str, output: str, max_length: int = 512) -> dict:
    """Tokenizes and masks the prompt so loss is only calculated on the response."""
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Get length of formatted prompt tokens
    prompt_inputs = tokenizer(formatted_prompt, add_special_tokens=False)
    prompt_len = len(prompt_inputs.input_ids)
    
    # Combined tokenization
    full_text = formatted_prompt + output + tokenizer.eos_token
    full_inputs = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False
    )
    
    input_ids = full_inputs.input_ids
    attention_mask = full_inputs.attention_mask
    
    # Create labels, masking out the prompt tokens with -100
    labels = [-100] * len(input_ids)
    for j in range(prompt_len, len(input_ids)):
        labels[j] = input_ids[j]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def collate_fn(batch, tokenizer):
    """Pads a batch of masked sequence dicts to the maximum length in the batch."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids_batch.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask_batch.append(x["attention_mask"] + [0] * pad_len)
        labels_batch.append(x["labels"] + [-100] * pad_len)
        
    return {
        "input_ids": torch.tensor(input_ids_batch),
        "attention_mask": torch.tensor(attention_mask_batch),
        "labels": torch.tensor(labels_batch)
    }


def evaluate_task_alpaca(
    model,
    tokenizer,
    val_examples: list[dict],
    batch_size: int = 4,
    device: str = "cpu"
) -> float:
    """
    Computes cross-entropy validation loss on Alpaca instruction validation set.

    Args:
        model        : The HF model.
        tokenizer    : The tokenizer matching the model.
        val_examples : List of Alpaca validation dicts (prompt, output).
        batch_size   : Batch size for forward pass.
        device       : Running device.

    Returns:
        float : Average validation loss per label token.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 1. Tokenize all examples with masked prompts
    tokenized_examples = [
        tokenize_prompt_output(tokenizer, ex["prompt"], ex["output"])
        for ex in val_examples
    ]
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # 2. Compute loss across batches
    with torch.no_grad():
        for i in tqdm(range(0, len(tokenized_examples), batch_size), desc="Evaluating Alpaca Loss"):
            batch_data = tokenized_examples[i:i + batch_size]
            batch = collate_fn(batch_data, tokenizer)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Weighted average based on number of active label tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            
    return total_loss / total_tokens if total_tokens > 0 else 0.0


# ===========================================================================
# 4. Subspace Alignment Metric
# ===========================================================================

def compute_subspace_alignment(
    model,
    safety_directions: dict[int, torch.Tensor]
) -> dict[int, float]:
    """
    Computes the cosine alignment of the top right singular vector of the LoRA weight
    update matrix Delta W_l with the top-1 safety direction at each layer.

    Formula:
        Alignment = | \sigma_1(\Delta W_l) \cdot u_1^{safety} |

    Args:
        model             : The fine-tuned PEFT model.
        safety_directions : Dict mapping layer_idx -> safety directions tensor
                            of shape [d_model, k].

    Returns:
        dict[int, float] : Dict mapping layer_idx -> alignment score in [0, 1].
    """
    alignments = {}
    
    for layer_idx, directions in safety_directions.items():
        try:
            # Safely query the LoRA parameters from PEFT model structure
            q_proj = model.base_model.model.model.layers[layer_idx].self_attn.q_proj
            
            # Extract current weight components
            lora_A = q_proj.lora_A.default.weight.detach().to(torch.float32)  # Shape: [r, d_in]
            lora_B = q_proj.lora_B.default.weight.detach().to(torch.float32)  # Shape: [d_out, r]
            
            # delta_W = B @ A
            delta_W = lora_B @ lora_A  # Shape: [d_out, d_in] (d_in is d_model)
            
            # Compute SVD of Delta W
            # Vh returns right singular vectors (V^H). Vh[0] is the top right singular vector.
            _, _, Vh = torch.linalg.svd(delta_W, full_matrices=False)
            top_right_vector = Vh[0]  # Shape: [d_model]
            
            # The top-1 safety direction is the first column of the saved direction matrix
            top_safety_dir = directions[:, 0].to(torch.float32).to(top_right_vector.device)
            
            # Cosine alignment is the absolute value of the dot product (since both are unit vectors)
            alignment = torch.abs(torch.dot(top_right_vector, top_safety_dir)).item()
            alignments[layer_idx] = alignment
            
        except AttributeError:
            # If the layer does not have LoRA parameters (e.g. baseline or initial step)
            # return 0.0 as alignment is not defined
            alignments[layer_idx] = 0.0
        except Exception as e:
            logger.error(f"Error computing subspace alignment for layer {layer_idx}: {e}")
            alignments[layer_idx] = 0.0
            
    return alignments
