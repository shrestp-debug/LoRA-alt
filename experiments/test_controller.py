import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.simple_controller import FeedbackController, ControllerConfig

logging.basicConfig(level=logging.INFO)

def test_controller_logic():
    print("--- Running Controller Logic Test ---")
    
    # Mock model and safety directions (we don't need them to test the logic step)
    class MockModel:
        def named_modules(self):
            return []
            
    config = ControllerConfig(threshold_low=0.85, threshold_high=0.95, delta=0.05)
    controller = FeedbackController(MockModel(), {}, config=config)
    
    assert controller.current_alpha == 0.0, "Alpha should initialize to 0.0"
    
    test_sequence = [
        # Sequence of (refusal_rate, expected_alpha)
        (0.99, 0.00), # Very safe, alpha should stay 0.0 (clamped)
        (0.96, 0.00), # Still above high threshold, stay 0.0
        (0.90, 0.00), # In the safe zone (0.85-0.95), stay 0.0
        (0.80, 0.05), # Dropped below 0.85, alpha should increment
        (0.80, 0.10), # Still below, increment again
        (0.50, 0.15), # Still below, increment again
        (0.90, 0.15), # Back in safe zone, alpha holds steady
        (0.96, 0.10), # Too safe (above 0.95), alpha should decrement
        (0.99, 0.05), # Still too safe, decrement again
    ]
    
    for i, (refusal_rate, expected) in enumerate(test_sequence):
        alpha = controller.step(refusal_rate)
        print(f"Step {i+1}: Refusal Rate = {refusal_rate:.2f} -> Alpha = {alpha:.2f} (Expected: {expected:.2f})")
        assert abs(alpha - expected) < 1e-5, f"Mismatch at step {i+1}! Expected {expected}, got {alpha}"
        
    # Test upper clamping
    controller.current_alpha = 0.98
    alpha = controller.step(0.50) # Increment by 0.05
    print(f"Clamp Test High: Refusal Rate = 0.50 -> Alpha = {alpha:.2f} (Expected: 1.00)")
    assert alpha == 1.0, "Failed to clamp at 1.0"
    
    print("--- All tests passed! ---")

if __name__ == "__main__":
    test_controller_logic()
