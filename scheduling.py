"""
Learning rate and resolution scheduling utilities.
"""

import numpy as np


def get_resolution_blend_weights(
    iteration: int, 
    low_reso_stage: int, 
    blend_window: int = 500
) -> tuple:
    """
    Compute smooth transition weights from low-res to high-res training.
    
    Uses cosine annealing for smooth blending, avoiding sudden jumps
    that can cause artifacts.
    
    Args:
        iteration: Current training iteration
        low_reso_stage: Iteration at which to fully switch to high-res
        blend_window: Number of iterations for blending transition
        
    Returns:
        Tuple of (low_weight, high_weight) that sum to 1.0
    """
    blend_start = low_reso_stage - blend_window
    
    if iteration < blend_start:
        # Pure low-res phase
        return 1.0, 0.0
    
    if iteration < low_reso_stage:
        # Blending phase: cosine annealing
        progress = (iteration - blend_start) / float(blend_window)
        high_weight = 0.5 * (1 - np.cos(np.pi * progress))
        return 1.0 - high_weight, high_weight
    
    # Pure high-res phase
    return 0.0, 1.0
