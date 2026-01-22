"""
Optimization parameters for Gaussian model training.
"""


class OptimizationParams:
    """
    Default optimization hyperparameters for 3D Gaussian Splatting.
    
    Attributes:
        position_lr_init: Initial learning rate for positions
        position_lr_final: Final learning rate for positions
        position_lr_delay_mult: Delay multiplier for position LR scheduling
        position_lr_max_steps: Max steps for position LR scheduling
        intensity_lr: Learning rate for intensity
        scaling_lr: Learning rate for scaling
        rotation_lr: Learning rate for rotation
        percent_dense: Percentage for densification threshold
    """
    
    def __init__(
        self,
        position_lr_init: float = 0.002,
        position_lr_final: float = 0.000002,
        position_lr_delay_mult: float = 0.01,
        position_lr_max_steps: int = 30_000,
        intensity_lr: float = 0.05,
        scaling_lr: float = 0.005,
        rotation_lr: float = 0.001,
        percent_dense: float = 0.01,
    ):
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.position_lr_max_steps = position_lr_max_steps
        self.intensity_lr = intensity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr
        self.percent_dense = percent_dense
    
    @classmethod
    def from_config(cls, config: dict) -> 'OptimizationParams':
        """Create OptimizationParams from a config dictionary."""
        return cls(
            position_lr_init=config.get('position_lr_init', 0.002),
            position_lr_final=config.get('position_lr_final', 0.000002),
            position_lr_delay_mult=config.get('position_lr_delay_mult', 0.01),
            position_lr_max_steps=config.get('position_lr_max_steps', 30_000),
            intensity_lr=config.get('intensity_lr', 0.05),
            scaling_lr=config.get('scaling_lr', 0.005),
            rotation_lr=config.get('rotation_lr', 0.001),
            percent_dense=config.get('percent_dense', 0.01),
        )
