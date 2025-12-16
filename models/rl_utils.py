"""
Legacy utility module.
This module re-exports functions from the new modular structure to maintain backward compatibility.
Please prefer importing from the new locations in 'models/utils/' and 'models/training/'.
"""

from models.utils.features import (
    get_representation_state,
    state_to_tensor,
    get_grid_state,
    flip_action
)
from models.utils.rewards import get_shaped_reward
from models.utils.persistence import (
    generate_save_path,
    save_model,
    load_model
)
from models.training.engine import (
    run_episode,
    train_loop
)
