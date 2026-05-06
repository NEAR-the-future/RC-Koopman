"""Base dataset interface for benchmark trajectory generation/loading."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict


class BaseDataset(ABC):
    """Abstract dataset class used by all benchmark systems."""

    def __init__(self, normalize: bool = True, random_seed: Optional[int] = None):
        """Initialize dataset state and optional reproducibility seed."""
        self.normalize_flag = normalize
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.t = None              # Time array, shape (T,)
        self.states = None         # State trajectory, shape (T, n_states)
        self.inputs = None         # Input sequence, shape (T, n_inputs) or None
        self.data_range = None     # Min/max stats for normalization

    @abstractmethod
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Generate or load data and return (t, states, inputs)."""
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return state dimension."""
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return input dimension (0 for autonomous systems)."""
        pass

    def normalize_data(
        self,
        data: np.ndarray,
        data_range: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        target_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> Tuple[np.ndarray, Tuple]:
        """Normalize data to target_range and return normalized data plus min/max stats."""
        target_min, target_max = target_range

        if data_range is None:
            data_min = np.min(data, axis=0, keepdims=True)
            data_max = np.max(data, axis=0, keepdims=True)
        else:
            data_min, data_max = data_range
            data_min = data_min.reshape(1, -1)
            data_max = data_max.reshape(1, -1)

        data_range_span = data_max - data_min
        data_range_span[data_range_span == 0] = 1.0  # Avoid divide-by-zero

        normalized_data = (data - data_min) / data_range_span
        normalized_data = normalized_data * (target_max - target_min) + target_min

        return normalized_data, (data_min.flatten(), data_max.flatten())

    def denormalize_data(
        self,
        normalized_data: np.ndarray,
        data_range: Tuple[np.ndarray, np.ndarray],
        target_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> np.ndarray:
        """Map normalized data back to the original value range."""
        target_min, target_max = target_range
        data_min, data_max = data_range

        data_min = data_min.reshape(1, -1)
        data_max = data_max.reshape(1, -1)

        data_range_span = data_max - data_min
        data_range_span[data_range_span == 0] = 1.0

        denormalized_data = (normalized_data - target_min) / (target_max - target_min)
        denormalized_data = denormalized_data * data_range_span + data_min

        return denormalized_data

    def get_training_data(self) -> Dict:
        """Return data in a unified dict format consumed by all methods."""
        if self.states is None:
            self.generate_data()

        return {
            't': self.t,
            'states': self.states,
            'inputs': self.inputs,
            'n_states': self.get_state_dim(),
            'n_inputs': self.get_input_dim(),
            'has_input': self.get_input_dim() > 0,
            'data_range': self.data_range,
        }

    def get_info(self) -> Dict:
        """Return dataset metadata and value ranges for inspection."""
        if self.states is None:
            self.generate_data()

        info = {
            'name': self.__class__.__name__,
            'n_samples': len(self.t),
            'n_states': self.get_state_dim(),
            'n_inputs': self.get_input_dim(),
            'has_input': self.get_input_dim() > 0,
            'normalized': self.normalize_flag,
            'time_range': (self.t[0], self.t[-1]) if self.t is not None else None,
        }

        if self.states is not None:
            info['state_range'] = (self.states.min(axis=0), self.states.max(axis=0))
        if self.inputs is not None:
            info['input_range'] = (self.inputs.min(axis=0), self.inputs.max(axis=0))

        return info
