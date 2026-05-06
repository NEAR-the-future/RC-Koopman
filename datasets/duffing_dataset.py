"""Duffing oscillator benchmark dataset (autonomous, no control input)."""

import numpy as np
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class DuffingDataset(BaseDataset):
    """Generate trajectories for the 2D Duffing system."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = -1.0,
        gamma: float = 0.5,
        t_start: float = 0.0,
        t_end: float = 20.0,
        time_step: float = 0.04,
        initial_condition: Optional[np.ndarray] = None,
        normalize: bool = True,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        random_seed: Optional[int] = None,
    ):
        """Configure Duffing dynamics and auto-generate data."""
        super().__init__(normalize=normalize, random_seed=random_seed)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.t_start = t_start
        self.t_end = t_end
        self.time_step = time_step
        self.normalize_range = normalize_range

        self.initial_condition = np.array([-1.21, 0.81]) if initial_condition is None else np.array(initial_condition)

        self.generate_data()

    def duffing_dynamics(self, state: np.ndarray) -> np.ndarray:
        """Continuous-time Duffing dynamics dx/dt = f(x)."""
        x1, x2 = state
        x1_dot = x2
        x2_dot = -self.gamma * x2 - (self.alpha * x1**2 + self.beta) * x1
        return np.array([x1_dot, x2_dot])

    def generate_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate dynamics with RK4 and return (t, states)."""
        t = np.arange(self.t_start, self.t_end + self.time_step / 2, self.time_step)
        num_points = len(t)

        states = np.zeros((num_points, 2))
        states[0] = self.initial_condition

        h = self.time_step
        for i in range(num_points - 1):
            state = states[i]
            k1 = self.duffing_dynamics(state)
            k2 = self.duffing_dynamics(state + h / 2 * k1)
            k3 = self.duffing_dynamics(state + h / 2 * k2)
            k4 = self.duffing_dynamics(state + h * k3)
            states[i + 1] = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return t, states

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, None]:
        """Generate and optionally normalize Duffing trajectories."""
        t, states_raw = self.generate_trajectory()

        if self.normalize_flag:
            states, self.data_range = self.normalize_data(states_raw, target_range=self.normalize_range)
        else:
            states = states_raw
            self.data_range = None

        self.t = t
        self.states = states
        self.inputs = None

        return t, states, None

    def get_state_dim(self) -> int:
        """Duffing has 2 state variables."""
        return 2

    def get_input_dim(self) -> int:
        """Duffing is autonomous (no input)."""
        return 0

    def get_info(self) -> dict:
        """Return dataset metadata for logging and reporting."""
        info = super().get_info()
        info.update(
            {
                'system': 'Duffing Oscillator',
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'time_step': self.time_step,
                'initial_condition': self.initial_condition.tolist(),
            }
        )
        return info
