"""Differential-drive robot benchmark dataset (controlled system)."""

import numpy as np
from typing import Tuple, Optional
from .base_dataset import BaseDataset


class DiffDriveDataset(BaseDataset):
    """Generate multi-trajectory data for a differential-drive robot."""

    def __init__(
        self,
        dt: float = 0.05,
        n_trajectories: int = 200,
        steps_per_traj: int = 50,
        x_range: Tuple[float, float] = (-2.0, 2.0),
        y_range: Tuple[float, float] = (-2.0, 2.0),
        theta_range: Tuple[float, float] = (-np.pi, np.pi),
        v_range: Tuple[float, float] = (-0.6, 0.6),
        omega_range: Tuple[float, float] = (-2.0, 2.0),
        normalize: bool = True,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        random_seed: Optional[int] = None,
    ):
        """Configure trajectory generation and auto-generate data."""
        super().__init__(normalize=normalize, random_seed=random_seed)

        self.dt = dt
        self.n_trajectories = n_trajectories
        self.steps_per_traj = steps_per_traj
        self.x_range = x_range
        self.y_range = y_range
        self.theta_range = theta_range
        self.v_range = v_range
        self.omega_range = omega_range
        self.normalize_range = normalize_range

        self.generate_data()

    def dynamics_continuous(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time kinematics: [x_dot, y_dot, theta_dot]."""
        x, y, theta = state
        v, omega = u
        return np.array([v * np.cos(theta), v * np.sin(theta), omega])

    def dynamics_discrete(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Forward Euler discretization."""
        return state + self.dynamics_continuous(state, u) * self.dt

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random excited trajectories and optionally normalize them."""
        X_list = []  # x_t
        U_list = []  # u_t

        for _ in range(self.n_trajectories):
            x0 = np.array(
                [
                    np.random.uniform(*self.x_range),
                    np.random.uniform(*self.y_range),
                    np.random.uniform(*self.theta_range),
                ]
            )
            state = x0.copy()

            for _ in range(self.steps_per_traj):
                # Persistent excitation: occasional impulse plus baseline random actuation.
                if np.random.rand() < 0.05:
                    v = np.random.uniform(-1.2, 1.2)
                    omega = np.random.uniform(-4.0, 4.0)
                else:
                    v = np.random.uniform(*self.v_range) + 0.05 * np.random.randn()
                    omega = np.random.uniform(*self.omega_range) + 0.2 * np.random.randn()

                u = np.array([v, omega])
                X_list.append(state.copy())
                U_list.append(u.copy())
                state = self.dynamics_discrete(state, u)

        states_raw = np.array(X_list)
        inputs_raw = np.array(U_list)

        T = len(X_list)
        t = np.arange(T) * self.dt

        if self.normalize_flag:
            states, state_range = self.normalize_data(states_raw, target_range=self.normalize_range)
            inputs, input_range = self.normalize_data(inputs_raw, target_range=self.normalize_range)
            self.data_range = {'states': state_range, 'inputs': input_range}
        else:
            states = states_raw
            inputs = inputs_raw
            self.data_range = None

        self.t = t
        self.states = states
        self.inputs = inputs

        return t, states, inputs

    def get_state_dim(self) -> int:
        """Differential-drive state dimension."""
        return 3

    def get_input_dim(self) -> int:
        """Differential-drive input dimension."""
        return 2

    def get_info(self) -> dict:
        """Return dataset metadata for logging and reporting."""
        info = super().get_info()
        info.update(
            {
                'system': 'Differential-Drive Robot',
                'dt': self.dt,
                'n_trajectories': self.n_trajectories,
                'steps_per_traj': self.steps_per_traj,
                'total_samples': self.n_trajectories * self.steps_per_traj,
            }
        )
        return info
