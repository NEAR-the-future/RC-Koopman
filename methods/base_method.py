"""Abstract base interface for Koopman-learning methods."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseMethod(ABC):
    """Common API for all method implementations in this repository."""

    def __init__(self, n_states: int = None, n_inputs: int = None, name: str = "BaseMethod"):
        self.name = name
        self.K = None
        self.G = None
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_lifted = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, data_dict: Dict) -> None:
        """Fit model parameters from a unified dataset dict."""
        pass

    @abstractmethod
    def reconstruct(self, states: np.ndarray, inputs: Optional[np.ndarray] = None, method: str = 'one-step-ahead') -> np.ndarray:
        """Reconstruct a state trajectory using the learned model."""
        pass

    @abstractmethod
    def predict(self, initial_state: np.ndarray, inputs: Optional[np.ndarray] = None, n_steps: int = 100) -> np.ndarray:
        """Predict future states from an initial condition."""
        pass

    @abstractmethod
    def get_koopman_matrix(self) -> np.ndarray:
        """Return Koopman matrix K (or K=[A,B] for controlled systems)."""
        pass

    @abstractmethod
    def get_gramian_matrix(self) -> np.ndarray:
        """Return Gramian matrix G if available."""
        pass

    def compute_reconstruction_error(self, states_true: np.ndarray, states_recon: np.ndarray) -> Dict:
        """Compute RMSE/NRMSE diagnostics for reconstructed trajectories."""
        rmse = np.sqrt(np.mean((states_true - states_recon) ** 2))

        state_range = states_true.max(axis=0) - states_true.min(axis=0)
        state_range[state_range == 0] = 1.0
        nrmse = np.sqrt(np.mean(((states_true - states_recon) / state_range) ** 2))

        nrmse_per_dim = []
        for i in range(self.n_states):
            dim_range = state_range[i]
            dim_nrmse = np.sqrt(np.mean(((states_true[:, i] - states_recon[:, i]) / dim_range) ** 2))
            nrmse_per_dim.append(dim_nrmse)

        return {
            'rmse': rmse,
            'nrmse': nrmse,
            'nrmse_per_dim': nrmse_per_dim,
            'max_error': np.max(np.abs(states_true - states_recon)),
            'mean_error': np.mean(np.abs(states_true - states_recon)),
        }

    def analyze_koopman_matrix(self) -> Dict:
        """Return spectral/conditioning diagnostics for Koopman matrix."""
        try:
            K_matrix = self.get_koopman_matrix()
        except Exception:
            return {}

        if K_matrix is None:
            return {}

        if K_matrix.shape[0] != K_matrix.shape[1]:
            K_matrix = K_matrix[:, :K_matrix.shape[0]]

        eigenvalues = np.linalg.eigvals(K_matrix)
        spectral_radius = np.max(np.abs(eigenvalues))

        try:
            cond_number = np.linalg.cond(K_matrix)
        except Exception:
            cond_number = np.inf

        n_unstable = np.sum(np.abs(eigenvalues) > 1.0)
        matrix_norm = np.linalg.norm(K_matrix, ord=2)

        return {
            'eigenvalues': eigenvalues,
            'spectral_radius': spectral_radius,
            'condition_number': cond_number,
            'n_unstable_modes': n_unstable,
            'matrix_norm': matrix_norm,
            'matrix_shape': K_matrix.shape,
        }

    def analyze_gramian_matrix(self) -> Dict:
        """Return conditioning/rank diagnostics for Gramian matrix."""
        if self.G is None:
            return {}

        try:
            cond_number = np.linalg.cond(self.G)
        except Exception:
            cond_number = np.inf

        try:
            rank = np.linalg.matrix_rank(self.G)
        except Exception:
            rank = None

        return {
            'condition_number': cond_number,
            'rank': rank,
            'matrix_shape': self.G.shape,
            'matrix_norm': np.linalg.norm(self.G, ord=2),
        }

    def get_info(self) -> Dict:
        """Return method metadata and optional fitted diagnostics."""
        info = {
            'name': self.name,
            'n_states': self.n_states,
            'n_inputs': self.n_inputs,
            'n_lifted': self.n_lifted,
            'is_fitted': self.is_fitted,
        }

        if self.is_fitted:
            info.update(
                {
                    'koopman_analysis': self.analyze_koopman_matrix(),
                    'gramian_analysis': self.analyze_gramian_matrix(),
                }
            )

        return info
