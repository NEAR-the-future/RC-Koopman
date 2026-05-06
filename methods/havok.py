"""HAVOK baseline with delay embedding and linear lifted dynamics."""

import numpy as np
from typing import Optional, Tuple
from methods.base_method import BaseMethod
from config_unified import HAVOKConfig, GlobalConfig


class HAVOK(BaseMethod):
    """HAVOK using state-only delay embedding with optional exogenous controls."""

    def __init__(
        self,
        n_states: int,
        n_inputs: int = 0,
        embedding_dim: int = HAVOKConfig.embedding_dim,
        delay_steps: int = HAVOKConfig.delay_steps,
        include_input_in_hankel: bool = HAVOKConfig.include_input_in_hankel,
        use_regularization: bool = HAVOKConfig.use_regularization,
        regularization_param: float = HAVOKConfig.regularization_param,
        rcond: float = HAVOKConfig.rcond,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_states=n_states, n_inputs=n_inputs, name="HAVOK")

        self.r = embedding_dim
        self.tau = delay_steps
        self.include_input = include_input_in_hankel
        self.use_regularization = use_regularization
        self.regularization_param = regularization_param
        self.rcond = rcond

        seed = random_seed if random_seed is not None else GlobalConfig.random_seed
        if seed is not None:
            np.random.seed(seed)
        self.random_seed = seed

        self.D = self.n_states * self.r

        # K is A or [A,B], depending on input availability.
        self.K_matrix = None
        self.G_matrix = None
        self.is_trained = False
        self.delay_length = None

    def _construct_delay_embedding(self, states: np.ndarray, inputs: Optional[np.ndarray] = None, start_idx: int = 0) -> np.ndarray:
        """Build one delay-embedded vector psi(t) from states only."""
        psi_parts = []
        for k in range(self.n_states):
            delays = []
            for i in range(self.r):
                idx = start_idx + i * self.tau
                delays.append(states[idx, k])
            psi_parts.extend(delays)
        return np.array(psi_parts)

    def _construct_data_matrices(self, states: np.ndarray, inputs: Optional[np.ndarray] = None):
        """Build Psi_current, Psi_next and aligned control matrix U_current."""
        T = states.shape[0]
        self.delay_length = (self.r - 1) * self.tau
        T_eff = T - self.delay_length - 1

        if T_eff <= 0:
            raise ValueError(
                f"Insufficient data length. Need at least {self.delay_length + 2} points, got {T}."
            )

        Psi_current = np.zeros((self.D, T_eff))
        Psi_next = np.zeros((self.D, T_eff))

        for t in range(T_eff):
            Psi_current[:, t] = self._construct_delay_embedding(states, start_idx=t)
            Psi_next[:, t] = self._construct_delay_embedding(states, start_idx=t + 1)

        U_current = None
        if inputs is not None and self.n_inputs > 0 and self.include_input:
            U_current = inputs[:T_eff].T

        return Psi_current, Psi_next, U_current

    def fit(self, data_dict: dict) -> None:
        """Fit HAVOK matrix by least squares in delay-embedded coordinates."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        states = data_dict['states']
        inputs = data_dict.get('inputs', None)
        T = states.shape[0]

        if GlobalConfig.verbose:
            print(f"\n{'='*60}")
            print("HAVOK training start")
            print(f"{'='*60}")
            print(f"Samples: {T}")
            print(f"State dimension: {self.n_states}")
            print(f"Input dimension: {self.n_inputs}")
            print(f"Embedding dimension r: {self.r}")
            print(f"Delay step tau: {self.tau}")
            print(f"Model: psi(t+1) = K @ [psi(t); u(t)] = A psi(t) + B u(t)")
            print(f"Lifted dimension D = n_states * r: {self.D}")

        Psi_current, Psi_next, U_current = self._construct_data_matrices(states, inputs)
        self.Psi_current_train = Psi_current.T
        has_input = U_current is not None

        if GlobalConfig.verbose:
            print(f"Psi_current shape: {Psi_current.shape}")
            print(f"Psi_next shape: {Psi_next.shape}")
            if has_input:
                print(f"U_current shape: {U_current.shape}")
            print(f"Effective samples: {Psi_current.shape[1]}")

        X_input = np.vstack([Psi_current, U_current]) if has_input else Psi_current

        if self.use_regularization:
            lam = self.regularization_param
            X_XT = X_input @ X_input.T
            self.K_matrix = Psi_next @ X_input.T @ np.linalg.inv(X_XT + lam * np.eye(X_input.shape[0]))
        else:
            self.K_matrix = Psi_next @ np.linalg.pinv(X_input, rcond=self.rcond)

        self.G_matrix = Psi_current @ Psi_current.T / Psi_current.shape[1]
        self.is_trained = True

        if GlobalConfig.verbose:
            print(f"Koopman matrix K shape: {self.K_matrix.shape}")
            if has_input:
                A = self.K_matrix[:, :self.D]
                B = self.K_matrix[:, self.D:]
                rho_A = np.max(np.abs(np.linalg.eigvals(A)))
                print(f"A block shape: {A.shape}, spectral radius: {rho_A:.4f}")
                print(f"B block shape: {B.shape}")
            else:
                rho_K = np.max(np.abs(np.linalg.eigvals(self.K_matrix)))
                print(f"K=A spectral radius: {rho_K:.4f}")
            print(f"G shape: {self.G_matrix.shape}, cond(G): {np.linalg.cond(self.G_matrix):.2e}")
            print(f"{'='*60}")
            print("Training complete.")
            print(f"{'='*60}\n")

    def reconstruct(self, states: np.ndarray, inputs: Optional[np.ndarray] = None, method: str = 'one-step-ahead') -> Tuple[np.ndarray, Tuple[int, int]]:
        """Reconstruct trajectories and return valid predicted index range."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")

        T = states.shape[0]
        reconstructed = np.zeros((T, self.n_states))
        has_input = (inputs is not None) and (self.n_inputs > 0) and self.include_input

        if method == 'one-step-ahead':
            for t in range(T - self.r):
                psi_t = self._construct_delay_embedding(states, start_idx=t)
                input_vec = np.concatenate([psi_t, inputs[t]]) if has_input else psi_t
                psi_next = self.K_matrix @ input_vec
                for k in range(self.n_states):
                    reconstructed[t + self.r, k] = psi_next[k * self.r + (self.r - 1)]

            reconstructed[:self.r] = states[:self.r]
            valid_range = (self.r, T)

        elif method in ['multi-step', 'iterative']:
            psi_current = self._construct_delay_embedding(states, start_idx=0)
            for t in range(T - self.r):
                input_vec = np.concatenate([psi_current, inputs[t]]) if has_input else psi_current
                psi_current = self.K_matrix @ input_vec
                for k in range(self.n_states):
                    reconstructed[t + 1, k] = psi_current[k * self.r]

            reconstructed[:self.r] = states[:self.r]
            valid_range = (1, T - self.r + 1)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")

        return reconstructed, valid_range

    def predict(self, initial_state: np.ndarray, inputs: Optional[np.ndarray], n_steps: int) -> np.ndarray:
        """Predict future states from a history window and optional input sequence."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")

        if initial_state.ndim == 1:
            raise ValueError(f"HAVOK requires at least {self.delay_length + 1} history states for delay embedding.")
        if initial_state.shape[0] < self.delay_length + 1:
            raise ValueError(f"HAVOK requires at least {self.delay_length + 1} history states.")

        has_input = (inputs is not None) and (self.n_inputs > 0) and self.include_input
        history_states = initial_state[: self.delay_length + 1, :]
        psi_current = self._construct_delay_embedding(history_states, start_idx=0)

        predicted_states = np.zeros((n_steps + 1, self.n_states))
        predicted_states[0] = history_states[-1]

        for t in range(n_steps):
            if has_input and inputs is not None and t < len(inputs):
                input_vec = np.concatenate([psi_current, inputs[t]])
            else:
                input_vec = psi_current
            psi_current = self.K_matrix @ input_vec
            for k in range(self.n_states):
                predicted_states[t + 1, k] = psi_current[k * self.r]

        return predicted_states

    def get_koopman_matrix(self) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.K_matrix

    def get_gramian_matrix(self) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.G_matrix

    def analyze_koopman_matrix(self) -> dict:
        """Return spectral and conditioning diagnostics for K (and A block)."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")

        K = self.K_matrix
        A = K[:, :self.D]

        eigenvalues = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigenvalues))
        k_condition_number = np.linalg.cond(K)
        a_condition_number = np.linalg.cond(A)
        matrix_rank = np.linalg.matrix_rank(K)

        return {
            'spectral_radius': spectral_radius,
            'eigenvalues_real': eigenvalues.real.tolist(),
            'eigenvalues_imag': eigenvalues.imag.tolist(),
            'condition_number': k_condition_number,
            'a_condition_number': a_condition_number,
            'matrix_rank': matrix_rank,
            'matrix_shape': K.shape,
            'frobenius_norm': np.linalg.norm(K, 'fro'),
        }

    def get_info(self) -> dict:
        """Return method metadata and post-fit diagnostics."""
        info = {
            'method_name': 'HAVOK',
            'n_states': self.n_states,
            'n_inputs': self.n_inputs,
            'embedding_dim': self.r,
            'delay_steps': self.tau,
            'include_input_in_embedding': self.include_input,
            'lifted_dim': self.D,
            'delay_length': self.delay_length,
            'is_trained': self.is_trained,
        }

        if self.is_trained:
            info['koopman_analysis'] = self.analyze_koopman_matrix()
            info['gramian_condition_number'] = np.linalg.cond(self.G_matrix)

        return info
