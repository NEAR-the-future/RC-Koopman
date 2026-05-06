

"""RC-Koopman method using reservoir lifting plus EDMD-style regression."""

import numpy as np
from typing import Optional, Callable
from scipy.sparse import random as sparse_random
from methods.base_method import BaseMethod
from config_unified import RCKoopmanConfig, GlobalConfig


class RCKoopman(BaseMethod):
    """Improved RC+Koopman method with ESP condition rho(Wres) < 1."""
    
    def __init__(self,
                 n_states: int,
                 n_inputs: int = 0,
                 lifted_dim_D: int = None,
                 rho_wres: float = RCKoopmanConfig.rho_wres,
                 noise_level: float = RCKoopmanConfig.noise_level,
                 w_range: float = RCKoopmanConfig.w_range,
                 win_range: float = RCKoopmanConfig.win_range,
                 density: float = RCKoopmanConfig.density,
                 activation: str = RCKoopmanConfig.activation,
                 washout: int = RCKoopmanConfig.washout,
                 use_regularization: bool = RCKoopmanConfig.use_regularization,
                 regularization_param: float = RCKoopmanConfig.regularization_param,
                 rcond: float = RCKoopmanConfig.rcond,
                 random_seed: Optional[int] = None):
        
        super().__init__(n_states=n_states, n_inputs=n_inputs, name="RC_Koopman")
        
        self.K = n_states  # Original state dimension.
        if lifted_dim_D is None:
            lifted_dim_D = RCKoopmanConfig.lifted_dim_D
        self.N = lifted_dim_D - self.K  # Reservoir size N = D - K.
        self.rho_wres = rho_wres
        self.noise_level = noise_level
        self.washout = washout
        self.activation_name = activation
        
        self.use_regularization = use_regularization
        self.regularization_param = regularization_param
        self.rcond = rcond
        
        self.D = self.K + self.N  # lifted_dim_D
        
        seed = random_seed if random_seed is not None else GlobalConfig.random_seed
        if seed is not None:
            np.random.seed(seed)
        
        self.random_seed = seed
        
        self.Win = self._initialize_input_weights(win_range)
        self.Wres = self._initialize_reservoir_weights(w_range, density, rho_wres)
        
        self.activation = self._get_activation_function(activation)
        
        self._check_esp()
        
        self.r = None  # Current reservoir state.
        self.K_matrix = None  # Learned Koopman matrix.
        self.is_trained = False
    
    def _initialize_input_weights(self, win_range: float) -> np.ndarray:
        
        Win = np.random.uniform(-win_range, win_range, size=(self.N, self.K))
        return Win
    
    def _initialize_reservoir_weights(self, w_range: float, density: float, rho: float) -> np.ndarray:
        
        W_sparse = sparse_random(
            self.N, self.N,
            density=density,
            data_rvs=lambda s: np.random.uniform(-w_range, w_range, size=s)
        )
        Wres = W_sparse.toarray()
        
        eigenvalues = np.linalg.eigvals(Wres)
        current_rho = np.max(np.abs(eigenvalues))
        
        if current_rho > 0:
            Wres = Wres * (rho / current_rho)
        
        return Wres
    
    def _get_activation_function(self, activation_type: str) -> Callable:
        
        activations = {
            'tanh': np.tanh,
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'relu': lambda x: np.maximum(0, x),
            'identity': lambda x: x
        }
        
        if activation_type not in activations:
            raise ValueError(f"Unsupported activation function: {activation_type}")
        
        return activations[activation_type]
    
    def _check_esp(self):
        
        actual_rho = np.max(np.abs(np.linalg.eigvals(self.Wres)))
        
        if actual_rho >= 1.0:
            print(f"Warning: ESP condition violated. rho(Wres) = {actual_rho:.4f} >= 1.0")
        elif GlobalConfig.verbose:
            print(f"ESP condition satisfied: rho(Wres) = {actual_rho:.4f} < 1.0")
    
    def _reset_reservoir_state(self):
        
        self.r = np.random.uniform(0.0, 1.0, size=self.N)
    
    def _collect_reservoir_states(self, states: np.ndarray) -> np.ndarray:
        
        T = states.shape[0]
        self._reset_reservoir_state()
        
        all_reservoir_states = np.zeros((T, self.N))
        
        for t in range(T):
            v = states[t]
            
            noise = 0
            if self.noise_level > 0:
                noise = np.random.uniform(-self.noise_level, self.noise_level, size=self.N)
            
            pre_activation = self.Wres @ self.r + self.Win @ v + noise
            self.r = self.activation(pre_activation)
            
            all_reservoir_states[t] = self.r
        
        reservoir_states = all_reservoir_states[self.washout:]
        
        return reservoir_states
    
    def fit(self, data_dict: dict) -> None:
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        states = data_dict['states']  # (T, K)
        T = states.shape[0]
        
        if GlobalConfig.verbose:
            print(f"\n{'='*60}")
            print("RC_Koopman training start")
            print(f"{'='*60}")
            print(f"Samples: {T}")
            print(f"State dimension: {self.K}")
            print(f"Reservoir dimension: {self.N}")
            print(f"Washout: {self.washout}")
        
        reservoir_states = self._collect_reservoir_states(states)  # (T-washout, N)
        T_effective = T - self.washout
        
        states_effective = states[self.washout:]  # (T-washout, K)
        
        augmented_states = np.hstack([states_effective, reservoir_states])  # (T-washout, K+N)
        
        if GlobalConfig.verbose:
            print(f"Effective samples (after washout): {T_effective}")
            print(f"Lifted dimension: {self.D}")
        
        raw_inputs = data_dict.get('inputs', None)
        has_control_input = (raw_inputs is not None) and (self.n_inputs > 0)
        inputs_effective = raw_inputs[self.washout:] if has_control_input else None  # (T_effective, M)
        
        self._train_koopman(augmented_states, inputs_effective=inputs_effective)
        
        self.augmented_states_train = augmented_states
        self.states_train_full = states
        
        self.is_trained = True
        
        if GlobalConfig.verbose:
            A_part = self.K_matrix[:, :self.D]
            rho_K = np.max(np.abs(np.linalg.eigvals(A_part)))
            print(f"\nKoopman matrix summary:")
            print(f"  Shape: {self.K_matrix.shape}")
            print(f"  Spectral radius of A block: {rho_K:.6f}")
            print(f"  Condition number of K: {np.linalg.cond(self.K_matrix):.2e}")
            print(f"  Condition number of regression matrix S: {self.S_condition_number:.2e}")
            print(f"{'='*60}")
            print("Training complete.")
            print(f"{'='*60}\n")
    
    def _train_koopman(self, augmented_states: np.ndarray, inputs_effective: Optional[np.ndarray] = None):
        
        T = augmented_states.shape[0]
        has_input = (inputs_effective is not None and self.n_inputs > 0
                     and inputs_effective.shape[0] >= T)
        
        Psi = augmented_states[:-1].T      # (D, T-1)  Z_t
        Psi_next = augmented_states[1:].T  # (D, T-1), target lifted states.
        
        if has_input:
            U = inputs_effective[:-1].T  # (M, T-1)
            X_input = np.vstack([Psi, U])  # (D+M, T-1)
        else:
            X_input = Psi  # (D, T-1)
        
        if self.use_regularization:
            n_in = X_input.shape[0]
            A_mat = Psi_next @ X_input.T
            G = X_input @ X_input.T + self.regularization_param * np.eye(n_in)
            self.K_matrix = np.linalg.solve(G, A_mat.T).T
        else:
            X_input_pinv = np.linalg.pinv(X_input, rcond=self.rcond)
            self.K_matrix = Psi_next @ X_input_pinv  # (D, D+M) or (D, D)
        
        self.S_condition_number = np.linalg.cond(X_input)
        
        if GlobalConfig.verbose:
            A_part = self.K_matrix[:, :self.D]
            print(f"Koopman(K=[A,B]): {self.K_matrix.shape}")
            if has_input:
                print(f"  A block shape (D x D): {A_part.shape}, cond(A): {np.linalg.cond(A_part):.2e}")
            print(f"S: {self.S_condition_number:.2e}")
            print(f"K(A): {np.max(np.abs(np.linalg.eigvals(A_part))):.4f}")
    
    def reconstruct(self, states: np.ndarray, inputs: Optional[np.ndarray] = None,
                    method: str = 'one-step-ahead') -> np.ndarray:
        
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")
        
        if not hasattr(self, 'augmented_states_train') or self.augmented_states_train is None:
            raise RuntimeError("Training lifted states were not saved. Please check training pipeline.")
        
        augmented_states = self.augmented_states_train  # Lifted state psi = [x; r], excluding control input u.
        T_eff = augmented_states.shape[0]
        
        has_input = (inputs is not None) and (self.n_inputs > 0)
        if has_input:
            inputs_eff = inputs[self.washout:]  # (T_eff, M)
        
        if method == 'one-step-ahead':
            reconstructed_aug = np.zeros_like(augmented_states)
            reconstructed_aug[0] = augmented_states[0]
            
            for t in range(T_eff - 1):
                if has_input:
                    input_vec = np.concatenate([augmented_states[t], inputs_eff[t]])
                else:
                    input_vec = augmented_states[t]
                reconstructed_aug[t + 1] = self.K_matrix @ input_vec
        
        elif method in ['multi-step', 'iterative']:
            reconstructed_aug = np.zeros_like(augmented_states)
            reconstructed_aug[0] = augmented_states[0]
            
            current_psi = augmented_states[0].copy()
            for t in range(1, T_eff):
                if has_input:
                    input_vec = np.concatenate([current_psi, inputs_eff[t - 1]])
                else:
                    input_vec = current_psi
                current_psi = self.K_matrix @ input_vec
                reconstructed_aug[t] = current_psi
        
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
        
        reconstructed_states = reconstructed_aug[:, :self.K]
        
        if hasattr(self, 'states_train_full') and self.states_train_full is not None:
            T_original = self.states_train_full.shape[0]
            full_reconstructed = np.zeros((T_original, self.K))
            full_reconstructed[:self.washout] = self.states_train_full[:self.washout]
            full_reconstructed[self.washout:] = reconstructed_states
        else:
            full_reconstructed = reconstructed_states
        
        return full_reconstructed
    
    def predict(self, initial_state: np.ndarray, inputs: Optional[np.ndarray],
                n_steps: int) -> np.ndarray:
        
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")
        
        has_input = (inputs is not None) and (self.n_inputs > 0)
        
        self._reset_reservoir_state()
        
        v = initial_state[:self.K]
        pre_activation = self.Wres @ self.r + self.Win @ v
        self.r = self.activation(pre_activation)
        
        initial_aug = np.concatenate([v, self.r])
        
        predicted_aug = np.zeros((n_steps + 1, self.D))
        predicted_aug[0] = initial_aug
        
        current_state = initial_aug
        for t in range(1, n_steps + 1):
            if has_input and inputs is not None and t - 1 < len(inputs):
                input_vec = np.concatenate([current_state, inputs[t - 1]])
            else:
                input_vec = current_state
            current_state = self.K_matrix @ input_vec
            predicted_aug[t] = current_state
        
        predicted_states = predicted_aug[:, :self.K]
        
        return predicted_states
    
    def get_koopman_matrix(self) -> np.ndarray:
        
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.K_matrix
    
    def get_gramian_matrix(self) -> np.ndarray:
        
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        if not hasattr(self, 'augmented_states_train'):
            return None
        Psi = self.augmented_states_train.T  # (D, T)
        G = Psi @ Psi.T / Psi.shape[1]
        return G
    
    def get_info(self) -> dict:
        
        info = {
            'method_name': 'RC_Koopman',
            'n_states': self.n_states,
            'n_inputs': self.n_inputs,
            'reservoir_size': self.N,
            'dictionary_dim': self.D,
            'esp_condition': f"rho(Wres) = {self.rho_wres:.4f} < 1",
            'noise_level': self.noise_level,
            'washout': self.washout,
            'activation': self.activation_name,
            'is_trained': self.is_trained
        }
        
        if self.is_trained:
            info['koopman_analysis'] = self.analyze_koopman_matrix()
        
        return info
