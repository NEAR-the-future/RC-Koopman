"""Traditional EDMD baseline with explicit lifting functions."""

import numpy as np
from typing import Optional
from methods.base_method import BaseMethod
from lifting_functions import LiftingFunctions
from config_unified import EDMDConfig, GlobalConfig


class EDMDTraditional(BaseMethod):
    """EDMD using handcrafted dictionary functions (poly/trig/cross/RBF)."""

    def __init__(
        self,
        n_states: int,
        n_inputs: int = 0,
        lifted_dim_D: int = None,
        poly_order: int = EDMDConfig.poly_order,
        use_trig: bool = EDMDConfig.use_trig,
        use_cross_terms: bool = EDMDConfig.use_cross_terms,
        use_rbf: bool = EDMDConfig.use_rbf,
        rbf_eta: float = EDMDConfig.rbf_eta,
        rbf_center_method: str = EDMDConfig.rbf_center_method,
        include_constant: bool = EDMDConfig.include_constant,
        use_regularization: bool = EDMDConfig.use_regularization,
        regularization_param: float = EDMDConfig.regularization_param,
        rcond: float = EDMDConfig.rcond,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_states=n_states, n_inputs=n_inputs, name="EDMD_Traditional")

        seed = random_seed if random_seed is not None else GlobalConfig.random_seed
        if seed is not None:
            np.random.seed(seed)

        if lifted_dim_D is None:
            lifted_dim_D = EDMDConfig.lifted_dim_D

        # Estimate non-RBF feature count; remaining budget is used for RBFs.
        n_other_features = n_states
        if include_constant:
            n_other_features += 1
        if poly_order > 0:
            n_other_features += sum((n_states + d - 1) for d in range(1, poly_order + 1))
        if use_trig and n_states >= 3:
            n_other_features += 2
        if use_cross_terms:
            n_other_features += n_states * (n_states - 1) // 2

        effective_lifted_dim = lifted_dim_D
        rbf_n_centers = max(0, effective_lifted_dim - n_other_features) if use_rbf else 0

        self.lifting = LiftingFunctions(
            n_states=n_states,
            n_inputs=n_inputs,
            poly_order=poly_order,
            use_trig=use_trig,
            use_cross_terms=use_cross_terms,
            use_rbf=use_rbf,
            rbf_eta=rbf_eta,
            rbf_n_centers=rbf_n_centers,
            rbf_center_method=rbf_center_method,
            include_constant=include_constant,
            random_seed=seed,
        )

        self.random_seed = seed
        self.use_regularization = use_regularization
        self.regularization_param = regularization_param
        self.rcond = rcond

        self.K_matrix = None
        self.G_matrix = None
        self.is_trained = False
        self.D = None

        self.config = {
            'poly_order': poly_order,
            'use_trig': use_trig,
            'use_cross_terms': use_cross_terms,
            'use_rbf': use_rbf,
            'rbf_eta': rbf_eta if use_rbf else None,
            'rbf_n_centers': rbf_n_centers if use_rbf else None,
            'rbf_center_method': rbf_center_method if use_rbf else None,
            'include_constant': include_constant,
        }

    def fit(self, data_dict: dict) -> None:
        """Fit K by least squares on lifted state transitions."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        states = data_dict['states']
        inputs = data_dict.get('inputs', None)
        T = states.shape[0]
        has_input = (inputs is not None) and (self.n_inputs > 0)

        if GlobalConfig.verbose:
            print(f"\n{'='*60}")
            print("EDMD training start")
            print(f"{'='*60}")
            print(f"Samples: {T}")
            print(f"State dimension: {self.n_states}")
            print(f"Input dimension: {self.n_inputs}")
            print(f"Has input: {has_input}")
            print("Lifting config:")
            for k, v in self.config.items():
                print(f"  {k}: {v}")

        if self.config['use_rbf']:
            self.lifting.initialize_rbf_centers(states)
            if GlobalConfig.verbose:
                print(f"Initialized RBF centers: {self.config['rbf_n_centers']}")

        Phi_X = self.lifting.compute_lifted_states(states[:-1], inputs[:-1] if has_input else None)
        Phi_Y = self.lifting.compute_lifted_states(states[1:], inputs[1:] if has_input else None)
        self.Phi_X_train = Phi_X
        self.D = Phi_X.shape[1]

        if GlobalConfig.verbose:
            print(f"Lifted feature dimension: {self.D}")

        if has_input:
            X_input = np.vstack([Phi_X.T, inputs[:-1].T])
            self.full_D = self.D + self.n_inputs
        else:
            X_input = Phi_X.T
            self.full_D = self.D

        # Predict lifted state only; control is treated as exogenous.
        Y_target = Phi_Y.T

        if self.use_regularization:
            I = np.eye(self.full_D)
            X_reg = X_input @ X_input.T + self.regularization_param * I
            self.K_matrix = Y_target @ X_input.T @ np.linalg.inv(X_reg)
        else:
            X_input_pinv = np.linalg.pinv(X_input, rcond=self.rcond)
            self.K_matrix = Y_target @ X_input_pinv

        self.G_matrix = Phi_X.T @ Phi_X
        self.is_trained = True

        if GlobalConfig.verbose:
            A_matrix = self.K_matrix[:, :self.D]
            print(f"Koopman matrix K=[A,B] shape: {self.K_matrix.shape}")
            if has_input:
                print(f"  A block shape (D x D): {A_matrix.shape}, cond(A): {np.linalg.cond(A_matrix):.2e}")
                print(f"  B block shape (D x M): {self.K_matrix[:, self.D:].shape}")
            print(f"cond(X_input): {np.linalg.cond(X_input):.2e}")
            print(f"cond(G): {np.linalg.cond(self.G_matrix):.2e}")
            print(f"{'='*60}")
            print("Training complete.")
            print(f"{'='*60}\n")

    def reconstruct(self, states: np.ndarray, inputs: Optional[np.ndarray] = None, method: str = 'one-step-ahead') -> np.ndarray:
        """Reconstruct trajectory using one-step or iterative multi-step rollout."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")

        T = states.shape[0]
        has_input = (inputs is not None) and (self.n_inputs > 0)

        Phi = self.lifting.compute_lifted_states(states, inputs)
        reconstructed_phi = np.zeros_like(Phi)
        reconstructed_phi[0] = Phi[0]

        if method == 'one-step-ahead':
            for t in range(T - 1):
                input_vec = np.concatenate([Phi[t], inputs[t]]) if has_input else Phi[t]
                reconstructed_phi[t + 1] = self.K_matrix @ input_vec

        elif method in ['multi-step', 'iterative']:
            current_phi = Phi[0].copy()
            for t in range(1, T):
                input_vec = np.concatenate([current_phi, inputs[t - 1]]) if has_input else current_phi
                current_phi = self.K_matrix @ input_vec

                if np.any(np.isnan(current_phi)) or np.any(np.isinf(current_phi)):
                    if GlobalConfig.verbose:
                        print(f"Warning: NaN/Inf detected during multi-step rollout at step {t}; stopping early.")
                    reconstructed_phi[t:] = reconstructed_phi[t - 1]
                    break

                reconstructed_phi[t] = current_phi
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")

        reconstructed_states = reconstructed_phi[:, :self.n_states]

        if np.any(np.isnan(reconstructed_states)):
            if GlobalConfig.verbose:
                print("Warning: NaN in reconstructed states; replacing NaNs with ground truth entries.")
            nan_mask = np.isnan(reconstructed_states)
            reconstructed_states[nan_mask] = states[nan_mask]

        return reconstructed_states

    def predict(self, initial_state: np.ndarray, inputs: Optional[np.ndarray], n_steps: int) -> np.ndarray:
        """Predict future states from an initial condition and optional control sequence."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")

        has_input = (inputs is not None) and (self.n_inputs > 0)
        initial_phi = self.lifting.phi(initial_state, inputs[0] if has_input else None)

        predicted_phi = np.zeros((n_steps + 1, self.D))
        predicted_phi[0] = initial_phi

        current_phi = initial_phi
        for t in range(1, n_steps + 1):
            phi_aug = np.concatenate([current_phi, inputs[t - 1]]) if has_input else current_phi
            current_phi = self.K_matrix @ phi_aug
            predicted_phi[t] = current_phi

        return predicted_phi[:, :self.n_states]

    def get_koopman_matrix(self) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.K_matrix

    def get_gramian_matrix(self) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        return self.G_matrix

    def get_info(self) -> dict:
        """Return method metadata and post-fit diagnostics."""
        info = {
            'method_name': 'EDMDTraditional',
            'n_states': self.n_states,
            'n_inputs': self.n_inputs,
            'feature_dim': self.D if self.is_trained else None,
            'lifting_config': self.config,
            'is_trained': self.is_trained,
        }

        if self.is_trained:
            info['koopman_analysis'] = self.analyze_koopman_matrix()
            info['gramian_analysis'] = self.analyze_gramian_matrix()
            info['lifting_info'] = self.lifting.get_info()

        return info
