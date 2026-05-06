"""Feature lifting utilities for EDMD.

This module builds explicit dictionary features (polynomial, trigonometric,
cross terms, and optional RBFs) used by the EDMD baseline.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans


class LiftingFunctions:
    """Construct and evaluate lifting maps phi(x, u)."""

    def __init__(
        self,
        n_states: int,
        n_inputs: int = 0,
        poly_order: int = 2,
        use_trig: bool = True,
        use_cross_terms: bool = True,
        use_rbf: bool = False,
        rbf_centers: Optional[np.ndarray] = None,
        rbf_eta: float = 1.0,
        rbf_n_centers: int = 50,
        rbf_center_method: str = 'random',
        include_constant: bool = False,
        include_u_in_phi: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.n = n_states
        self.m = n_inputs
        self.poly_order = poly_order
        self.use_trig = use_trig
        self.use_cross_terms = use_cross_terms
        self.use_rbf = use_rbf
        self.rbf_eta = rbf_eta
        self.rbf_n_centers = rbf_n_centers
        self.rbf_center_method = rbf_center_method
        self.include_constant = include_constant
        self.include_u_in_phi = include_u_in_phi
        self.random_seed = random_seed  # Stored for reproducible center selection.

        self.rbf_centers = rbf_centers
        self.rbf_centers_initialized = (rbf_centers is not None)
        self.poly_terms = self._poly_term_indices(self.n, self.poly_order)

    def _poly_term_indices(self, n: int, order: int) -> List[Tuple]:
        """Enumerate exponent tuples for polynomial monomials up to given order."""
        terms = []
        for deg in range(1, order + 1):
            def rec_build(prefix, k, remain):
                if k == 0:
                    if remain == 0:
                        terms.append(tuple(prefix))
                    return
                for val in range(remain + 1):
                    rec_build(prefix + [val], k - 1, remain - val)

            rec_build([], n, deg)
        return terms

    def initialize_rbf_centers(self, data: np.ndarray):
        """Initialize RBF centers from data.

        If requested centers exceed sample count, sampling with replacement is used.
        """
        if self.rbf_centers_initialized:
            return

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        n_samples = data.shape[0]
        n_centers = self.rbf_n_centers
        replace = (n_centers > n_samples)

        if self.rbf_center_method == 'random':
            indices = np.random.choice(n_samples, n_centers, replace=replace)
            self.rbf_centers = data[indices]

        elif self.rbf_center_method == 'uniform':
            if self.n <= 3:
                n_per_dim = int(np.ceil(n_centers ** (1.0 / self.n)))
                grid_axes = []
                for dim in range(self.n):
                    dim_min = data[:, dim].min()
                    dim_max = data[:, dim].max()
                    grid_axes.append(np.linspace(dim_min, dim_max, n_per_dim))

                if self.n == 1:
                    centers = grid_axes[0].reshape(-1, 1)
                elif self.n == 2:
                    xx, yy = np.meshgrid(grid_axes[0], grid_axes[1])
                    centers = np.column_stack([xx.ravel(), yy.ravel()])
                else:
                    xx, yy, zz = np.meshgrid(grid_axes[0], grid_axes[1], grid_axes[2])
                    centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

                if centers.shape[0] < n_centers:
                    extra = n_centers - centers.shape[0]
                    idx = np.random.choice(centers.shape[0], extra, replace=True)
                    self.rbf_centers = np.vstack([centers, centers[idx]])
                else:
                    self.rbf_centers = centers[:n_centers]
            else:
                indices = np.random.choice(n_samples, n_centers, replace=replace)
                self.rbf_centers = data[indices]

        elif self.rbf_center_method == 'kmeans':
            if n_centers > n_samples:
                kmeans = KMeans(n_clusters=n_samples, random_state=0, n_init=10)
                kmeans.fit(data)
                base_centers = kmeans.cluster_centers_
                extra = n_centers - n_samples
                idx = np.random.choice(n_samples, extra, replace=True)
                self.rbf_centers = np.vstack([base_centers, base_centers[idx]])
            else:
                kmeans = KMeans(n_clusters=n_centers, random_state=0, n_init=10)
                kmeans.fit(data)
                self.rbf_centers = kmeans.cluster_centers_
        else:
            raise ValueError(f"Unknown RBF center method: {self.rbf_center_method}")

        self.rbf_centers_initialized = True
        if replace:
            print(
                f"RBF centers initialized: {self.rbf_centers.shape[0]} centers "
                f"using '{self.rbf_center_method}' (with replacement)."
            )
        else:
            print(
                f"RBF centers initialized: {self.rbf_centers.shape[0]} centers "
                f"using '{self.rbf_center_method}'."
            )

    def phi(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute lifted feature vector for one sample."""
        feats = []

        if self.include_constant:
            feats.append(1.0)

        # Identity features (raw states first for easy inverse readout).
        feats.extend(x.tolist())

        for exps in self.poly_terms:
            val = 1.0
            for xi, e in zip(x, exps):
                if e != 0:
                    val *= (xi ** e)
            feats.append(val)

        if self.use_trig and self.n >= 3:
            theta = x[2]
            feats.append(np.cos(theta))
            feats.append(np.sin(theta))

        if self.use_cross_terms:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    feats.append(x[i] * x[j])

        if self.use_rbf and self.rbf_centers_initialized:
            for center in self.rbf_centers:
                dist_sq = np.sum((x - center) ** 2)
                feats.append(np.exp(-self.rbf_eta * dist_sq))

        if self.include_u_in_phi and u is not None:
            feats.extend(u.tolist())
            for ui in u:
                for p in range(2, self.poly_order + 1):
                    feats.append(ui ** p)

        return np.array(feats)

    def compute_lifted_states(
        self,
        states: np.ndarray,
        inputs: Optional[np.ndarray] = None,
        initialize_rbf: bool = True,
    ) -> np.ndarray:
        """Compute lifted features for a state/input sequence."""
        if self.use_rbf and initialize_rbf and not self.rbf_centers_initialized:
            self.initialize_rbf_centers(states)

        T = states.shape[0]
        phi_list = []
        for t in range(T):
            x = states[t]
            u = inputs[t] if inputs is not None else None
            phi_list.append(self.phi(x, u))

        return np.array(phi_list)

    def dim(self) -> int:
        """Return lifted feature dimension."""
        z = self.phi(
            np.zeros(self.n),
            np.zeros(self.m) if self.include_u_in_phi and self.m > 0 else None,
        )
        return z.size

    def get_info(self) -> dict:
        """Return lifting configuration summary."""
        return {
            'n_states': self.n,
            'n_inputs': self.m,
            'poly_order': self.poly_order,
            'use_trig': self.use_trig,
            'use_cross_terms': self.use_cross_terms,
            'use_rbf': self.use_rbf,
            'rbf_n_centers': self.rbf_n_centers if self.use_rbf else 0,
            'rbf_eta': self.rbf_eta if self.use_rbf else None,
            'rbf_center_method': self.rbf_center_method if self.use_rbf else None,
            'include_constant': self.include_constant,
            'include_u_in_phi': self.include_u_in_phi,
            'n_features': self.dim(),
        }
