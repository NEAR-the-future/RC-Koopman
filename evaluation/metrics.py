"""Evaluation metrics and matrix diagnostics for Koopman-learning methods."""

import numpy as np
from typing import Dict, Optional, Tuple, List


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute NRMSE normalized by global standard deviation of the target."""
    rmse = compute_rmse(y_true, y_pred)
    std_true = np.std(y_true)
    if std_true < 1e-10:
        return 0.0
    return rmse / std_true


def compute_per_dim_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute NRMSE independently for each state dimension."""
    n_dims = y_true.shape[1]
    nrmse_per_dim = np.zeros(n_dims)

    for i in range(n_dims):
        rmse_i = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
        std_i = np.std(y_true[:, i])
        nrmse_per_dim[i] = 0.0 if std_i < 1e-10 else rmse_i / std_i

    return nrmse_per_dim


def compute_reconstruction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dim_names: Optional[List[str]] = None,
    valid_range: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Compute RMSE/NRMSE summary with optional valid prediction window."""
    if valid_range is not None:
        start_idx, end_idx = valid_range
        y_true_valid = y_true[start_idx:end_idx]
        y_pred_valid = y_pred[start_idx:end_idx]
    else:
        y_true_valid = y_true
        y_pred_valid = y_pred

    rmse = compute_rmse(y_true_valid, y_pred_valid)
    nrmse = compute_nrmse(y_true_valid, y_pred_valid)
    nrmse_per_dim = compute_per_dim_nrmse(y_true_valid, y_pred_valid)

    n_dims = y_true.shape[1]
    if dim_names is None:
        dim_names = [f"dim_{i+1}" for i in range(n_dims)]

    return {
        'rmse': rmse,
        'nrmse': nrmse,
        'nrmse_per_dim': nrmse_per_dim,
        'nrmse_mean': np.mean(nrmse_per_dim),
        'dim_names': dim_names,
        'valid_range': valid_range,
    }


def analyze_koopman_spectrum(K: np.ndarray) -> Dict:
    """Analyze eigen-spectrum and conditioning of Koopman matrix K (or its A block)."""
    n_rows = K.shape[0]
    n_cols = K.shape[1]

    # For controlled systems with K=[A,B], spectrum is computed from A.
    A = K[:, :n_rows]

    eigenvalues = np.linalg.eigvals(A)
    eigenvalue_magnitudes = np.abs(eigenvalues)

    spectral_radius = np.max(eigenvalue_magnitudes)
    k_condition_number = np.linalg.cond(K)
    a_condition_number = np.linalg.cond(A)

    n_unstable = np.sum(eigenvalue_magnitudes > 1.0)
    n_stable = np.sum(eigenvalue_magnitudes < 1.0)
    n_marginal = np.sum(np.abs(eigenvalue_magnitudes - 1.0) < 1e-6)

    matrix_rank = np.linalg.matrix_rank(A)

    return {
        'eigenvalues': eigenvalues,
        'eigenvalue_magnitudes': eigenvalue_magnitudes,
        'spectral_radius': spectral_radius,
        'condition_number': k_condition_number,
        'a_condition_number': a_condition_number,
        'matrix_rank': matrix_rank,
        'n_unstable_modes': n_unstable,
        'n_stable_modes': n_stable,
        'n_marginal_modes': n_marginal,
        'max_real_eigenvalue': np.max(eigenvalues.real),
        'max_imag_eigenvalue': np.max(np.abs(eigenvalues.imag)),
        'has_input': (n_cols > n_rows),
        'K_shape': K.shape,
        'A_shape': A.shape,
    }


def analyze_gram_condition(G: np.ndarray) -> Dict:
    """Analyze conditioning and rank of Gram matrix G."""
    condition_number = np.linalg.cond(G)
    matrix_rank = np.linalg.matrix_rank(G)

    singular_values = np.linalg.svd(G, compute_uv=False)
    min_singular_value = float(singular_values[-1])

    return {
        'condition_number': condition_number,
        'rank': matrix_rank,
        'matrix_rank': matrix_rank,
        'shape': G.shape,
        'min_singular_value': min_singular_value,
    }


def compare_methods(results_dict: Dict[str, Dict]) -> Dict:
    """Aggregate method-level scalar metrics into a comparison table dict."""
    method_names = list(results_dict.keys())

    nrmse_values = []
    spectral_radii = []
    koopman_cond_numbers = []
    a_cond_numbers = []
    gram_cond_numbers = []

    for name in method_names:
        result = results_dict[name]

        nrmse_values.append(result['reconstruction_errors']['nrmse'] if 'reconstruction_errors' in result else np.nan)

        if 'koopman_spectrum' in result:
            spectral_radii.append(result['koopman_spectrum']['spectral_radius'])
            koopman_cond_numbers.append(result['koopman_spectrum']['condition_number'])
            a_cond_numbers.append(result['koopman_spectrum'].get('a_condition_number', np.nan))
        else:
            spectral_radii.append(np.nan)
            koopman_cond_numbers.append(np.nan)
            a_cond_numbers.append(np.nan)

        if 'gram_condition' in result and result['gram_condition'] is not None:
            gram_cond_numbers.append(result['gram_condition']['condition_number'])
        else:
            gram_cond_numbers.append(np.nan)

    return {
        'method_names': method_names,
        'nrmse': nrmse_values,
        'spectral_radius': spectral_radii,
        'koopman_condition_number': koopman_cond_numbers,
        'a_condition_number': a_cond_numbers,
        'gram_condition_number': gram_cond_numbers,
        'best_nrmse_method': method_names[np.nanargmin(nrmse_values)] if nrmse_values else None,
        'best_stability_method': method_names[np.nanargmin(spectral_radii)] if spectral_radii else None,
    }


def print_evaluation_report(
    errors: Dict,
    spectrum_info: Optional[Dict] = None,
    gram_info: Optional[Dict] = None,
    method_name: str = "Method",
):
    """Pretty-print a method evaluation report to stdout."""
    print(f"\n{'='*70}")
    print(f"Evaluation Report: {method_name}")
    print(f"{'='*70}")

    print(f"\nReconstruction Errors:")
    print(f"  RMSE:  {errors['rmse']:.6e}")
    print(f"  NRMSE: {errors['nrmse']:.6e}")

    if 'nrmse_per_dim' in errors:
        print(f"\n  Per-dimension NRMSE:")
        for name, nrmse in zip(errors['dim_names'], errors['nrmse_per_dim']):
            print(f"    {name}: {nrmse:.6e}")
        print(f"    Mean: {errors['nrmse_mean']:.6e}")

    if spectrum_info is not None:
        print(f"\nKoopman Matrix Analysis:")
        print(f"  Spectral Radius: {spectrum_info['spectral_radius']:.6f}")
        print(f"  Condition Number: {spectrum_info['condition_number']:.2e}")
        print(f"  Unstable Modes (|lambda|>1): {spectrum_info['n_unstable_modes']}")
        print(f"  Stable Modes (|lambda|<1):   {spectrum_info['n_stable_modes']}")
        print(f"  Marginal Modes (|lambda|~1): {spectrum_info['n_marginal_modes']}")

    if gram_info is not None:
        print(f"\nGram Matrix Analysis:")
        print(f"  Condition Number: {gram_info['condition_number']:.2e}")
        print(f"  Rank: {gram_info['rank']} / {gram_info['shape'][0]}")
        print(f"  Min Singular Value: {gram_info['min_singular_value']:.2e}")

    print(f"\n{'='*70}\n")


def save_evaluation_to_csv(
    errors: Dict,
    spectrum_info: Optional[Dict],
    gram_info: Optional[Dict],
    filepath: str,
    method_name: str = "Method",
):
    """Save the evaluation report in CSV format."""
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Evaluation Results', method_name])
        writer.writerow([])

        writer.writerow(['Reconstruction Errors'])
        writer.writerow(['RMSE', errors['rmse']])
        writer.writerow(['NRMSE', errors['nrmse']])
        writer.writerow([])

        if 'nrmse_per_dim' in errors:
            writer.writerow(['Per-dimension NRMSE'])
            for name, nrmse in zip(errors['dim_names'], errors['nrmse_per_dim']):
                writer.writerow([name, nrmse])
            writer.writerow([])

        if spectrum_info is not None:
            writer.writerow(['Koopman Matrix Analysis'])
            writer.writerow(['Spectral Radius', spectrum_info['spectral_radius']])
            writer.writerow(['Condition Number', spectrum_info['condition_number']])
            writer.writerow(['Unstable Modes', spectrum_info['n_unstable_modes']])
            writer.writerow(['Stable Modes', spectrum_info['n_stable_modes']])
            writer.writerow([])

        if gram_info is not None:
            writer.writerow(['Gram Matrix Analysis'])
            writer.writerow(['Condition Number', gram_info['condition_number']])
            writer.writerow(['Rank', gram_info['rank']])
            writer.writerow([])


__all__ = [
    'compute_rmse',
    'compute_nrmse',
    'compute_per_dim_nrmse',
    'compute_reconstruction_errors',
    'analyze_koopman_spectrum',
    'analyze_gram_condition',
    'compare_methods',
    'print_evaluation_report',
    'save_evaluation_to_csv',
]
