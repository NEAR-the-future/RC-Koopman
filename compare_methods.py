

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np

from config_unified import GlobalConfig, DatasetConfig, MethodConfig
from datasets import DuffingDataset, DiffDriveDataset
from methods import RCKoopman, EDMDTraditional, HAVOK
from evaluation.metrics import (
    compute_reconstruction_errors,
    analyze_koopman_spectrum,
    analyze_gram_condition,
    compare_methods,
)
from evaluation.visualizer import Visualizer


DATASET_ALIASES = {
    "do": "duffing",
    "ddr": "diffdrive",
}


def _resolve_alias(value: str, alias_map: Dict[str, str], field_name: str) -> str:
    if value in alias_map:
        return alias_map[value]
    if value in alias_map.values():
        return value
    valid = list(alias_map.keys()) + list(alias_map.values())
    raise ValueError(f"{field_name} invalid: {value}. choices: {valid}")


def parse_args():
    parser = argparse.ArgumentParser(description="Open-source RC-Koopman benchmark runner.")
    parser.add_argument("-d", "--dataset", default=None, help="dataset: do/duffing, ddr/diffdrive")
    parser.add_argument("-s", "--seed", type=int, default=None, help="random seed")
    return parser.parse_args()


def run_single_method(method_name: str, method_obj, data_dict: Dict) -> Dict:
    print(f"\n{'='*60}\nRunning: {method_name}\n{'='*60}")
    method_obj.fit(data_dict)

    reconstruct_result = method_obj.reconstruct(
        data_dict["states"],
        data_dict["inputs"] if data_dict["has_input"] else None,
        method="one-step-ahead",
    )

    if isinstance(reconstruct_result, tuple):
        reconstructed_states, valid_range = reconstruct_result
    else:
        reconstructed_states, valid_range = reconstruct_result, None

    errors = compute_reconstruction_errors(
        data_dict["states"],
        reconstructed_states,
        valid_range=valid_range,
    )

    K = method_obj.get_koopman_matrix()
    spectrum_info = analyze_koopman_spectrum(K)

    try:
        G = method_obj.get_gramian_matrix()
        gram_info = analyze_gram_condition(G) if G is not None else None
    except Exception:
        gram_info = None

    print(
        f"NRMSE={errors['nrmse']:.6e}, RMSE={errors['rmse']:.6e}, "
        f"rho={spectrum_info['spectral_radius']:.6f}"
    )

    return {
        "method_name": method_name,
        "reconstructed_states": reconstructed_states,
        "reconstruction_errors": errors,
        "koopman_spectrum": spectrum_info,
        "gram_condition": gram_info,
        "valid_range": valid_range,
        "success": True,
    }


def load_dataset(dataset_name: str, seed: int):
    if dataset_name == "duffing":
        dataset = DuffingDataset(random_seed=seed)
    elif dataset_name == "diffdrive":
        dataset = DiffDriveDataset(random_seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset.generate_data()
    return dataset.get_training_data()


def save_summary_csv(comparison: Dict, successful_results: Dict[str, Dict], out_csv: str):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method",
            "RMSE",
            "NRMSE",
            "Spectral Radius",
            "K Condition Number",
            "A Condition Number",
            "G Condition Number",
        ])

        for i, name in enumerate(comparison["method_names"]):
            rmse = successful_results[name]["reconstruction_errors"]["rmse"]
            gcond = comparison["gram_condition_number"][i]
            writer.writerow([
                name,
                f"{rmse:.6e}",
                f"{comparison['nrmse'][i]:.6e}",
                f"{comparison['spectral_radius'][i]:.6f}",
                f"{comparison['koopman_condition_number'][i]:.4e}",
                f"{comparison['a_condition_number'][i]:.4e}",
                f"{gcond:.4e}" if not np.isnan(gcond) else "N/A",
            ])


def run_dataset(dataset_name: str, seed: int):
    DatasetConfig.dataset = dataset_name
    GlobalConfig.random_seed = seed
    MethodConfig.reconstruction_method = "one-step-ahead"

    if seed is not None:
        np.random.seed(seed)

    print(f"\nDataset={dataset_name}, Seed={seed}, Reconstruction=one-step-ahead")

    data_dict = load_dataset(dataset_name, seed)
    n_states = data_dict["n_states"]
    n_inputs = data_dict["n_inputs"]

    methods = {
        "RC_Koopman": RCKoopman(n_states, n_inputs, random_seed=seed),
        "EDMD": EDMDTraditional(n_states, n_inputs, random_seed=seed),
        "HAVOK": HAVOK(n_states, n_inputs, random_seed=seed),
    }

    results = {}
    for method_name, method_obj in methods.items():
        try:
            results[method_name] = run_single_method(method_name, method_obj, data_dict)
        except Exception as exc:
            results[method_name] = {"method_name": method_name, "success": False, "error": str(exc)}
            print(f"Failed: {method_name}: {exc}")

    successful_results = {k: v for k, v in results.items() if v.get("success")}
    if not successful_results:
        raise RuntimeError("All methods failed.")

    comparison = compare_methods(successful_results)

    dataset_tag = "DO" if dataset_name == "duffing" else "DDR"
    output_dir = os.path.join("./results", dataset_tag)
    os.makedirs(output_dir, exist_ok=True)
    vis = Visualizer(save_dir=output_dir)

    vis.plot_method_comparison_table(
        comparison,
        title=f"Method Comparison ({dataset_name})",
        save_name="comparison_table.png",
        show=False,
    )

    for method_name, result in successful_results.items():
        vis.plot_trajectory_comparison(
            data_dict["t"],
            data_dict["states"],
            result["reconstructed_states"],
            title=f"{method_name} Trajectory",
            save_name=f"{method_name}_trajectory.png",
            show=False,
            valid_range=result.get("valid_range"),
        )
        if data_dict["states"].shape[1] >= 2:
            vis.plot_phase_portrait_2d(
                data_dict["states"][:, :2],
                reconstructed_states=result["reconstructed_states"][:, :2],
                title=f"{method_name} Phase Portrait ({dataset_name})",
                xlabel="$x_1$",
                ylabel="$x_2$",
                save_name=f"{method_name}_phase_portrait.png",
                show=False,
            )

    eigenvalues_dict = {
        method_name: result["koopman_spectrum"]["eigenvalues"]
        for method_name, result in successful_results.items()
    }
    if len(eigenvalues_dict) >= 2:
        vis.plot_eigenvalues_combined(
            eigenvalues_dict,
            colors={"RC_Koopman": "tab:blue", "EDMD": "tab:green", "HAVOK": "tab:orange"},
            title=f"Combined Eigenvalues ({dataset_name})",
            save_name="combined_eigenvalues.png",
            show=False,
            max_radius=3.0,
        )

    save_summary_csv(comparison, successful_results, os.path.join(output_dir, "comparison_summary.csv"))

    print(f"\nSaved outputs to {output_dir}:")
    print("- comparison_summary.csv")
    print("- comparison_table.png")
    print("- combined_eigenvalues.png")
    print("- <METHOD>_trajectory.png")
    print("- <METHOD>_phase_portrait.png")


def main():
    args = parse_args()

    seed = GlobalConfig.random_seed
    if args.seed is not None:
        seed = args.seed

    if args.dataset is not None:
        dataset_list: List[str] = [_resolve_alias(args.dataset, DATASET_ALIASES, "dataset")]
    else:
        dataset_list = ["duffing", "diffdrive"]

    for dataset_name in dataset_list:
        run_dataset(dataset_name, seed)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
