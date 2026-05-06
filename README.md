# RC-Koopman

Python implementation of the RC-Koopman framework for data-driven modeling of nonlinear systems using stateful reservoir liftings.

## 🧭 What's included?
- Benchmark datasets (`datasets/`): `Duffing` (autonomous), `DiffDrive` (non-autonomous)
- Methods (`methods/`): `RC_Koopman`, `EDMD`, `HAVOK`
- Evaluation (`evaluation/`):
  - One-step-ahead reconstruction errors: `RMSE`, `NRMSE`
  - Koopman diagnostics: eigenvalues, spectral radius, condition numbers
  - Gramian diagnostics
  - Plots and CSV summaries (`results/`)
- `compare_methods.py`: main entry point
- `config_unified.py`: configuration file

## 🚀 Quick Start

Installation:

```bash
pip install -r requirements.txt
```

Run both datasets:
```bash
python compare_methods.py
```

Run one dataset only:
```bash
python compare_methods.py -d do -s 42
python compare_methods.py -d ddr -s 42
```

> Aliases: `do` stands for `duffing`, `ddr` stands for `diffdrive`.

## 📖 Citation
If you use this repository, please consider citing our work:

Gu, Weibin, Chen Yang, and Lu Shi. "Koopman Identification of Nonlinear Systems via Reservoir Liftings." arXiv preprint (2026).

```bibtex
@article{gu2026koopman,
  title={Koopman Identification of Nonlinear Systems via Reservoir Liftings},
  author={Gu, Weibin and Yang, Chen and Shi, Lu},
  journal={arXiv preprint},
  year={2026}
}
```
