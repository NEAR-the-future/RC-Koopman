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
If you use this repository, please cite:

> @misc{Gu2026RCKoopman,
>   author = {Gu, Weibin and Yang, Chen and Shi, Lu},
>   title = {RC-Koopman: Koopman Identification of Nonlinear Systems via Reservoir Liftings},
>   year = {2026},
>   publisher = {GitHub},
>   howpublished = {\url{https://github.com/NEAR-the-future/RC-Koopman}},
> }
