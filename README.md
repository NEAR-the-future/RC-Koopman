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

```bibtex
@misc{gu2026koopmanidentificationnonlinearsystems,
      title={Koopman Identification of Nonlinear Systems via Reservoir Liftings}, 
      author={Weibin Gu and Chen Yang and Lu Shi},
      year={2026},
      eprint={2605.04917},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.04917}, 
}
```
