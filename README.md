# Cleaned Pipeline Layout

This repository has been reduced to the latest functional gas-turbine CRC/EMPC pipeline and its current results.

## Structure

- `pipeline/`
  - `surrogate_model.py`: trains surrogate models, conformal bounds, CRC classifiers, and journal tables
  - `RiskToCashFrontier.py`: runs the closed-loop risk-to-cash and epsilon-frontier simulations
  - `plot_risk_to_cash_results.py`: renders the full plot suite from the saved results bundle
  - `plot_target_risk_to_cash_frontier.py`: renders the compact target-wise frontier plot
  - `shared_config.py`: central paths and modeling configuration
- `data/`
  - `full_ds_conf.csv`: raw combined dataset used by the pipeline
- `results/`
  - trained model artifacts
  - time splits and prepared features
  - conformal and CRC artifacts
  - risk-to-cash tables, bundle, and plots
  - journal tables and summaries

## Run Order

1. `python3 pipeline/surrogate_model.py`
2. `python3 pipeline/RiskToCashFrontier.py`
3. `python3 pipeline/plot_risk_to_cash_results.py`

The current `results/` directory already contains the latest retained outputs from the refactored pipeline.
