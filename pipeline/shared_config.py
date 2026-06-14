import os

# Shared paths for the cleaned repository layout.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Data paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "full_ds_conf.csv")
DATA_WITH_TIME_PATH = os.path.join(ARTIFACTS_DIR, "time_indexed_dataset.csv")

# Artifact paths
SPLIT_DIR = os.path.join(ARTIFACTS_DIR, "time_splits")
SURROGATE_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "surrogate_models")
PREPARED_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "prepared_feature_dataset.joblib")
SURROGATE_MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "surrogate_training_manifest.json")
SURROGATE_EVAL_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "surrogate_evaluation_report.json")
CONFORMAL_BOUNDS_PATH = os.path.join(ARTIFACTS_DIR, "conformal_bounds_summary.json")
CRC_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "crc_classifiers")
CRC_MANIFEST_PATH = os.path.join(CRC_MODEL_DIR, "crc_classifier_manifest.json")

# Risk-to-cash simulation outputs (used by external plotting script)
RISK_TO_CASH_RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "risk_to_cash_results")
RISK_TO_CASH_FRONTIER_CSV_PATH = os.path.join(RISK_TO_CASH_RESULTS_DIR, "risk_to_cash_frontier.csv")
PARETO_EPSILON_FRONTIER_CSV_PATH = os.path.join(RISK_TO_CASH_RESULTS_DIR, "pareto_epsilon_frontier.csv")
RISK_TO_CASH_RESULTS_BUNDLE_PATH = os.path.join(RISK_TO_CASH_RESULTS_DIR, "risk_to_cash_results_bundle.joblib")

# Script paths
SURROGATE_TRAINER_PATH = os.path.join(SCRIPT_DIR, "surrogate_model.py")

# Shared modelling configuration
ALPHAS = [0.05, 0.07, 0.10, 0.15, 0.20]
TARGETS = ["TIT", "NOX", "CO"]
UPPER_TAU = 0.95
BASE_EXOG = ["AT", "AP", "AH", "AFDP", "TEY"]

# History/state feature configuration for surrogate models.
# These are lagged process states available at decision time (t-1 and earlier).
STATE_HISTORY_COLS = ["TEY", "TIT", "GTEP", "TAT", "CDP", "NOX"]
STATE_HISTORY_LAGS = [1, 2, 3]
