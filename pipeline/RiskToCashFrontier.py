# 1) Imports (collected for local runtime)
import json
import os
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from shared_config import (
    ALPHAS,
    ARTIFACTS_DIR,
    CONFORMAL_BOUNDS_PATH,
    CRC_MANIFEST_PATH,
    CRC_MODEL_DIR,
    PARETO_EPSILON_FRONTIER_CSV_PATH,
    PREPARED_FEATURES_PATH,
    RISK_TO_CASH_FRONTIER_CSV_PATH,
    RISK_TO_CASH_RESULTS_BUNDLE_PATH,
    RISK_TO_CASH_RESULTS_DIR,
    SURROGATE_MODEL_DIR,
    SURROGATE_TRAINER_PATH,
    TARGETS,
)

script_start_time = time.perf_counter()
# define simulation size
sim_set = None # can choose between 'none' aka the full test set, or a number from 1 to the test set
# single risk grid shared by training + EMPC simulations
PLOT_ALPHA = 0.1  # Representative alpha for single-alpha plots (falls back to ALPHAS[0] if missing)

# Local paths
STATIC_TEY_GRID_SIZE = 40  # number of TEY points in static grid
# Full CRC (independent-trajectory, closed-loop calibration)
CRC_NUM_TRAJECTORIES = 50  # number of independent trajectories for CRC calibration
CRC_TRAJECTORY_HOURS = 48  # hours per trajectory (None to use full CALIB length)
CRC_LAMBDA_GRID_SIZE = 61  # candidate λ grid size (quantiles of CALIB scores)
CRC_LOSS_MODE = "any_breach_rate"  # "any_breach_rate" (per-hour) or "any_breach_indicator" (per-trajectory)
CRC_SEED = 123  # RNG seed for CRC calibration trajectories
EPSILON_GRID_SIZE = 35  # ε grid resolution for Pareto sweeps
# ε-gate mode:
# - "predicted": uses point NOX prediction only
# - "hybrid": uses pred_NOX + k * q_NOX (k in [0,1])
# - "ub": uses full upper bound pred_NOX + q_NOX (legacy)
EPSILON_GATE_MODE = "predicted" # "predicted", "hybrid", or "ub"
EPSILON_UB_WEIGHT = 0.35  # used only when EPSILON_GATE_MODE == "hybrid"

# Master control for how strongly ε drives Pareto behavior.
# 1.0 keeps the configured base behavior.
# >1.0 makes ε stricter/more influential for NOX-oriented Pareto separation.
# <1.0 makes ε looser/less influential.
EPSILON_SENSITIVITY_DIAL = 1.5
EPSILON_BASE_Q_MIN = 0.25
EPSILON_BASE_Q_MAX = 0.75
EPSILON_BASE_GATE_MODE = EPSILON_GATE_MODE
EPSILON_BASE_UB_WEIGHT = EPSILON_UB_WEIGHT

# Speed preset: reduce resolution for much faster runs
FAST_MODE = False
if FAST_MODE:
    # Surrogate training is external in surrogate_model.py.
    sim_set = 500  # smaller test subset
    STATIC_TEY_GRID_SIZE = 15
    CRC_NUM_TRAJECTORIES = 20
    CRC_TRAJECTORY_HOURS = 24
    CRC_LAMBDA_GRID_SIZE = 10
    EPSILON_GRID_SIZE = 10

# Quick runtime estimate from a FAST_MODE baseline calibration.
# Baseline reference (about 110s):
#   sim_set=100, STATIC_TEY_GRID_SIZE=10, CRC_NUM_TRAJECTORIES=20,
#   CRC_TRAJECTORY_HOURS=24, CRC_LAMBDA_GRID_SIZE=10, EPSILON_GRID_SIZE=10.
RUNTIME_EST_BASE_SECONDS = 110.0
RUNTIME_EST_SIM_SET_NONE_ASSUMED = 5600
RUNTIME_EST_BASE = {
    "alphas": len(ALPHAS),
    "sim_set": 100,
    "tey_grid": 10,
    "crc_traj": 20,
    "crc_hours": 24,
    "crc_lambda": 10,
    "eps_grid": 10,
}

def _estimate_runtime_seconds(sim_hours: int | None = None) -> float | None:
    if CRC_TRAJECTORY_HOURS is None:
        return None

    a = int(len(ALPHAS))
    if sim_set is None:
        s = int(sim_hours) if sim_hours is not None else int(RUNTIME_EST_SIM_SET_NONE_ASSUMED)
    else:
        s = int(sim_set)
    g = int(STATIC_TEY_GRID_SIZE)
    r = int(CRC_NUM_TRAJECTORIES)
    h = int(CRC_TRAJECTORY_HOURS)
    l = int(CRC_LAMBDA_GRID_SIZE)
    e = int(EPSILON_GRID_SIZE)

    # Rough work model split by major blocks in this script.
    work_crc = a * r * h * l * g
    work_frontier = a * s * g
    work_epsilon = a * e * s * g
    work_total = float(work_crc + work_frontier + work_epsilon)

    b = RUNTIME_EST_BASE
    base_crc = b["alphas"] * b["crc_traj"] * b["crc_hours"] * b["crc_lambda"] * b["tey_grid"]
    base_frontier = b["alphas"] * b["sim_set"] * b["tey_grid"]
    base_epsilon = b["alphas"] * b["eps_grid"] * b["sim_set"] * b["tey_grid"]
    base_total = float(base_crc + base_frontier + base_epsilon)
    if base_total <= 0:
        return None
    return float(RUNTIME_EST_BASE_SECONDS * (work_total / base_total))

def _print_runtime_estimate(sim_hours: int | None = None) -> None:
    est = _estimate_runtime_seconds(sim_hours=sim_hours)
    if est is None:
        print("Estimated runtime: unavailable (requires numeric CRC_TRAJECTORY_HOURS).")
        return
    if sim_set is None and sim_hours is None:
        sim_label = f"assuming sim_set={RUNTIME_EST_SIM_SET_NONE_ASSUMED}"
    elif sim_set is None:
        sim_label = f"using full test size sim_set={int(sim_hours)}"
    else:
        sim_label = f"sim_set={int(sim_set)}"
    print(
        "Estimated runtime (rough): "
        f"{est:.1f}s (~{est / 60.0:.2f} min) "
        f"from FAST_MODE baseline {RUNTIME_EST_BASE_SECONDS:.0f}s ({sim_label})."
    )

def _resolve_epsilon_sensitivity_controls(dial: float) -> dict:
    d = max(0.0, float(dial))
    delta = d - 1.0

    # Higher dial tightens ε range toward lower NOX quantiles.
    q_min = float(np.clip(EPSILON_BASE_Q_MIN - 0.10 * delta, 0.05, 0.70))
    q_max = float(np.clip(EPSILON_BASE_Q_MAX - 0.20 * delta, q_min + 0.08, 0.95))

    # Keep base behavior at dial=1.0; only switch predicted->hybrid once dial is clearly >1.
    gate_mode = str(EPSILON_BASE_GATE_MODE)
    if gate_mode == "predicted" and d > 1.2:
        gate_mode = "hybrid"

    ub_weight = float(np.clip(EPSILON_BASE_UB_WEIGHT + 0.30 * delta, 0.0, 1.0))

    # Extra NOX pressure inside ε-sweeps only (1.0 means no change).
    nox_objective_weight = float(np.clip(1.0 + 1.5 * max(delta, 0.0), 1.0, 5.0))

    return {
        "dial": d,
        "eps_q_min": q_min,
        "eps_q_max": q_max,
        "gate_mode": gate_mode,
        "ub_weight": ub_weight,
        "nox_objective_weight": nox_objective_weight,
    }

EPSILON_CONTROLS = _resolve_epsilon_sensitivity_controls(EPSILON_SENSITIVITY_DIAL)
EPSILON_EFFECTIVE_Q_MIN = EPSILON_CONTROLS["eps_q_min"]
EPSILON_EFFECTIVE_Q_MAX = EPSILON_CONTROLS["eps_q_max"]
EPSILON_EFFECTIVE_GATE_MODE = EPSILON_CONTROLS["gate_mode"]
EPSILON_EFFECTIVE_UB_WEIGHT = EPSILON_CONTROLS["ub_weight"]
EPSILON_OBJECTIVE_NOX_WEIGHT = EPSILON_CONTROLS["nox_objective_weight"]

_print_runtime_estimate()
print(
    "Epsilon master dial:",
    f"dial={EPSILON_SENSITIVITY_DIAL:.2f},",
    f"q_range=[{EPSILON_EFFECTIVE_Q_MIN:.3f}, {EPSILON_EFFECTIVE_Q_MAX:.3f}],",
    f"gate_mode={EPSILON_EFFECTIVE_GATE_MODE},",
    f"ub_weight={EPSILON_EFFECTIVE_UB_WEIGHT:.2f},",
    f"pareto_nox_weight={EPSILON_OBJECTIVE_NOX_WEIGHT:.2f}",
)

"""load precomputed prepared features from surrogate_model.py run"""

def _load_prepared_feature_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing prepared feature artifact. Generate it first by running:\n"
            f"  python \"{SURROGATE_TRAINER_PATH}\"\n"
            f"Missing file:\n - {path}"
        )
    prepared = joblib.load(path)
    required_keys = ("df", "X", "mask_train", "mask_calib", "mask_test", "features", "feature_pos")
    missing_keys = [k for k in required_keys if k not in prepared]
    if missing_keys:
        raise KeyError(
            "Prepared feature artifact is incomplete. Re-run surrogate_model.py.\n"
            f"Missing keys: {missing_keys}"
        )
    return prepared

prepared = _load_prepared_feature_artifact(PREPARED_FEATURES_PATH)
df = prepared["df"]
X = prepared["X"]
m_train = np.asarray(prepared["mask_train"], dtype=bool)
m_calib = np.asarray(prepared["mask_calib"], dtype=bool)
m_test = np.asarray(prepared["mask_test"], dtype=bool)
FEATURES = list(prepared["features"])
FEATURE_POS = dict(prepared["feature_pos"])

def _extract_history_spec(feature_names: list[str]) -> tuple[list[str], list[int]]:
    pat = re.compile(r"^([A-Za-z0-9]+)_lag(\d+)$")
    cols = []
    lags = []
    for name in feature_names:
        m = pat.match(name)
        if not m:
            continue
        cols.append(m.group(1))
        lags.append(int(m.group(2)))
    uniq_cols = []
    seen_cols = set()
    for c in cols:
        if c in seen_cols:
            continue
        seen_cols.add(c)
        uniq_cols.append(c)
    uniq_lags = sorted(set(lags))
    return uniq_cols, uniq_lags


def _aligned_frame_for_model(model, frame: pd.DataFrame) -> pd.DataFrame:
    feat_names = getattr(model, "feature_name_", None)
    if feat_names is None or len(feat_names) == 0:
        return frame
    missing = [f for f in feat_names if f not in frame.columns]
    if missing:
        raise KeyError(f"Missing model features for prediction: {missing[:10]}")
    return frame[list(feat_names)]


def _build_model_feature_index(model, feature_pos: dict, model_label: str) -> np.ndarray:
    feat_names = getattr(model, "feature_name_", None)
    if feat_names is None or len(feat_names) == 0:
        raise ValueError(
            f"{model_label} has no feature_name_. "
            "Re-train artifacts so model feature names are available."
        )
    missing = [f for f in feat_names if f not in feature_pos]
    if missing:
        raise KeyError(f"{model_label} uses features missing from prepared artifact: {missing[:10]}")
    return np.asarray([feature_pos[f] for f in feat_names], dtype=np.int32)


_FEATURE_NAME_WARNING_MSG = r"X does not have valid feature names, but .* was fitted with feature names"


def _predict_regressor_fast(model, x_np: np.ndarray) -> np.ndarray:
    # Fast path intentionally passes numpy arrays after explicit feature-index alignment.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_FEATURE_NAME_WARNING_MSG, category=UserWarning)
        return model.predict(x_np)


def _predict_proba_fast(model, x_np: np.ndarray) -> np.ndarray:
    # Same rationale as _predict_regressor_fast.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_FEATURE_NAME_WARNING_MSG, category=UserWarning)
        return model.predict_proba(x_np)


HISTORY_STATE_COLS, HISTORY_STATE_LAGS = _extract_history_spec(FEATURES)
if HISTORY_STATE_COLS and HISTORY_STATE_LAGS:
    print(
        "Detected state-history features:",
        f"cols={HISTORY_STATE_COLS}, lags={HISTORY_STATE_LAGS}",
    )


def _initialize_lagged_state(seed_row: pd.Series, df_context: pd.DataFrame) -> dict:
    if not HISTORY_STATE_COLS or not HISTORY_STATE_LAGS:
        return {}
    lagged = {}
    max_lag = int(max(HISTORY_STATE_LAGS))
    seed_idx = seed_row.name
    can_use_context = isinstance(seed_idx, (int, np.integer)) and seed_idx in df_context.index
    for col in HISTORY_STATE_COLS:
        fallback_val = float(seed_row[col]) if col in seed_row else np.nan
        # Keep a dense lag history internally so sparse lag sets
        # (e.g., [1, 2, 3, 6, 12, 24]) propagate exactly across hours.
        hist = []
        for lag in range(1, max_lag + 1):
            if can_use_context:
                src_idx = int(seed_idx) - (lag - 1)
                if src_idx in df_context.index and col in df_context.columns:
                    hist.append(float(df_context.loc[src_idx, col]))
                    continue
            hist.append(fallback_val)
        lagged[f"__hist__{col}"] = hist
        for lag in HISTORY_STATE_LAGS:
            lagged[f"{col}_lag{lag}"] = float(hist[lag - 1])
    return lagged


def _advance_lagged_state(current_state: dict, latest_values: dict) -> dict:
    if not HISTORY_STATE_COLS or not HISTORY_STATE_LAGS:
        return current_state
    next_state = dict(current_state)
    max_lag = int(max(HISTORY_STATE_LAGS))
    lag1_default = np.nan
    for col in HISTORY_STATE_COLS:
        hist_key = f"__hist__{col}"
        prev_hist = current_state.get(hist_key)
        if isinstance(prev_hist, (list, tuple, np.ndarray)):
            hist = [float(v) if pd.notna(v) else np.nan for v in list(prev_hist)[:max_lag]]
        else:
            # Backward-compatibility for runs started before dense-history state existed.
            lag1_key = f"{col}_lag1"
            lag1_default = float(current_state.get(lag1_key, np.nan))
            hist = [lag1_default] * max_lag

        if len(hist) < max_lag:
            fill = hist[-1] if hist else lag1_default
            hist.extend([fill] * (max_lag - len(hist)))

        latest_val = float(latest_values.get(col, np.nan))
        new_hist = [latest_val] + hist[: max_lag - 1]
        next_state[hist_key] = new_hist

        for lag in HISTORY_STATE_LAGS:
            next_state[f"{col}_lag{lag}"] = float(new_hist[lag - 1])
    return next_state

# Build splits
train_df = df[m_train]
calib_df = df[m_calib]

"""Surrogate training is external (surrogate_model.py) and features are loaded from artifacts."""

print(f"Using pre-trained surrogate artifacts from {ARTIFACTS_DIR}")

"""### EMPC"""

start_time = time.time()


# here we begin the EMPC process by loading in the models and conformal bounds from the previous steps
# 1. Paths, constants, and alpha grid (aligned with previous runs)
RNG = np.random.default_rng(42)

def _assert_surrogate_artifacts_ready(targets: list[str]) -> None:
    missing = []
    if not os.path.exists(PREPARED_FEATURES_PATH):
        missing.append(PREPARED_FEATURES_PATH)
    if not os.path.exists(CONFORMAL_BOUNDS_PATH):
        missing.append(CONFORMAL_BOUNDS_PATH)
    if not os.path.exists(CRC_MANIFEST_PATH):
        missing.append(CRC_MANIFEST_PATH)
    for target in targets:
        point_path = os.path.join(SURROGATE_MODEL_DIR, target, "point_regressor.joblib")
        if not os.path.exists(point_path):
            missing.append(point_path)
        crc_clf_path = os.path.join(CRC_MODEL_DIR, f"{target}_breach_probability_classifier.joblib")
        if not os.path.exists(crc_clf_path):
            missing.append(crc_clf_path)
    if missing:
        missing_lines = "\n".join(f" - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing surrogate/CRC artifacts. Train them first by running:\n"
            f"  python \"{SURROGATE_TRAINER_PATH}\"\n"
            f"Missing files:\n{missing_lines}"
        )

# 2. Load conformal bounds and models
_assert_surrogate_artifacts_ready(TARGETS)
with open(CONFORMAL_BOUNDS_PATH, "r") as f:
    conformal_results = json.load(f)

bounds_alpha_grid = conformal_results.get("alpha_grid")
if not bounds_alpha_grid:
    raise ValueError("conformal_bounds_summary.json missing 'alpha_grid'; retrain conformal bounds.")

missing_alphas = [a for a in ALPHAS if a not in bounds_alpha_grid]
if missing_alphas:
    raise ValueError(
        f"Requested ALPHAS {ALPHAS} contain values not present in saved conformal bounds grid "
        f"{bounds_alpha_grid}. Rerun training with this grid or align ALPHAS to the stored bounds."
    )

if ALPHAS != bounds_alpha_grid:
    print(
        f"ALPHAS {ALPHAS} differ from conformal bounds grid {bounds_alpha_grid}; "
        "using stored grid indices for quantile lookup."
    )

ALPHA_TO_IDX = {a: bounds_alpha_grid.index(a) for a in ALPHAS}

point_models = {}
conformal_quantiles = {}

for target in TARGETS:
    tgt_dir = os.path.join(SURROGATE_MODEL_DIR, target)
    point_model_path = os.path.join(tgt_dir, "point_regressor.joblib")

    point_models[target] = joblib.load(point_model_path)

    conformal_quantiles[target] = {
        "point_residual_quantiles": conformal_results["targets"][target]["point_residual_quantiles"],
        "cqr_score_quantiles": conformal_results["targets"][target]["cqr_score_quantiles"],
    }

# 4. Load precomputed data/features prepared during surrogate training
X_calib = X[m_calib]
X_test  = X[m_test]
if sim_set is None:
    _print_runtime_estimate(sim_hours=len(X_test))

# 5. Data-driven TEY bounds and proxy limits (hourly, proxy basis)

# Use central quantiles of TEY to avoid extreme extrapolation
TEY_min = float(train_df["TEY"].quantile(0.05))
TEY_max = float(train_df["TEY"].quantile(0.95))

def build_tey_candidates(
    tey_min: float,
    tey_max: float,
    base_count: int,
) -> np.ndarray:
    """Returns a static TEY candidate grid."""
    return np.linspace(tey_min, tey_max, num=base_count)


TEY_CANDIDATES = build_tey_candidates(
    tey_min=TEY_min,
    tey_max=TEY_max,
    base_count=STATIC_TEY_GRID_SIZE,
)

# Proxy safety limits for NOX and TIT and CO – dataset basis only
limit_NOX = float(train_df["NOX"].quantile(0.95))
limit_TIT = float(train_df["TIT"].quantile(0.95))
limit_CO  = float(train_df["CO"].quantile(0.95))

print(f"Data-driven TEY bounds: [{TEY_min:.1f}, {TEY_max:.1f}]")
print(f"Proxy limits: NOX={limit_NOX:.2f}, TIT={limit_TIT:.2f}, CO={limit_CO:.2f}")

# 5.1 CRC: load pre-trained per-pollutant breach classifiers from surrogate_model.py
with open(CRC_MANIFEST_PATH, "r") as f:
    crc_manifest = json.load(f)

saved_limits = crc_manifest.get("proxy_limits", {})
for pollutant, runtime_limit in (("NOX", limit_NOX), ("TIT", limit_TIT), ("CO", limit_CO)):
    if pollutant in saved_limits and not np.isclose(float(saved_limits[pollutant]), float(runtime_limit), atol=1e-9, rtol=0.0):
        print(
            f"Warning: runtime proxy limit for {pollutant} ({runtime_limit:.6f}) differs from "
            f"CRC-training limit ({float(saved_limits[pollutant]):.6f})."
        )

breach_clfs = {}
for pollutant in TARGETS:
    clf_path = os.path.join(CRC_MODEL_DIR, f"{pollutant}_breach_probability_classifier.joblib")
    breach_clfs[pollutant] = joblib.load(clf_path)

POINT_MODEL_FEATURE_IDX = {
    target: _build_model_feature_index(point_models[target], FEATURE_POS, f"point model [{target}]")
    for target in TARGETS
}
CRC_MODEL_FEATURE_IDX = {
    target: _build_model_feature_index(breach_clfs[target], FEATURE_POS, f"CRC classifier [{target}]")
    for target in TARGETS
}
print("Inner-loop optimization path: NumPy candidate matrix (fast)")

train_breach_rates = {
    "NOX": float((train_df["NOX"] > limit_NOX).mean()),
    "TIT": float((train_df["TIT"] > limit_TIT).mean()),
    "CO": float((train_df["CO"] > limit_CO).mean()),
}
print(
    "Raw TRAIN breach rates:",
    "NOX", train_breach_rates["NOX"],
    "TIT", train_breach_rates["TIT"],
    "CO", train_breach_rates["CO"],
)

# CRC uses CALIB scores to build a λ grid (full CRC calibrates via rollouts)
calib_scores_nox = breach_clfs["NOX"].predict_proba(_aligned_frame_for_model(breach_clfs["NOX"], X_calib))[:, 1]
calib_scores_tit = breach_clfs["TIT"].predict_proba(_aligned_frame_for_model(breach_clfs["TIT"], X_calib))[:, 1]
calib_scores_co = breach_clfs["CO"].predict_proba(_aligned_frame_for_model(breach_clfs["CO"], X_calib))[:, 1]

def build_lambda_grid(scores: np.ndarray, grid_size: int) -> np.ndarray:
    """Builds a candidate λ grid from score quantiles (in [0,1])."""
    scores = np.asarray(scores)
    if scores.size == 0:
        raise ValueError("Empty score array; cannot build λ grid.")
    grid_size = max(3, int(grid_size))
    qs = np.linspace(0.01, 0.99, grid_size)
    grid = np.unique(np.quantile(scores, qs))
    grid = np.clip(grid, 0.0, 1.0)
    if grid.size == 1:
        grid = np.unique(np.array([0.0, float(grid[0]), 1.0]))
    return np.sort(grid)

def _compute_any_breach_loss(sim_df: pd.DataFrame, mode: str) -> float:
    breach_cols = ["sim_nox_breach", "sim_tit_breach", "sim_co_breach"]
    if "sim_TIT" in sim_df.columns:
        feasible_mask = sim_df["sim_TIT"].notna().to_numpy()
    else:
        feasible_mask = np.ones(len(sim_df), dtype=bool)

    if not np.any(feasible_mask):
        # No feasible operating points in this rollout; treat realised breach loss as zero.
        return 0.0

    any_breach = (
        sim_df.loc[feasible_mask, breach_cols]
        .eq(True)
        .any(axis=1)
    )
    if mode == "any_breach_indicator":
        return float(any_breach.any())
    if mode == "any_breach_rate":
        return float(any_breach.mean())
    raise ValueError(f"Unknown CRC_LOSS_MODE: {mode}")

def build_iid_trajectories(
    df_source: pd.DataFrame,
    init_source: pd.DataFrame,
    n_traj: int,
    horizon: int,
    rng: np.random.Generator,
):
    """Samples independent trajectories (with replacement) and initial lag rows."""
    n_traj = int(n_traj)
    horizon = int(horizon)
    if n_traj <= 0 or horizon <= 0:
        raise ValueError("CRC_NUM_TRAJECTORIES and CRC_TRAJECTORY_HOURS must be positive.")
    if len(df_source) == 0 or len(init_source) == 0:
        raise ValueError("Source data for CRC trajectories/initialization is empty.")

    trajs = []
    init_rows = []
    for _ in range(n_traj):
        idx = rng.integers(0, len(df_source), size=horizon)
        traj = df_source.iloc[idx].copy().reset_index(drop=True)
        trajs.append(traj)
        init_idx = int(rng.integers(0, len(init_source)))
        init_rows.append(init_source.iloc[init_idx])
    return trajs, init_rows

def calibrate_crc_threshold_any_full(
    alpha: float,
    trajectories: list,
    init_rows: list,
    lambda_grid: np.ndarray,
    limit_NOX: float,
    limit_TIT: float,
    limit_CO: float,
    point_models: dict,
    conformal_quantiles: dict,
    residuals_map: dict,
    breach_clfs: dict,
    crc_tey_candidates: np.ndarray,
    feature_pos: dict,
    alpha_idx_map: dict,
    loss_mode: str,
    monotone: bool,
    base_seed: int,
) -> float:
    """
    Full CRC calibration for a single λ controlling an any-breach gate.
    Each trajectory is an independent sample; loss is computed per trajectory.
    """
    if lambda_grid.size == 0:
        raise ValueError("Empty λ grid for CRC calibration.")
    n_traj = len(trajectories)
    n_lambda = len(lambda_grid)
    if n_traj == 0:
        raise ValueError("No CRC trajectories provided.")

    losses = np.zeros((n_traj, n_lambda), dtype=float)
    for t_idx, traj_df in enumerate(trajectories):
        init_row = init_rows[t_idx]
        X_dummy = pd.DataFrame(index=traj_df.index)
        traj_seed = int(base_seed + t_idx * 10007)

        for l_idx, lam in enumerate(lambda_grid):
            rng_local = np.random.default_rng(traj_seed)
            crc_thresholds_tmp = {"ANY": {alpha: float(lam)}}
            sim_df = _simulate_empc_core(
                alpha=alpha,
                X_test_subset=X_dummy,
                limit_NOX=limit_NOX,
                limit_TIT=limit_TIT,
                limit_CO=limit_CO,
                point_models=point_models,
                conformal_quantiles=conformal_quantiles,
                df_full=traj_df,
                m_calib=None,
                residuals_map=residuals_map,
                breach_clfs=breach_clfs,
                crc_thresholds=crc_thresholds_tmp,
                tey_candidates=crc_tey_candidates,
                feature_pos=feature_pos,
                alpha_idx_map=alpha_idx_map,
                epsilon_nox=None,
                include_real_fields=False,
                attach_epsilon=False,
                return_dataframe=True,
                init_row=init_row,
                rng=rng_local,
            )
            losses[t_idx, l_idx] = _compute_any_breach_loss(sim_df, loss_mode)

    if monotone:
        losses = np.maximum.accumulate(losses, axis=1)

    risks = losses.mean(axis=0)
    n = n_traj
    corrected = (1.0 + n * risks) / (n + 1.0)
    ok = np.where(corrected <= alpha)[0]
    if ok.size:
        return float(lambda_grid[ok[-1]])
    return float(lambda_grid[0])

# Pack for downstream use
score_any_calib = np.maximum.reduce([calib_scores_nox, calib_scores_tit, calib_scores_co])
lambda_grid_any = build_lambda_grid(score_any_calib, CRC_LAMBDA_GRID_SIZE)


# 6. Economic cost model (placeholders; can be refined)

def C_fuel(TEY: float) -> float:
    """Placeholder fuel cost: proportional to energy produced."""
    return 0.5 * TEY

def C_wear(TIT_pred: float) -> float:
    """Placeholder turbine wear cost: grows with TIT."""
    return 0.1 * TIT_pred

def C_env(NOX_pred: float, CO_pred: float, nox_weight: float = 1.0) -> float:
    """Placeholder environmental penalty based on NOX and CO."""
    return 0.2 * float(nox_weight) * NOX_pred + 0.05 * CO_pred

def economic_objective(
    TEY: float,
    TIT_pred: float,
    NOX_pred: float,
    CO_pred: float,
    price: float = 10.0,
    nox_weight: float = 1.0,
) -> float:
    """
    Economic objective for one hour:
    revenue - fuel cost - wear cost - environmental cost.

    Note: price and coefficients are placeholders (arbitrary units).
    """
    revenue = price * TEY
    return revenue - C_fuel(TEY) - C_wear(TIT_pred) - C_env(NOX_pred, CO_pred, nox_weight=nox_weight)

# 7. Empirical residuals for model-based plant simulation (Option B)
# Compute residuals on calibration set: y_calib - yhat_calib
residuals_map = {}
for target in TARGETS:
    y_cal = calib_df[target].to_numpy()
    yhat_cal = point_models[target].predict(_aligned_frame_for_model(point_models[target], X_calib))
    residuals_map[target] = (y_cal - yhat_cal).astype(float)

# 9. One-hour EMPC optimization (TEY search with conformal safety + CRC)

# Shared inner optimizer for baseline and ε-constraint variants
def _run_one_hour_optimization_core(
    current_hour_data: pd.Series,
    lagged_values: dict,
    alpha: float,
    limit_NOX: float,
    limit_TIT: float,
    limit_CO: float,
    point_models: dict,
    conformal_quantiles: dict,
    breach_clfs: dict,        # CRC classifiers per pollutant
    crc_thresholds: dict,     # CRC thresholds for any-breach gate per alpha
    tey_candidates: np.ndarray,
    feature_pos: dict,
    alpha_idx_map: dict,
    epsilon_nox: float | None = None,
) -> tuple:
    """Core EMPC optimizer; epsilon_nox optionally enforces a NOX cap."""
    if alpha not in alpha_idx_map:
        raise ValueError(f"Alpha {alpha} not in grid {list(alpha_idx_map.keys())}")

    alpha_idx = alpha_idx_map[alpha]
    q_TIT = conformal_quantiles["TIT"]["point_residual_quantiles"][alpha_idx]
    q_NOX = conformal_quantiles["NOX"]["point_residual_quantiles"][alpha_idx]
    q_CO  = conformal_quantiles["CO"]["point_residual_quantiles"][alpha_idx]
    crc_thr_any = crc_thresholds["ANY"][alpha]

    hod = float(current_hour_data["hour_idx"] % 24)
    hod_sin = float(np.sin(2 * np.pi * hod / 24.0))
    hod_cos = float(np.cos(2 * np.pi * hod / 24.0))

    base_row = np.zeros(len(feature_pos), dtype=float)

    def set_feat(name, value):
        idx = feature_pos.get(name)
        if idx is not None:
            base_row[idx] = float(value)

    set_feat("AT", current_hour_data.get("AT", np.nan))
    set_feat("AP", current_hour_data.get("AP", np.nan))
    set_feat("AH", current_hour_data.get("AH", np.nan))
    set_feat("AFDP", current_hour_data.get("AFDP", np.nan))

    for lag_col, lag_val in lagged_values.items():
        set_feat(lag_col, lag_val)

    set_feat("hour_of_day", hod)
    set_feat("hod_sin", hod_sin)
    set_feat("hod_cos", hod_cos)

    n_cand = len(tey_candidates)
    Xcand = np.empty((n_cand, len(feature_pos)), dtype=np.float64)
    Xcand[:] = base_row
    Xcand[:, feature_pos["TEY"]] = tey_candidates

    pred_TIT = _predict_regressor_fast(point_models["TIT"], Xcand[:, POINT_MODEL_FEATURE_IDX["TIT"]])
    pred_NOX = _predict_regressor_fast(point_models["NOX"], Xcand[:, POINT_MODEL_FEATURE_IDX["NOX"]])
    pred_CO = _predict_regressor_fast(point_models["CO"], Xcand[:, POINT_MODEL_FEATURE_IDX["CO"]])

    breach_scores_nox = _predict_proba_fast(breach_clfs["NOX"], Xcand[:, CRC_MODEL_FEATURE_IDX["NOX"]])[:, 1]
    breach_scores_tit = _predict_proba_fast(breach_clfs["TIT"], Xcand[:, CRC_MODEL_FEATURE_IDX["TIT"]])[:, 1]
    breach_scores_co = _predict_proba_fast(breach_clfs["CO"], Xcand[:, CRC_MODEL_FEATURE_IDX["CO"]])[:, 1]
    score_any = np.maximum.reduce([breach_scores_nox, breach_scores_tit, breach_scores_co])

    UB_TIT = pred_TIT + q_TIT
    UB_NOX = pred_NOX + q_NOX
    UB_CO  = pred_CO  + q_CO

    mask_nox = (UB_NOX <= limit_NOX)
    mask_tit = (UB_TIT <= limit_TIT)
    mask_co = (UB_CO <= limit_CO)
    mask_crc = (score_any <= crc_thr_any)
    if epsilon_nox is not None:
        if EPSILON_EFFECTIVE_GATE_MODE == "predicted":
            eps_gate_nox = pred_NOX
        elif EPSILON_EFFECTIVE_GATE_MODE == "hybrid":
            eps_gate_nox = pred_NOX + float(EPSILON_EFFECTIVE_UB_WEIGHT) * q_NOX
        elif EPSILON_EFFECTIVE_GATE_MODE == "ub":
            eps_gate_nox = UB_NOX
        else:
            raise ValueError(
                f"Unknown EPSILON_EFFECTIVE_GATE_MODE: {EPSILON_EFFECTIVE_GATE_MODE}. "
                "Use one of: 'predicted', 'hybrid', 'ub'."
            )
        mask_eps = (eps_gate_nox <= epsilon_nox)
    else:
        eps_gate_nox = pred_NOX
        mask_eps = np.ones_like(mask_nox, dtype=bool)

    feasible = mask_nox & mask_tit & mask_co & mask_crc & mask_eps

    if not feasible.any():
        fail_rates = {
            "nox_ub": float((~mask_nox).mean()),
            "tit_ub": float((~mask_tit).mean()),
            "co_ub": float((~mask_co).mean()),
            "crc_gate": float((~mask_crc).mean()),
            "epsilon_gate": float((~mask_eps).mean()) if epsilon_nox is not None else 0.0,
        }
        primary_blocker = max(fail_rates, key=fail_rates.get)
        diagnostics = {
            "optimizer_status": "infeasible",
            "dominant_constraint": "infeasible",
            "primary_blocker": primary_blocker,
            "n_feasible_candidates": 0,
            "candidate_count": int(n_cand),
            "active_nox_constraint": False,
            "active_tit_constraint": False,
            "active_co_constraint": False,
            "active_crc_constraint": False,
            "active_eps_constraint": False,
        }
        return (
            None,
            np.nan,
            {"TIT": np.nan, "NOX": np.nan, "CO": np.nan},
            {"TIT": np.nan, "NOX": np.nan, "CO": np.nan},
            diagnostics,
        )

    objective_nox_weight = EPSILON_OBJECTIVE_NOX_WEIGHT if epsilon_nox is not None else 1.0
    objs = np.asarray(
        economic_objective(
            tey_candidates,
            pred_TIT,
            pred_NOX,
            pred_CO,
            nox_weight=objective_nox_weight,
        )
    )
    objs[~feasible] = -np.inf

    best_idx = int(np.argmax(objs))
    best_tey = float(tey_candidates[best_idx])
    best_obj = float(objs[best_idx])

    best_preds = {
        "TIT": float(pred_TIT[best_idx]),
        "NOX": float(pred_NOX[best_idx]),
        "CO":  float(pred_CO[best_idx]),
    }
    best_ubs = {
        "TIT": float(UB_TIT[best_idx]),
        "NOX": float(UB_NOX[best_idx]),
        "CO":  float(UB_CO[best_idx]),
    }

    slack_nox = float(limit_NOX - UB_NOX[best_idx])
    slack_tit = float(limit_TIT - UB_TIT[best_idx])
    slack_co = float(limit_CO - UB_CO[best_idx])
    slack_crc = float(crc_thr_any - score_any[best_idx])
    slack_eps = float(epsilon_nox - eps_gate_nox[best_idx]) if epsilon_nox is not None else np.nan

    norm_slacks = {
        "nox_ub": slack_nox / max(abs(limit_NOX), 1e-9),
        "tit_ub": slack_tit / max(abs(limit_TIT), 1e-9),
        "co_ub": slack_co / max(abs(limit_CO), 1e-9),
        "crc_gate": slack_crc / max(abs(crc_thr_any), 1e-9),
    }
    if epsilon_nox is not None:
        norm_slacks["epsilon_gate"] = slack_eps / max(abs(epsilon_nox), 1e-9)

    dominant_constraint = min(norm_slacks, key=norm_slacks.get) if norm_slacks else "other"
    near_bind_tol = 0.01  # <=1% normalized slack is treated as near-binding
    diagnostics = {
        "optimizer_status": "feasible",
        "dominant_constraint": dominant_constraint,
        "primary_blocker": None,
        "n_feasible_candidates": int(feasible.sum()),
        "candidate_count": int(n_cand),
        "active_nox_constraint": bool(norm_slacks.get("nox_ub", np.inf) <= near_bind_tol),
        "active_tit_constraint": bool(norm_slacks.get("tit_ub", np.inf) <= near_bind_tol),
        "active_co_constraint": bool(norm_slacks.get("co_ub", np.inf) <= near_bind_tol),
        "active_crc_constraint": bool(norm_slacks.get("crc_gate", np.inf) <= near_bind_tol),
        "active_eps_constraint": bool(norm_slacks.get("epsilon_gate", np.inf) <= near_bind_tol),
    }

    return best_tey, best_obj, best_preds, best_ubs, diagnostics

# 10. Receding-horizon EMPC simulation on test subset (model-based plant)
def _simulate_empc_core(
    alpha: float,
    X_test_subset: pd.DataFrame,
    limit_NOX: float,
    limit_TIT: float,
    limit_CO: float,
    point_models: dict,
    conformal_quantiles: dict,
    df_full: pd.DataFrame,
    m_calib: np.ndarray | None,
    residuals_map: dict,
    breach_clfs,
    crc_thresholds: dict,
    tey_candidates: np.ndarray,
    feature_pos: dict,
    alpha_idx_map: dict,
    epsilon_nox: float | None,
    include_real_fields: bool,
    attach_epsilon: bool,
    return_dataframe: bool,
    init_row: pd.Series | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Shared simulator for EMPC+CRC with optional ε-constraint.
    Controls output shape via return_dataframe and field toggles.
    init_row overrides m_calib for initial lagged state; rng controls residual draws.
    """
    simulation_results = []

    if init_row is None:
        if m_calib is None or not np.any(m_calib):
            raise ValueError("Calibration mask (m_calib) is empty and no init_row provided.")
        last_calib_global_idx = df_full[m_calib].index[-1]
        last_calib_data = df_full.loc[last_calib_global_idx]
    else:
        last_calib_data = init_row

    lagged_values = _initialize_lagged_state(last_calib_data, df_full)

    rng_use = rng if rng is not None else RNG

    for i in range(len(X_test_subset)):
        original_df_index = X_test_subset.index[i]
        current_full_row = df_full.loc[original_df_index]

        current_hour_idx = int(current_full_row["hour_idx"])
        current_hour_data_for_opt = current_full_row[["AT", "AP", "AH", "AFDP", "hour_idx"]]

        real_TIT = float(current_full_row["TIT"])
        real_NOX = float(current_full_row["NOX"])
        real_CO  = float(current_full_row["CO"])
        real_TEY = float(current_full_row["TEY"])
        real_GTEP = float(current_full_row["GTEP"])
        real_TAT  = float(current_full_row["TAT"])
        real_CDP  = float(current_full_row["CDP"])

        optimal_TEY, econ_val, pred_outputs, ub_outputs, opt_diag = _run_one_hour_optimization_core(
            current_hour_data=current_hour_data_for_opt,
            lagged_values=lagged_values,
            alpha=alpha,
            limit_NOX=limit_NOX,
            limit_TIT=limit_TIT,
            limit_CO=limit_CO,
            point_models=point_models,
            conformal_quantiles=conformal_quantiles,
            breach_clfs=breach_clfs,
            crc_thresholds=crc_thresholds,
            tey_candidates=tey_candidates,
            feature_pos=feature_pos,
            alpha_idx_map=alpha_idx_map,
            epsilon_nox=epsilon_nox,
        )

        if optimal_TEY is None or np.isnan(optimal_TEY):
            chosen_TEY = 0.0
            econ_val = 0.0
            sim_TIT_true = np.nan
            sim_NOX_true = np.nan
            sim_CO_true  = np.nan
            # Infeasible hour => realised breach/violation is undefined, not "False".
            sim_tit_breach = np.nan
            sim_nox_breach = np.nan
            sim_co_breach  = np.nan
            conf_tit_violation = np.nan
            conf_nox_violation = np.nan
            conf_co_violation  = np.nan
        else:
            chosen_TEY = float(optimal_TEY)

            res_TIT = rng_use.choice(residuals_map["TIT"])
            res_NOX = rng_use.choice(residuals_map["NOX"])
            res_CO  = rng_use.choice(residuals_map["CO"])

            sim_TIT_true = float(pred_outputs["TIT"] + res_TIT)
            sim_NOX_true = float(pred_outputs["NOX"] + res_NOX)
            sim_CO_true  = float(pred_outputs["CO"]  + res_CO)

            sim_tit_breach = sim_TIT_true > limit_TIT
            sim_nox_breach = sim_NOX_true > limit_NOX
            sim_co_breach  = sim_CO_true  > limit_CO

            conf_tit_violation = sim_TIT_true > ub_outputs["TIT"]
            conf_nox_violation = sim_NOX_true > ub_outputs["NOX"]
            conf_co_violation  = sim_CO_true  > ub_outputs["CO"]

        result = {
            "hour_idx": current_hour_idx,
            "alpha_val": alpha,
            "optimal_TEY": chosen_TEY,
            "economic_objective": float(econ_val),
            "sim_TIT": sim_TIT_true,
            "sim_NOX": sim_NOX_true,
            "sim_CO": sim_CO_true,
            "sim_tit_breach": sim_tit_breach,
            "sim_nox_breach": sim_nox_breach,
            "sim_co_breach": sim_co_breach,
            "conf_tit_violation": conf_tit_violation,
            "conf_nox_violation": conf_nox_violation,
            "conf_co_violation":  conf_co_violation,
            "optimizer_status": str(opt_diag.get("optimizer_status", "unknown")),
            "dominant_constraint": str(opt_diag.get("dominant_constraint", "other")),
            "primary_blocker": opt_diag.get("primary_blocker"),
            "n_feasible_candidates": int(opt_diag.get("n_feasible_candidates", 0)),
            "candidate_count": int(opt_diag.get("candidate_count", 0)),
            "active_nox_constraint": bool(opt_diag.get("active_nox_constraint", False)),
            "active_tit_constraint": bool(opt_diag.get("active_tit_constraint", False)),
            "active_co_constraint": bool(opt_diag.get("active_co_constraint", False)),
            "active_crc_constraint": bool(opt_diag.get("active_crc_constraint", False)),
            "active_eps_constraint": bool(opt_diag.get("active_eps_constraint", False)),
        }

        if attach_epsilon:
            result["epsilon_nox"] = float(epsilon_nox)

        if include_real_fields:
            result.update(
                {
                    "real_TEY": real_TEY,
                    "real_TIT": real_TIT,
                    "real_NOX": real_NOX,
                    "real_CO": real_CO,
                    "real_GTEP": real_GTEP,
                    "real_TAT": real_TAT,
                    "real_CDP": real_CDP,
                }
            )

        simulation_results.append(result)

        if optimal_TEY is not None and not np.isnan(optimal_TEY):
            next_state_values = {
                "TEY": chosen_TEY,
                "TIT": sim_TIT_true,
                "NOX": sim_NOX_true,
                "CO": sim_CO_true,
                "GTEP": real_GTEP,
                "TAT": real_TAT,
                "CDP": real_CDP,
            }
        else:
            next_state_values = {
                "TEY": chosen_TEY,
                "TIT": real_TIT,
                "NOX": real_NOX,
                "CO": real_CO,
                "GTEP": real_GTEP,
                "TAT": real_TAT,
                "CDP": real_CDP,
            }
        lagged_values = _advance_lagged_state(lagged_values, next_state_values)

    if return_dataframe:
        return pd.DataFrame(simulation_results)
    return simulation_results


def simulate_empc_test_set(
    alpha: float,
    X_test_subset: pd.DataFrame,
    limit_NOX: float,
    limit_TIT: float,
    limit_CO: float,
    point_models: dict,
    conformal_quantiles: dict,
    df_full: pd.DataFrame,
    m_calib: np.ndarray,
    residuals_map: dict,
    breach_clfs,
    crc_thresholds: dict,
    tey_candidates: np.ndarray,
    feature_pos: dict,
    alpha_idx_map: dict,
) -> list:
    return _simulate_empc_core(
        alpha=alpha,
        X_test_subset=X_test_subset,
        limit_NOX=limit_NOX,
        limit_TIT=limit_TIT,
        limit_CO=limit_CO,
        point_models=point_models,
        conformal_quantiles=conformal_quantiles,
        df_full=df_full,
        m_calib=m_calib,
        residuals_map=residuals_map,
        breach_clfs=breach_clfs,
        crc_thresholds=crc_thresholds,
        tey_candidates=tey_candidates,
        feature_pos=feature_pos,
        alpha_idx_map=alpha_idx_map,
        epsilon_nox=None,
        include_real_fields=True,
        attach_epsilon=False,
        return_dataframe=False,
    )


# 10.5 CRC threshold calibration (full CRC via independent trajectories)
print("\nCalibrating CRC thresholds (full CRC, independent trajectories)...")
crc_rng = np.random.default_rng(CRC_SEED)
crc_horizon = len(calib_df) if CRC_TRAJECTORY_HOURS is None else int(CRC_TRAJECTORY_HOURS)
crc_trajectories, crc_init_rows = build_iid_trajectories(
    df_source=calib_df,
    init_source=train_df,
    n_traj=CRC_NUM_TRAJECTORIES,
    horizon=crc_horizon,
    rng=crc_rng,
)

crc_thresholds_any = {}
for alpha in ALPHAS:
    crc_thresholds_any[alpha] = calibrate_crc_threshold_any_full(
        alpha=alpha,
        trajectories=crc_trajectories,
        init_rows=crc_init_rows,
        lambda_grid=lambda_grid_any,
        limit_NOX=limit_NOX,
        limit_TIT=limit_TIT,
        limit_CO=limit_CO,
        point_models=point_models,
        conformal_quantiles=conformal_quantiles,
        residuals_map=residuals_map,
        breach_clfs=breach_clfs,
        crc_tey_candidates=TEY_CANDIDATES,
        feature_pos=FEATURE_POS,
        alpha_idx_map=ALPHA_TO_IDX,
        loss_mode=CRC_LOSS_MODE,
        monotone=True,
        base_seed=CRC_SEED,
    )
    print(f"  α={alpha:.3f} -> λ_any={crc_thresholds_any[alpha]:.4f}")

crc_thresholds = {"ANY": crc_thresholds_any}


def _summarize_frontier_alpha_metrics(alpha_val: float, sim_df: pd.DataFrame) -> dict:
    total_mwh = sim_df["optimal_TEY"].fillna(0.0).sum()
    total_economic_objective = sim_df["economic_objective"].fillna(0.0).sum()

    feasible_mask = sim_df["sim_TIT"].notna()
    sim_df_feasible = sim_df.loc[feasible_mask]
    breach_cols = ["sim_nox_breach", "sim_tit_breach", "sim_co_breach"]

    if len(sim_df_feasible) > 0:
        sim_nox_breach_rate = float(sim_df_feasible["sim_nox_breach"].eq(True).mean())
        sim_tit_breach_rate = float(sim_df_feasible["sim_tit_breach"].eq(True).mean())
        sim_co_breach_rate = float(sim_df_feasible["sim_co_breach"].eq(True).mean())
        sim_any_breach_rate = float(
            sim_df_feasible[breach_cols].eq(True).any(axis=1).mean()
        )
        sim_any_breach_indicator = float(
            sim_df_feasible[breach_cols].eq(True).any(axis=1).any()
        )
    else:
        sim_nox_breach_rate = np.nan
        sim_tit_breach_rate = np.nan
        sim_co_breach_rate = np.nan
        sim_any_breach_rate = np.nan
        sim_any_breach_indicator = np.nan

    if {"conf_nox_violation", "conf_tit_violation", "conf_co_violation"}.issubset(sim_df.columns):
        conf_cols = ["conf_nox_violation", "conf_tit_violation", "conf_co_violation"]
        if len(sim_df_feasible) > 0:
            conf_nox_violation_rate = float(sim_df_feasible["conf_nox_violation"].eq(True).mean())
            conf_tit_violation_rate = float(sim_df_feasible["conf_tit_violation"].eq(True).mean())
            conf_co_violation_rate = float(sim_df_feasible["conf_co_violation"].eq(True).mean())
            conf_any_violation_rate = float(
                sim_df_feasible[conf_cols].eq(True).any(axis=1).mean()
            )
        else:
            conf_nox_violation_rate = np.nan
            conf_tit_violation_rate = np.nan
            conf_co_violation_rate = np.nan
            conf_any_violation_rate = np.nan
    else:
        conf_nox_violation_rate = np.nan
        conf_tit_violation_rate = np.nan
        conf_co_violation_rate = np.nan
        conf_any_violation_rate = np.nan

    infeasible_hours = int(sim_df["sim_TIT"].isna().sum())
    feasible_hours = int(len(sim_df) - infeasible_hours)
    feasible_fraction = float(feasible_hours / len(sim_df)) if len(sim_df) else np.nan
    infeasible_fraction = float(infeasible_hours / len(sim_df)) if len(sim_df) else np.nan
    mean_tey = float(sim_df["optimal_TEY"].mean())
    std_tey = float(sim_df["optimal_TEY"].std())

    return {
        "alpha": alpha_val,
        "total_mwh": total_mwh,
        "total_economic_objective": total_economic_objective,
        "sim_nox_breach_rate": sim_nox_breach_rate,
        "sim_tit_breach_rate": sim_tit_breach_rate,
        "sim_co_breach_rate": sim_co_breach_rate,
        "sim_any_breach_rate": sim_any_breach_rate,
        "sim_any_breach_indicator": sim_any_breach_indicator,
        "conf_nox_violation_rate": conf_nox_violation_rate,
        "conf_tit_violation_rate": conf_tit_violation_rate,
        "conf_co_violation_rate": conf_co_violation_rate,
        "conf_any_violation_rate": conf_any_violation_rate,
        "mean_tey": mean_tey,
        "std_tey": std_tey,
        "feasible_hours": feasible_hours,
        "infeasible_hours": infeasible_hours,
        "feasible_fraction": feasible_fraction,
        "infeasible_fraction": infeasible_fraction,
    }
# 11. Risk→Cash frontier on test set (model-based EMPC+CRC)

risk_to_cash_frontier_data = []
frontier_simulation_traces = {}

# For final experiments, default is full test set.
# Set simulation_hours to an integer if you want a quick debug run.
simulation_hours = sim_set  # e.g. 50 for fast tests

if simulation_hours is None:
    X_test_subset = X_test
else:
    X_test_subset = X_test.head(simulation_hours)

for alpha_val in ALPHAS:
    print(f"\n Running EMPC+CRC simulation for alpha={alpha_val} ")
    hourly_simulation_results = simulate_empc_test_set(
        alpha=alpha_val,
        X_test_subset=X_test_subset,
        limit_NOX=limit_NOX,
        limit_TIT=limit_TIT,
        limit_CO=limit_CO,
        point_models=point_models,
        conformal_quantiles=conformal_quantiles,
        df_full=df,
        m_calib=m_calib,
        residuals_map=residuals_map,
        breach_clfs=breach_clfs,
        crc_thresholds=crc_thresholds,
        tey_candidates=TEY_CANDIDATES,
        feature_pos=FEATURE_POS,
        alpha_idx_map=ALPHA_TO_IDX,
    )
    sim_df = pd.DataFrame(hourly_simulation_results)
    frontier_simulation_traces[alpha_val] = sim_df.copy()
    alpha_results = _summarize_frontier_alpha_metrics(alpha_val, sim_df)
    risk_to_cash_frontier_data.append(alpha_results)

risk_to_cash_df = pd.DataFrame(risk_to_cash_frontier_data)

print("\n Risk-to-Cash Frontier (model-based EMPC+CRC simulation) ")
print(risk_to_cash_df.to_string(index=False))
print(" %s seconds " % (time.time() - start_time))

"""RESULTS FROM EMPC AND CRC RUNNING"""

# 12. Comprehensive EMPC+CRC results: Risk→Cash + safety diagnostics

# Print a concise summary table
summary_cols = [
    "alpha",
    "total_mwh",
    "total_economic_objective",
    "feasible_hours",
    "infeasible_hours",
    "feasible_fraction",
    "infeasible_fraction",
    "mean_tey",
    "std_tey",
    "sim_any_breach_rate",
    "conf_any_violation_rate",
    "sim_any_breach_indicator",
]
print("\n EMPC+CRC Risk→Cash Summary (per α) ")
print(risk_to_cash_df[summary_cols].to_string(index=False))

# Plotting is handled externally in plot_risk_to_cash_results.py.

"""PARETO SET"""

# 13. ε-constraint Pareto frontier at fixed α (EMPC+CRC with per-hour NOX cap)

#  construct a Pareto frontier between emissions and money at a fixed α by
# sweeping an additional per-hour NOX cap ε. For each ε we re-run EMPC+CRC,
# now rejecting TEY candidates whose predicted NOX exceeds ε, and record
# (total emissions, total economic objective).


def simulate_empc_with_epsilon(
    alpha: float,
    epsilon_nox: float,
    X_test_subset: pd.DataFrame,
    limit_NOX: float,
    limit_TIT: float,
    limit_CO: float,
    point_models: dict,
    conformal_quantiles: dict,
    df_full: pd.DataFrame,
    m_calib: np.ndarray,
    residuals_map: dict,
    breach_clfs,
    crc_thresholds: dict,
    tey_candidates: np.ndarray,
    feature_pos: dict,
    alpha_idx_map: dict,
) -> pd.DataFrame:
    """
    Closed-loop EMPC+CRC simulation on the test subset with an additional
    per-hour NOX cap ε (ε-constraint). Uses the same model-based plant
    (surrogate + empirical residuals) as before.
    """
    return _simulate_empc_core(
        alpha=alpha,
        X_test_subset=X_test_subset,
        limit_NOX=limit_NOX,
        limit_TIT=limit_TIT,
        limit_CO=limit_CO,
        point_models=point_models,
        conformal_quantiles=conformal_quantiles,
        df_full=df_full,
        m_calib=m_calib,
        residuals_map=residuals_map,
        breach_clfs=breach_clfs,
        crc_thresholds=crc_thresholds,
        tey_candidates=tey_candidates,
        feature_pos=feature_pos,
        alpha_idx_map=alpha_idx_map,
        epsilon_nox=epsilon_nox,
        include_real_fields=False,
        attach_epsilon=True,
        return_dataframe=True,
    )


# Helper: energy-weighted mean with safe handling of empty/zero weights
def safe_energy_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    weights = weights.fillna(0.0)
    values = values.fillna(0.0)
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return float("nan")
    return float(np.dot(values, weights) / w_sum)


def _summarize_epsilon_row(alpha_val: float, eps_val: float, sim_eps_df: pd.DataFrame) -> dict:
    total_econ = float(sim_eps_df["economic_objective"].fillna(0.0).sum())
    total_nox = float(sim_eps_df["sim_NOX"].fillna(0.0).sum())
    total_co = float(sim_eps_df["sim_CO"].fillna(0.0).sum())
    mean_nox_all = float(sim_eps_df["sim_NOX"].fillna(0.0).mean())
    mean_nox_feasible = float(sim_eps_df["sim_NOX"].dropna().mean()) if sim_eps_df["sim_NOX"].notna().any() else np.nan

    tey_series = sim_eps_df["optimal_TEY"].fillna(0.0)
    energy_weighted_nox = safe_energy_weighted_mean(sim_eps_df["sim_NOX"], tey_series)
    energy_weighted_co = safe_energy_weighted_mean(sim_eps_df["sim_CO"], tey_series)

    infeasible_hours = int(sim_eps_df["sim_TIT"].isna().sum())
    feasible_hours = int(len(sim_eps_df) - infeasible_hours)
    feasible_fraction = float(feasible_hours / len(sim_eps_df)) if len(sim_eps_df) else np.nan
    infeasible_fraction = float(infeasible_hours / len(sim_eps_df)) if len(sim_eps_df) else np.nan

    feasible_mask = sim_eps_df["sim_TIT"].notna()
    sim_eps_feasible = sim_eps_df.loc[feasible_mask]
    if len(sim_eps_feasible) > 0:
        any_breach_rate = float(
            sim_eps_feasible[["sim_nox_breach", "sim_tit_breach", "sim_co_breach"]]
            .eq(True)
            .any(axis=1)
            .mean()
        )
        any_conf_violation_rate = float(
            sim_eps_feasible[["conf_nox_violation", "conf_tit_violation", "conf_co_violation"]]
            .eq(True)
            .any(axis=1)
            .mean()
        )
    else:
        any_breach_rate = np.nan
        any_conf_violation_rate = np.nan

    return {
        "alpha": alpha_val,
        "epsilon_nox_cap": float(eps_val),
        "total_economic_objective": total_econ,
        "total_sim_nox": total_nox,
        "total_sim_co": total_co,
        "mean_sim_nox": mean_nox_all,
        "mean_sim_nox_feasible": mean_nox_feasible,
        "energy_weighted_nox": energy_weighted_nox,
        "energy_weighted_co": energy_weighted_co,
        "total_tey": float(tey_series.sum()),
        "sim_any_breach_rate": any_breach_rate,
        "conf_any_violation_rate": any_conf_violation_rate,
        "feasible_hours": feasible_hours,
        "infeasible_hours": infeasible_hours,
        "feasible_fraction": feasible_fraction,
        "infeasible_fraction": infeasible_fraction,
    }
# 14. Build ε-constraint Pareto frontier across all α values

# Use full test set (or shorten for debugging via epsilon_simulation_hours)
epsilon_simulation_hours = sim_set  # set e.g. 200 for quick runs

if epsilon_simulation_hours is None:
    X_test_eps = X_test
else:
    X_test_eps = X_test.head(epsilon_simulation_hours)

# Define an ε grid for NOX based on TRAIN distribution
# from relatively clean to relatively dirty but within proxy limits
eps_min = float(train_df["NOX"].quantile(EPSILON_EFFECTIVE_Q_MIN))
eps_max = float(train_df["NOX"].quantile(EPSILON_EFFECTIVE_Q_MAX))
num_eps = EPSILON_GRID_SIZE  # ε grid resolution (smaller = faster)
epsilon_grid = np.linspace(eps_min, eps_max, num=num_eps)
print(
    "Pareto epsilon sweep controls:",
    f"eps_min={eps_min:.3f} (q={EPSILON_EFFECTIVE_Q_MIN:.3f}),",
    f"eps_max={eps_max:.3f} (q={EPSILON_EFFECTIVE_Q_MAX:.3f}),",
    f"gate_mode={EPSILON_EFFECTIVE_GATE_MODE},",
    f"ub_weight={EPSILON_EFFECTIVE_UB_WEIGHT:.2f},",
    f"pareto_nox_weight={EPSILON_OBJECTIVE_NOX_WEIGHT:.2f}",
)

pareto_eps_results = []
pareto_by_alpha = {}
simulation_traces = {}  # store per-(alpha, epsilon) trajectories for derived plots

for alpha_val in ALPHAS:
    print(f" ε-constraint EMPC+CRC sweeps at α={alpha_val}")

    alpha_rows = []

    for eps_val in epsilon_grid:
        print(f"\n Running EMPC+CRC for α={alpha_val}, ε_NOX={eps_val:.2f} ")
        sim_eps_df = simulate_empc_with_epsilon(
            alpha=alpha_val,
            epsilon_nox=float(eps_val),
            X_test_subset=X_test_eps,
            limit_NOX=limit_NOX,
            limit_TIT=limit_TIT,
            limit_CO=limit_CO,
            point_models=point_models,
            conformal_quantiles=conformal_quantiles,
            df_full=df,
            m_calib=m_calib,
            residuals_map=residuals_map,
            breach_clfs=breach_clfs,
            crc_thresholds=crc_thresholds,
            tey_candidates=TEY_CANDIDATES,
            feature_pos=FEATURE_POS,
            alpha_idx_map=ALPHA_TO_IDX,
        )

        # Persist full trajectory for later time-series/knee/MAC analysis
        simulation_traces[(alpha_val, float(eps_val))] = sim_eps_df.copy()

        row = _summarize_epsilon_row(alpha_val, float(eps_val), sim_eps_df)
        pareto_eps_results.append(row)
        alpha_rows.append(row)

    df_alpha = pd.DataFrame(alpha_rows).sort_values("epsilon_nox_cap")
    pareto_by_alpha[alpha_val] = df_alpha

    print(f"\n ε-constraint Pareto frontier data for α={alpha_val} ")
    print(df_alpha.to_string(index=False))

pareto_eps_df = pd.DataFrame(pareto_eps_results).sort_values(["alpha", "epsilon_nox_cap"])

print("\n Combined ε-constraint Pareto frontier data (all α values) ")
print(pareto_eps_df.to_string(index=False))

# Plotting is handled externally in plot_risk_to_cash_results.py.

# summary view for reporting
summary_cols = [
    "alpha",
    "epsilon_nox_cap",
    "mean_sim_nox",
    "mean_sim_nox_feasible",
    "total_economic_objective",
    "sim_any_breach_rate",
    "conf_any_violation_rate",
]

print("\n ε-constraint summary (all α, sorted by α and ε) ")
print(pareto_eps_df[summary_cols].to_string(index=False))

print("\n ε-constraint summary (feasible hours only, sorted by α and ε) ")
print(
    pareto_eps_df[
        ["alpha", "epsilon_nox_cap", "mean_sim_nox_feasible", "total_economic_objective", "sim_any_breach_rate", "conf_any_violation_rate"]
    ].to_string(index=False)
)

# Feasibility recap per α (closed-loop EMPC+CRC)
print("\n EMPC+CRC Infeasible Hours (per α) ")
print(
    risk_to_cash_df[
        ["alpha", "feasible_hours", "infeasible_hours", "feasible_fraction", "infeasible_fraction"]
    ].to_string(index=False)
)

# Save outputs for the external plotting script.
os.makedirs(RISK_TO_CASH_RESULTS_DIR, exist_ok=True)
risk_to_cash_df.to_csv(RISK_TO_CASH_FRONTIER_CSV_PATH, index=False)
pareto_eps_df.to_csv(PARETO_EPSILON_FRONTIER_CSV_PATH, index=False)

results_bundle = {
    "risk_to_cash_df": risk_to_cash_df,
    "frontier_simulation_traces": frontier_simulation_traces,
    "pareto_eps_df": pareto_eps_df,
    "pareto_by_alpha": pareto_by_alpha,
    "simulation_traces": simulation_traces,
    "alphas": list(ALPHAS),
    "plot_alpha": float(PLOT_ALPHA),
    "x_emission_col": "energy_weighted_nox",
    "y_money_col": "total_economic_objective",
    "epsilon_control_summary": {
        "epsilon_sensitivity_dial": float(EPSILON_SENSITIVITY_DIAL),
        "epsilon_base_q_min": float(EPSILON_BASE_Q_MIN),
        "epsilon_base_q_max": float(EPSILON_BASE_Q_MAX),
        "epsilon_effective_q_min": float(EPSILON_EFFECTIVE_Q_MIN),
        "epsilon_effective_q_max": float(EPSILON_EFFECTIVE_Q_MAX),
        "epsilon_base_gate_mode": str(EPSILON_BASE_GATE_MODE),
        "epsilon_effective_gate_mode": str(EPSILON_EFFECTIVE_GATE_MODE),
        "epsilon_base_ub_weight": float(EPSILON_BASE_UB_WEIGHT),
        "epsilon_effective_ub_weight": float(EPSILON_EFFECTIVE_UB_WEIGHT),
        "pareto_nox_objective_weight": float(EPSILON_OBJECTIVE_NOX_WEIGHT),
    },
    "risk_summary_columns": [
        "alpha",
        "total_mwh",
        "total_economic_objective",
        "feasible_hours",
        "infeasible_hours",
        "feasible_fraction",
        "infeasible_fraction",
        "mean_tey",
        "std_tey",
        "sim_any_breach_rate",
        "conf_any_violation_rate",
        "sim_any_breach_indicator",
    ],
    "epsilon_summary_columns": [
        "alpha",
        "epsilon_nox_cap",
        "mean_sim_nox",
        "mean_sim_nox_feasible",
        "total_economic_objective",
        "sim_any_breach_rate",
        "conf_any_violation_rate",
        "feasible_fraction",
        "infeasible_fraction",
    ],
}
joblib.dump(results_bundle, RISK_TO_CASH_RESULTS_BUNDLE_PATH)

print("\nSaved risk-to-cash result tables:")
print(f" - {RISK_TO_CASH_FRONTIER_CSV_PATH}")
print(f" - {PARETO_EPSILON_FRONTIER_CSV_PATH}")
print("Saved plotting input bundle:")
print(f" - {RISK_TO_CASH_RESULTS_BUNDLE_PATH}")
print("\nTo render all plots, run:")
print("  python \"plot_risk_to_cash_results.py\"")

total_runtime = time.perf_counter() - script_start_time
print(f"\n>>> Total runtime: {total_runtime:.2f} seconds")
