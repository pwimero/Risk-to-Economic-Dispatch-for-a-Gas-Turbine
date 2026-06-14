import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from math import ceil

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from shared_config import (
    ALPHAS,
    ARTIFACTS_DIR,
    BASE_EXOG,
    CONFORMAL_BOUNDS_PATH,
    CRC_MANIFEST_PATH,
    CRC_MODEL_DIR,
    DATA_WITH_TIME_PATH,
    PREPARED_FEATURES_PATH,
    RAW_DATA_PATH,
    SPLIT_DIR,
    SURROGATE_EVAL_REPORT_PATH,
    SURROGATE_MANIFEST_PATH,
    SURROGATE_MODEL_DIR,
    STATE_HISTORY_COLS,
    STATE_HISTORY_LAGS,
    TARGETS,
    UPPER_TAU,
)

# Dependency bootstrap (optional)
INSTALL_DEPS = False
LGBM_PINNED_VERSION = "lightgbm==4.6.0"

# Tuning knobs (edit these independently from RiskToCashFrontier.py)
LGBM_SEED = 42
LGBM_POINT_KW = dict(
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=96,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_split_gain=1e-3,
    min_data_in_leaf=120,
    lambda_l1=0.0,
    lambda_l2=0.0,
    max_bin=255,
    random_state=LGBM_SEED,
    n_jobs=-1,
    verbose=-1,
)
LGBM_QUANTILE_KW = dict(LGBM_POINT_KW)
LGBM_CRC_CLASSIFIER_KW = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_split_gain=0.0,
    min_data_in_leaf=60,
    lambda_l1=0.0,
    lambda_l2=0.0,
    max_bin=255,
    random_state=LGBM_SEED,
    n_jobs=-1,
    verbose=-1,
)
LGBM_VAL_FRAC = 0.1
LGBM_VAL_MIN = 200
LGBM_EARLY_STOPPING_ROUNDS = 100
LGBM_LOG_EVAL_PERIOD = 50
# Extra target-only lags to improve CO persistence modeling without
# changing TIT/NOX feature definitions.
CO_EXTRA_TARGET_LAGS = [6, 12, 24]
JOURNAL_RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "journal_results")


def maybe_install_dependencies() -> None:
    if not INSTALL_DEPS:
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", LGBM_PINNED_VERSION, "joblib"],
        check=True,
    )


def ensure_time_indexed_dataset(raw_path: str, data_with_time_path: str) -> None:
    df_raw = pd.read_csv(raw_path)
    df_raw["hour_idx"] = range(len(df_raw))
    os.makedirs(os.path.dirname(data_with_time_path), exist_ok=True)
    df_raw.to_csv(data_with_time_path, index=False)


def create_time_splits(data_path: str, split_dir: str):
    out_dir = split_dir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    if "hour_idx" not in df.columns:
        raise ValueError("Expected an 'hour_idx' column.")

    df = df.sort_values("hour_idx", kind="mergesort").reset_index(drop=True)
    dups = int(df["hour_idx"].duplicated().sum())
    if dups > 0:
        raise ValueError(f"Duplicate hour_idx values found: {dups}")

    diffs = df["hour_idx"].diff().dropna()
    if not (diffs == 1).all():
        print("Warning: hour_idx has gaps. Proceeding in chronological order.")

    n = len(df)
    n_train = int(n * 0.70)
    n_calib = int(n * 0.15)

    mask_train = np.zeros(n, dtype=bool)
    mask_train[:n_train] = True
    mask_calib = np.zeros(n, dtype=bool)
    mask_calib[n_train:n_train + n_calib] = True
    mask_test = np.zeros(n, dtype=bool)
    mask_test[n_train + n_calib:] = True

    np.save(os.path.join(out_dir, "train_mask.npy"), mask_train)
    np.save(os.path.join(out_dir, "calibration_mask.npy"), mask_calib)
    np.save(os.path.join(out_dir, "test_mask.npy"), mask_test)

    split_labels = np.where(mask_train, "train", np.where(mask_calib, "calib", "test"))
    pd.DataFrame({"hour_idx": df["hour_idx"], "split": split_labels}).to_csv(
        os.path.join(out_dir, "split_labels.csv"),
        index=False,
    )

    manifest = {
        "data_path": data_path,
        "split_dir": out_dir,
        "n_total": int(n),
        "n_train": int(mask_train.sum()),
        "n_calib": int(mask_calib.sum()),
        "n_test": int(mask_test.sum()),
        "train_range_hour_idx": [int(df["hour_idx"].iloc[0]), int(df["hour_idx"].iloc[n_train - 1])],
        "calib_range_hour_idx": [int(df["hour_idx"].iloc[n_train]), int(df["hour_idx"].iloc[n_train + n_calib - 1])],
        "test_range_hour_idx": [int(df["hour_idx"].iloc[n_train + n_calib]), int(df["hour_idx"].iloc[-1])],
        "notes": [
            "Contiguous time blocks to avoid leakage (no shuffling).",
            "Keep calibration untouched for conformal residual quantiles.",
        ],
    }
    with open(os.path.join(out_dir, "time_split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Saved splits:", out_dir)
    print(f"Counts -> train: {mask_train.sum()}, calib: {mask_calib.sum()}, test: {mask_test.sum()}")
    return df, mask_train, mask_calib, mask_test


def load_time_splits(data_path: str, split_dir: str):
    df = pd.read_csv(data_path).sort_values("hour_idx", kind="mergesort").reset_index(drop=True)
    mask_train = np.load(os.path.join(split_dir, "train_mask.npy"))
    mask_calib = np.load(os.path.join(split_dir, "calibration_mask.npy"))
    mask_test = np.load(os.path.join(split_dir, "test_mask.npy"))
    if not (len(df) == len(mask_train) == len(mask_calib) == len(mask_test)):
        raise ValueError("Mask lengths must match dataframe length.")
    return df, mask_train, mask_calib, mask_test


def add_time_features(frame: pd.DataFrame | pd.Series):
    if isinstance(frame, pd.Series):
        temp_df = frame.to_frame().T
    else:
        temp_df = frame.copy()
    if "hour_idx" not in temp_df.columns:
        raise ValueError("Input must contain 'hour_idx' column.")
    hod = temp_df["hour_idx"] % 24
    temp_df = temp_df.assign(
        hour_of_day=hod,
        hod_sin=np.sin(2 * np.pi * hod / 24.0),
        hod_cos=np.cos(2 * np.pi * hod / 24.0),
    )
    if isinstance(frame, pd.Series):
        return temp_df.iloc[0]
    return temp_df


def add_lags(frame: pd.DataFrame, cols, lags=(1,)):
    out = frame.copy()
    for c in cols:
        for lag in lags:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    return out


def _unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_target_feature_map(base_exog: list[str]) -> dict[str, list[str]]:
    time_feats = ["hour_of_day", "hod_sin", "hod_cos"]
    baseline_lag_cols = [f"{c}_lag1" for c in ("TEY", "TIT", "GTEP", "TAT", "CDP")]
    baseline_features = _unique_keep_order([*base_exog, *time_feats, *baseline_lag_cols])

    def _target_history_cols(target_name: str) -> list[str]:
        history_cols = []
        lag_orders = sorted({int(l) for l in STATE_HISTORY_LAGS})
        state_cols = _unique_keep_order([*STATE_HISTORY_COLS, target_name])
        for col in state_cols:
            for lag in lag_orders:
                # Keep lag1 for the target itself, and lag2+ for all configured state columns.
                if lag > 1 or col == target_name:
                    history_cols.append(f"{col}_lag{lag}")
        return history_cols

    feature_map = {t: list(baseline_features) for t in TARGETS}
    if "NOX" in TARGETS:
        feature_map["NOX"] = _unique_keep_order([*baseline_features, *_target_history_cols("NOX")])
    if "CO" in TARGETS:
        co_feats = _unique_keep_order([*baseline_features, *_target_history_cols("CO")])
        co_extra_lags = sorted({int(l) for l in CO_EXTRA_TARGET_LAGS if int(l) >= 1})
        for lag in co_extra_lags:
            co_feats.append(f"CO_lag{lag}")
        feature_map["CO"] = _unique_keep_order(co_feats)
    return feature_map


def _required_lag_feature_names(feature_map: dict[str, list[str]]) -> list[str]:
    lag_cols = []
    for cols in feature_map.values():
        for col in cols:
            if "_lag" in col:
                lag_cols.append(col)
    return _unique_keep_order(lag_cols)


def _required_lag_source_cols(required_lag_cols: list[str]) -> list[str]:
    src_cols = []
    for col in required_lag_cols:
        if "_lag" not in col:
            continue
        src_cols.append(col.rsplit("_lag", 1)[0])
    return _unique_keep_order(src_cols)


def _model_input_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    feat_names = getattr(model, "feature_name_", None)
    if feat_names is None or len(feat_names) == 0:
        return X
    missing = [f for f in feat_names if f not in X.columns]
    if missing:
        raise KeyError(f"Missing model features in input frame: {missing[:10]}")
    return X[list(feat_names)]


def prepare_features_and_masks(
    data_path: str,
    split_dir: str,
    base_exog: list[str] | None = None,
    lagged_cols: tuple[str, ...] | None = None,
    lags: tuple[int, ...] | None = None,
):
    base_exog = base_exog or ["AT", "AP", "AH", "AFDP", "TEY"]
    lagged_cols = lagged_cols or tuple(STATE_HISTORY_COLS)
    lag_orders = tuple(sorted({int(l) for l in (lags or tuple(STATE_HISTORY_LAGS))}))
    if not lag_orders:
        raise ValueError("At least one lag order is required.")

    df, m_train, m_calib, m_test = load_time_splits(data_path, split_dir)
    target_feature_map = _build_target_feature_map(base_exog)
    required_lag_cols = _required_lag_feature_names(target_feature_map)
    required_lag_orders = sorted(
        {
            int(col.rsplit("_lag", 1)[1])
            for col in required_lag_cols
            if "_lag" in col and col.rsplit("_lag", 1)[1].isdigit()
        }
    )
    lag_orders = tuple(sorted(set(lag_orders).union(required_lag_orders)))
    required_lag_sources = _required_lag_source_cols(required_lag_cols)
    lag_cols_to_build = _unique_keep_order([*list(lagged_cols), *required_lag_sources])
    feat_df = add_time_features(df)
    feat_df = add_lags(feat_df, cols=lag_cols_to_build, lags=lag_orders)

    valid_idx = feat_df[required_lag_cols].notna().all(axis=1)
    feat_df = feat_df.loc[valid_idx].reset_index(drop=True)
    df = df.loc[valid_idx].reset_index(drop=True)
    m_train = m_train[valid_idx.values]
    m_calib = m_calib[valid_idx.values]
    m_test = m_test[valid_idx.values]

    features = _unique_keep_order([f for cols in target_feature_map.values() for f in cols])
    X = feat_df[features]
    return df, X, m_train, m_calib, m_test, features, target_feature_map


def build_and_save_prepared_features(
    data_path: str,
    split_dir: str,
    out_path: str,
    base_exog: list[str] | None = None,
    lagged_cols: tuple[str, ...] | None = None,
    lags: tuple[int, ...] | None = None,
):
    df, X, m_train, m_calib, m_test, features, target_feature_map = prepare_features_and_masks(
        data_path=data_path,
        split_dir=split_dir,
        base_exog=base_exog,
        lagged_cols=lagged_cols,
        lags=lags,
    )
    artifact = {
        "df": df,
        "X": X,
        "mask_train": np.asarray(m_train, dtype=bool),
        "mask_calib": np.asarray(m_calib, dtype=bool),
        "mask_test": np.asarray(m_test, dtype=bool),
        "features": list(features),
        "feature_pos": {f: i for i, f in enumerate(features)},
        "target_feature_map": {k: list(v) for k, v in target_feature_map.items()},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artifact, out_path)
    print("Saved prepared feature artifact:", out_path)
    return artifact


def load_prepared_feature_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Prepared feature artifact missing. Run main() to generate it.\n"
            f"Missing file: {path}"
        )
    artifact = joblib.load(path)
    required = ("df", "X", "mask_train", "mask_calib", "mask_test", "features", "feature_pos")
    missing = [k for k in required if k not in artifact]
    if missing:
        raise KeyError(f"Prepared feature artifact missing keys: {missing}")
    if "target_feature_map" not in artifact:
        # Backward compatibility for older artifacts.
        default_features = list(artifact["features"])
        artifact["target_feature_map"] = {t: list(default_features) for t in TARGETS}
    return artifact


def conformal_upper_quantile(residuals: np.ndarray, alpha: float) -> float:
    res = np.sort(np.asarray(residuals))
    n = res.size
    k = int(ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(res[k - 1])


def inner_time_split(n: int, val_frac: float):
    n_val = max(LGBM_VAL_MIN, int(n * val_frac))
    n_val = min(n_val, n - 1)
    n_tr = n - n_val
    idx = np.arange(n)
    return idx[:n_tr], idx[n_tr:]


def fit_lgb_point(X_train, y_train, X_val, y_val):
    model = LGBMRegressor(objective="regression", **LGBM_POINT_KW)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[
            early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, first_metric_only=True),
            log_evaluation(period=LGBM_LOG_EVAL_PERIOD),
        ],
    )
    return model


def fit_lgb_quantile(X_train, y_train, X_val, y_val, tau: float):
    model = LGBMRegressor(objective="quantile", alpha=tau, **LGBM_QUANTILE_KW)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[
            early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, first_metric_only=True),
            log_evaluation(period=LGBM_LOG_EVAL_PERIOD),
        ],
    )
    return model


def train_surrogates() -> dict:
    os.makedirs(SURROGATE_MODEL_DIR, exist_ok=True)

    prepared = load_prepared_feature_artifact(PREPARED_FEATURES_PATH)
    df = prepared["df"]
    X = prepared["X"]
    m_train = np.asarray(prepared["mask_train"], dtype=bool)
    m_calib = np.asarray(prepared["mask_calib"], dtype=bool)
    m_test = np.asarray(prepared["mask_test"], dtype=bool)
    features = list(prepared["features"])
    target_feature_map = {k: list(v) for k, v in prepared.get("target_feature_map", {}).items()}
    if not target_feature_map:
        target_feature_map = {t: list(features) for t in TARGETS}
    for t in TARGETS:
        target_feature_map.setdefault(t, list(features))

    if "TIT" in features or "TAT" in features or "GTEP" in features or "CDP" in features:
        raise ValueError("Leakage detected: current process outputs were included in features.")
    if "TIT_lag1" not in target_feature_map.get("TIT", []):
        raise ValueError("Expected TIT_lag1 in TIT feature set.")
    if "NOX" in TARGETS and "NOX_lag1" not in target_feature_map.get("NOX", []):
        raise ValueError("Expected NOX_lag1 in NOX feature set.")
    if "CO" in TARGETS and "CO_lag1" not in target_feature_map.get("CO", []):
        raise ValueError("Expected CO_lag1 in CO feature set.")

    y_map = {t: df[t] for t in TARGETS}
    X_train, X_calib = X[m_train], X[m_calib]
    y_train_map = {t: y_map[t][m_train] for t in TARGETS}
    y_calib_map = {t: y_map[t][m_calib] for t in TARGETS}

    artifacts = {
        "features": features,
        "target_feature_map": target_feature_map,
        "targets": TARGETS,
        "upper_tau": UPPER_TAU,
        "seed": LGBM_SEED,
        "models": {},
    }

    for target in TARGETS:
        target_features = list(target_feature_map[target])
        Xtr_all = X_train[target_features].reset_index(drop=True)
        ytr_all = y_train_map[target].reset_index(drop=True)
        tr_idx, val_idx = inner_time_split(len(Xtr_all), LGBM_VAL_FRAC)
        Xtr, ytr = Xtr_all.iloc[tr_idx], ytr_all.iloc[tr_idx]
        Xva, yva = Xtr_all.iloc[val_idx], ytr_all.iloc[val_idx]

        m_point = fit_lgb_point(Xtr, ytr, Xva, yva)
        m_qu = fit_lgb_quantile(Xtr, ytr, Xva, yva, tau=UPPER_TAU)

        yhat_calib = m_point.predict(X_calib[target_features])
        mae_calib = mean_absolute_error(y_calib_map[target], yhat_calib)
        r2_calib = r2_score(y_calib_map[target], yhat_calib)
        resid_calib = y_calib_map[target].to_numpy() - yhat_calib

        tgt_dir = os.path.join(SURROGATE_MODEL_DIR, target)
        os.makedirs(tgt_dir, exist_ok=True)
        point_path = os.path.join(tgt_dir, "point_regressor.joblib")
        quantile_path = os.path.join(tgt_dir, f"upper_quantile_regressor_tau{UPPER_TAU:.2f}.joblib")
        residual_path = os.path.join(tgt_dir, "calibration_residuals.npy")

        joblib.dump(m_point, point_path)
        joblib.dump(m_qu, quantile_path)
        np.save(residual_path, resid_calib)

        artifacts["models"][target] = {
            "point_model": point_path,
            "upper_quantile_model": quantile_path,
            "calib_residuals": residual_path,
            "calib_mae": float(mae_calib),
            "calib_r2": float(r2_calib),
            "feature_columns": target_features,
            "best_iteration_point": int(getattr(m_point, "best_iteration_", m_point.n_estimators)),
            "best_iteration_quantile": int(getattr(m_qu, "best_iteration_", m_qu.n_estimators)),
            "n_train": int(m_train.sum()),
            "n_calib": int(m_calib.sum()),
            "n_test": int(m_test.sum()),
        }

        print(
            f"[{target}] calib MAE={mae_calib:.4f} R2={r2_calib:.4f} "
            f"best_iters={artifacts['models'][target]['best_iteration_point']}/"
            f"{artifacts['models'][target]['best_iteration_quantile']}"
        )

    artifacts["created_utc"] = datetime.now(timezone.utc).isoformat()
    artifacts["data_path"] = DATA_WITH_TIME_PATH
    artifacts["split_dir"] = SPLIT_DIR
    artifacts["prepared_features_path"] = PREPARED_FEATURES_PATH
    artifacts["surrogate_model_dir"] = SURROGATE_MODEL_DIR
    with open(SURROGATE_MANIFEST_PATH, "w") as f:
        json.dump(artifacts, f, indent=2)

    print("\nSaved surrogate artifacts:", SURROGATE_MODEL_DIR)
    print("Saved surrogate manifest:", SURROGATE_MANIFEST_PATH)
    print(
        json.dumps(
            {t: {"MAE": artifacts["models"][t]["calib_mae"], "R2": artifacts["models"][t]["calib_r2"]} for t in TARGETS},
            indent=2,
        )
    )
    return artifacts


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def _detect_overfit(train_mae: float, calib_mae: float, train_r2: float, calib_r2: float) -> bool:
    mae_gap = (calib_mae - train_mae) / max(calib_mae, 1e-9)
    r2_gap = train_r2 - calib_r2
    return (mae_gap > 0.20) or (r2_gap > 0.10)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    abs_err = np.abs(err)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(100.0 * _safe_mape(y_true, y_pred)),
        "smape_pct": float(100.0 * _safe_smape(y_true, y_pred)),
        "bias": float(np.mean(err)),
        "error_std": float(np.std(err)),
        "max_abs_error": float(np.max(abs_err)),
        "abs_error_p50": float(np.percentile(abs_err, 50)),
        "abs_error_p90": float(np.percentile(abs_err, 90)),
        "abs_error_p95": float(np.percentile(abs_err, 95)),
        "pearson_r": _safe_corr(y_true, y_pred),
    }


def _json_num(x: float):
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return None
    return x


def _json_metrics(metrics: dict) -> dict:
    return {k: _json_num(v) for k, v in metrics.items()}


def _safe_load_json(path: str, default: dict | None = None) -> dict:
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r") as f:
        return json.load(f)


def _lag_orders_from_feature_names(feature_names: list[str]) -> list[int]:
    lags = []
    for name in feature_names:
        if "_lag" not in name:
            continue
        suffix = name.rsplit("_lag", 1)[1]
        if suffix.isdigit():
            lags.append(int(suffix))
    return sorted(set(lags))


def _numeric_summary_rows(frame: pd.DataFrame, variable_cols: list[str], split_name: str | None = None) -> list[dict]:
    rows = []
    for col in variable_cols:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if s.empty:
            continue
        q = s.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        row = {
            "variable": str(col),
            "n_obs": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
            "min": float(s.min()),
            "p01": float(q.loc[0.01]),
            "p05": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]),
            "p50": float(q.loc[0.50]),
            "p75": float(q.loc[0.75]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
            "max": float(s.max()),
        }
        if split_name is not None:
            row["split"] = split_name
        rows.append(row)
    return rows


def _save_table_outputs(df: pd.DataFrame, out_dir: str, table_name: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{table_name}.csv")
    txt_path = os.path.join(out_dir, f"{table_name}.txt")
    tex_path = os.path.join(out_dir, f"{table_name}.tex")

    df.to_csv(csv_path, index=False)
    with open(txt_path, "w") as f:
        if df.empty:
            f.write("(empty)\n")
        else:
            f.write(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
            f.write("\n")

    latex_written = False
    try:
        latex_str = df.to_latex(
            index=False,
            na_rep="NaN",
            float_format=lambda x: f"{x:.6g}",
            escape=True,
        )
        with open(tex_path, "w") as f:
            f.write(latex_str)
        latex_written = True
    except Exception:
        latex_written = False

    out = {"csv": csv_path, "txt": txt_path}
    if latex_written:
        out["tex"] = tex_path
    return out


def _feature_importance_rows(model, target: str, model_kind: str, top_k: int) -> list[dict]:
    feature_names = list(getattr(model, "feature_name_", []) or [])
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(feature_names) == 0:
        return []
    importances = np.asarray(importances, dtype=float)
    if importances.shape[0] != len(feature_names):
        return []

    total_importance = float(np.sum(importances))
    denom = total_importance if total_importance > 0 else 1.0
    ranking = np.argsort(importances)[::-1]
    rows = []
    for rank, idx in enumerate(ranking[:top_k], start=1):
        imp = float(importances[idx])
        rows.append(
            {
                "target": str(target),
                "model_kind": str(model_kind),
                "rank": int(rank),
                "feature": str(feature_names[int(idx)]),
                "importance": imp,
                "importance_share_pct": float(100.0 * imp / denom),
            }
        )
    return rows


def export_journal_results_suite() -> dict:
    prepared = load_prepared_feature_artifact(PREPARED_FEATURES_PATH)
    df = prepared["df"].copy()
    X = prepared["X"]
    m_train = np.asarray(prepared["mask_train"], dtype=bool)
    m_calib = np.asarray(prepared["mask_calib"], dtype=bool)
    m_test = np.asarray(prepared["mask_test"], dtype=bool)
    features = list(prepared["features"])
    target_feature_map = {k: list(v) for k, v in prepared.get("target_feature_map", {}).items()}
    if not target_feature_map:
        target_feature_map = {t: list(features) for t in TARGETS}

    split_labels = np.full(len(df), "unknown", dtype=object)
    split_labels[m_train] = "train"
    split_labels[m_calib] = "calib"
    split_labels[m_test] = "test"
    df["split"] = split_labels

    training_manifest = _safe_load_json(SURROGATE_MANIFEST_PATH, default={})
    eval_report = _safe_load_json(SURROGATE_EVAL_REPORT_PATH, default={})
    conformal_bounds = _safe_load_json(CONFORMAL_BOUNDS_PATH, default={})
    time_split_manifest = _safe_load_json(os.path.join(SPLIT_DIR, "time_split_manifest.json"), default={})
    crc_manifest = _safe_load_json(CRC_MANIFEST_PATH, default={})

    generated_files = {}

    # Table 1: split overview and sample horizon.
    split_rows = []
    total_n = int(len(df))
    for split in ["train", "calib", "test"]:
        part = df.loc[df["split"] == split]
        if part.empty:
            continue
        split_rows.append(
            {
                "split": split,
                "n_rows": int(part.shape[0]),
                "pct_rows": float(100.0 * part.shape[0] / max(total_n, 1)),
                "hour_idx_start": int(part["hour_idx"].min()),
                "hour_idx_end": int(part["hour_idx"].max()),
                "duration_days_approx": float(part.shape[0] / 24.0),
            }
        )
    split_rows.append(
        {
            "split": "all",
            "n_rows": total_n,
            "pct_rows": 100.0,
            "hour_idx_start": int(df["hour_idx"].min()),
            "hour_idx_end": int(df["hour_idx"].max()),
            "duration_days_approx": float(total_n / 24.0),
        }
    )
    table_01 = pd.DataFrame(split_rows)
    generated_files["table_01_dataset_split_overview"] = _save_table_outputs(
        table_01, JOURNAL_RESULTS_DIR, "table_01_dataset_split_overview"
    )

    # Table 2: overall variable-level descriptive statistics.
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"split", "hour_idx"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    table_02 = pd.DataFrame(_numeric_summary_rows(df, numeric_cols))
    generated_files["table_02_dataset_variable_summary_overall"] = _save_table_outputs(
        table_02, JOURNAL_RESULTS_DIR, "table_02_dataset_variable_summary_overall"
    )

    # Table 3: variable-level descriptive statistics by split.
    split_stat_rows = []
    for split in ["train", "calib", "test"]:
        part = df.loc[df["split"] == split]
        if part.empty:
            continue
        split_stat_rows.extend(_numeric_summary_rows(part, numeric_cols, split_name=split))
    table_03 = pd.DataFrame(split_stat_rows)
    if not table_03.empty:
        table_03 = table_03[
            [
                "split",
                "variable",
                "n_obs",
                "mean",
                "std",
                "min",
                "p01",
                "p05",
                "p25",
                "p50",
                "p75",
                "p95",
                "p99",
                "max",
            ]
        ]
    generated_files["table_03_dataset_variable_summary_by_split"] = _save_table_outputs(
        table_03, JOURNAL_RESULTS_DIR, "table_03_dataset_variable_summary_by_split"
    )

    # Table 4: surrogate feature-set composition by target.
    feature_rows = []
    time_feats = {"hour_of_day", "hod_sin", "hod_cos"}
    for target in TARGETS:
        cols = list(target_feature_map.get(target, features))
        lag_cols = [c for c in cols if "_lag" in c]
        lag_orders = _lag_orders_from_feature_names(cols)
        lag_sources = sorted(
            {
                c.rsplit("_lag", 1)[0]
                for c in lag_cols
                if "_lag" in c and c.rsplit("_lag", 1)[1].isdigit()
            }
        )
        feature_rows.append(
            {
                "target": target,
                "n_features": int(len(cols)),
                "n_base_exogenous_features": int(sum(c in BASE_EXOG for c in cols)),
                "n_time_features": int(sum(c in time_feats for c in cols)),
                "n_lag_features": int(len(lag_cols)),
                "lag_orders": ",".join(str(v) for v in lag_orders),
                "lag_sources": ",".join(lag_sources),
                "feature_columns": ";".join(cols),
            }
        )
    table_04 = pd.DataFrame(feature_rows)
    generated_files["table_04_feature_set_by_target"] = _save_table_outputs(
        table_04, JOURNAL_RESULTS_DIR, "table_04_feature_set_by_target"
    )

    # Table 5: modeling hyperparameters and trainer controls.
    hyper_rows = []
    for param, value in sorted(LGBM_POINT_KW.items()):
        hyper_rows.append({"model_component": "point_regressor", "parameter": param, "value": value})
    for param, value in sorted(LGBM_QUANTILE_KW.items()):
        hyper_rows.append({"model_component": "upper_quantile_regressor", "parameter": param, "value": value})
    for param, value in sorted(LGBM_CRC_CLASSIFIER_KW.items()):
        hyper_rows.append({"model_component": "crc_breach_classifier", "parameter": param, "value": value})
    trainer_controls = {
        "upper_tau": UPPER_TAU,
        "alpha_grid": ",".join(str(a) for a in ALPHAS),
        "seed": LGBM_SEED,
        "val_frac": LGBM_VAL_FRAC,
        "val_min_rows": LGBM_VAL_MIN,
        "early_stopping_rounds": LGBM_EARLY_STOPPING_ROUNDS,
        "log_eval_period": LGBM_LOG_EVAL_PERIOD,
        "state_history_cols": ",".join(STATE_HISTORY_COLS),
        "state_history_lags": ",".join(str(l) for l in STATE_HISTORY_LAGS),
        "co_extra_target_lags": ",".join(str(l) for l in CO_EXTRA_TARGET_LAGS),
        "n_total_features_unioned": int(len(features)),
    }
    for param, value in trainer_controls.items():
        hyper_rows.append({"model_component": "training_control", "parameter": param, "value": value})
    table_05 = pd.DataFrame(hyper_rows)
    generated_files["table_05_surrogate_hyperparameters"] = _save_table_outputs(
        table_05, JOURNAL_RESULTS_DIR, "table_05_surrogate_hyperparameters"
    )

    # Table 6: trained model registry and calibration stats.
    model_rows = []
    for target in TARGETS:
        info = training_manifest.get("models", {}).get(target, {})
        cols = list(info.get("feature_columns", target_feature_map.get(target, [])))
        model_rows.append(
            {
                "target": target,
                "n_features": int(len(cols)),
                "n_train": int(info.get("n_train", int(m_train.sum()))),
                "n_calib": int(info.get("n_calib", int(m_calib.sum()))),
                "n_test": int(info.get("n_test", int(m_test.sum()))),
                "calib_mae": info.get("calib_mae"),
                "calib_r2": info.get("calib_r2"),
                "best_iteration_point": info.get("best_iteration_point"),
                "best_iteration_quantile": info.get("best_iteration_quantile"),
                "point_model_path": info.get("point_model"),
                "upper_quantile_model_path": info.get("upper_quantile_model"),
            }
        )
    table_06 = pd.DataFrame(model_rows)
    generated_files["table_06_trained_surrogate_registry"] = _save_table_outputs(
        table_06, JOURNAL_RESULTS_DIR, "table_06_trained_surrogate_registry"
    )

    # Table 7: point-model performance by split.
    point_rows = []
    quantile_rows = []
    for target in TARGETS:
        target_report = eval_report.get("targets", {}).get(target, {})
        point_metrics = target_report.get("point_metrics", {})
        quantile_metrics = target_report.get("quantile_metrics", {})
        for split, metrics in point_metrics.items():
            row = {
                "target": target,
                "split": split,
                "overfit": bool(target_report.get("overfit", False)),
            }
            row.update(metrics)
            point_rows.append(row)
        for split, metrics in quantile_metrics.items():
            row = {"target": target, "split": split}
            row.update(metrics)
            quantile_rows.append(row)
    table_07 = pd.DataFrame(point_rows)
    generated_files["table_07_point_model_metrics_by_split"] = _save_table_outputs(
        table_07, JOURNAL_RESULTS_DIR, "table_07_point_model_metrics_by_split"
    )

    # Table 8: quantile-model diagnostics by split.
    table_08 = pd.DataFrame(quantile_rows)
    generated_files["table_08_quantile_model_metrics_by_split"] = _save_table_outputs(
        table_08, JOURNAL_RESULTS_DIR, "table_08_quantile_model_metrics_by_split"
    )

    # Table 9: conformal coverage profile across alpha values.
    conformal_rows = []
    conformal_from_eval = eval_report.get("conformal_coverage", {})
    if conformal_from_eval:
        for target, rows in conformal_from_eval.items():
            for row in rows:
                conformal_rows.append({"target": target, **row})
    elif conformal_bounds:
        alpha_grid = list(conformal_bounds.get("alpha_grid", ALPHAS))
        for target in TARGETS:
            target_data = conformal_bounds.get("targets", {}).get(target, {})
            point_cov = target_data.get("point_test_coverage", [])
            cqr_cov = target_data.get("cqr_test_coverage", [])
            if cqr_cov is None:
                cqr_cov = [None] * len(alpha_grid)
            for alpha, p_cov, c_cov in zip(alpha_grid, point_cov, cqr_cov):
                nominal = 1.0 - float(alpha)
                conformal_rows.append(
                    {
                        "target": target,
                        "alpha": float(alpha),
                        "nominal_coverage": nominal,
                        "point_coverage": p_cov,
                        "point_gap": None if p_cov is None else float(p_cov - nominal),
                        "cqr_coverage": c_cov,
                        "cqr_gap": None if c_cov is None else float(c_cov - nominal),
                    }
                )
    table_09 = pd.DataFrame(conformal_rows)
    generated_files["table_09_conformal_coverage_by_alpha"] = _save_table_outputs(
        table_09, JOURNAL_RESULTS_DIR, "table_09_conformal_coverage_by_alpha"
    )

    # Table 10: CRC breach-classifier summary.
    crc_rows = []
    for target in TARGETS:
        c = crc_manifest.get("classifiers", {}).get(target, {})
        crc_rows.append(
            {
                "target": target,
                "proxy_limit": crc_manifest.get("proxy_limits", {}).get(target),
                "train_prevalence": c.get("train_prevalence"),
                "calib_score_mean": c.get("calib_score_mean"),
                "calib_score_std": c.get("calib_score_std"),
                "n_features": int(len(c.get("feature_columns", []))),
                "model_path": c.get("model_path"),
            }
        )
    table_10 = pd.DataFrame(crc_rows)
    generated_files["table_10_crc_classifier_summary"] = _save_table_outputs(
        table_10, JOURNAL_RESULTS_DIR, "table_10_crc_classifier_summary"
    )

    # Table 11: top feature importances (point + quantile models).
    fi_rows = []
    for target in TARGETS:
        target_manifest = training_manifest.get("models", {}).get(target, {})
        point_path = target_manifest.get("point_model") or os.path.join(
            SURROGATE_MODEL_DIR, target, "point_regressor.joblib"
        )
        if os.path.exists(point_path):
            point_model = joblib.load(point_path)
            fi_rows.extend(_feature_importance_rows(point_model, target, "point", top_k=20))

        quantile_path = target_manifest.get("upper_quantile_model") or os.path.join(
            SURROGATE_MODEL_DIR, target, f"upper_quantile_regressor_tau{UPPER_TAU:.2f}.joblib"
        )
        if os.path.exists(quantile_path):
            q_model = joblib.load(quantile_path)
            fi_rows.extend(_feature_importance_rows(q_model, target, "quantile", top_k=20))
    table_11 = pd.DataFrame(fi_rows)
    generated_files["table_11_feature_importance_top20"] = _save_table_outputs(
        table_11, JOURNAL_RESULTS_DIR, "table_11_feature_importance_top20"
    )

    # Table 12: target correlation matrix (all samples, leakage-safe descriptive only).
    corr_cols = [c for c in TARGETS if c in df.columns]
    table_12 = df[corr_cols].corr().reset_index().rename(columns={"index": "target"})
    generated_files["table_12_target_correlation_matrix"] = _save_table_outputs(
        table_12, JOURNAL_RESULTS_DIR, "table_12_target_correlation_matrix"
    )

    summary_path = os.path.join(JOURNAL_RESULTS_DIR, "journal_results_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Surrogate Journal Results Suite\n\n")
        f.write(f"- Created UTC: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- Data source: `{DATA_WITH_TIME_PATH}`\n")
        f.write(f"- Prepared features artifact: `{PREPARED_FEATURES_PATH}`\n")
        if time_split_manifest:
            f.write(
                f"- Time split counts: train={time_split_manifest.get('n_train')}, "
                f"calib={time_split_manifest.get('n_calib')}, test={time_split_manifest.get('n_test')}\n"
            )
        f.write(f"- Total rows after lag pruning: {len(df)}\n")
        f.write(f"- Unified feature count: {X.shape[1]}\n\n")
        f.write("## Exported Tables\n\n")
        for table_name, paths in generated_files.items():
            rel_csv = os.path.relpath(paths["csv"], JOURNAL_RESULTS_DIR)
            rel_txt = os.path.relpath(paths["txt"], JOURNAL_RESULTS_DIR)
            rel_tex = os.path.relpath(paths["tex"], JOURNAL_RESULTS_DIR) if "tex" in paths else None
            f.write(f"- `{table_name}`: [`csv`]({rel_csv}), [`txt`]({rel_txt})")
            if rel_tex is not None:
                f.write(f", [`tex`]({rel_tex})")
            f.write("\n")
        f.write("\n")
        f.write("## Quick Look: Point-Model Test Metrics\n\n")
        if not table_07.empty:
            preview = table_07.loc[table_07["split"] == "test"].copy()
            if preview.empty:
                preview = table_07.copy()
            f.write("```\n")
            f.write(preview.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
            f.write("\n```\n")
        else:
            f.write("_No point-model metrics available._\n")

    suite_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "journal_results_dir": JOURNAL_RESULTS_DIR,
        "source_artifacts": {
            "prepared_features_path": PREPARED_FEATURES_PATH,
            "surrogate_manifest_path": SURROGATE_MANIFEST_PATH,
            "surrogate_eval_report_path": SURROGATE_EVAL_REPORT_PATH,
            "conformal_bounds_path": CONFORMAL_BOUNDS_PATH,
            "time_split_manifest_path": os.path.join(SPLIT_DIR, "time_split_manifest.json"),
            "crc_manifest_path": CRC_MANIFEST_PATH,
        },
        "tables": generated_files,
        "summary_report": summary_path,
    }
    suite_manifest_path = os.path.join(JOURNAL_RESULTS_DIR, "journal_results_manifest.json")
    with open(suite_manifest_path, "w") as f:
        json.dump(suite_manifest, f, indent=2)

    print("\nSaved journal-grade surrogate tables:", JOURNAL_RESULTS_DIR)
    print("Saved journal suite manifest:", suite_manifest_path)
    print("Saved journal summary:", summary_path)
    return suite_manifest


def report_surrogate_performance() -> dict:
    prepared = load_prepared_feature_artifact(PREPARED_FEATURES_PATH)
    df = prepared["df"]
    X = prepared["X"]
    m_train = np.asarray(prepared["mask_train"], dtype=bool)
    m_calib = np.asarray(prepared["mask_calib"], dtype=bool)
    m_test = np.asarray(prepared["mask_test"], dtype=bool)

    y_map = {t: df[t] for t in TARGETS}
    X_train = X[m_train]
    X_calib = X[m_calib]
    X_test = X[m_test]

    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upper_tau": float(UPPER_TAU),
        "targets": {},
        "overfit_targets": [],
    }
    print("\nLightGBM surrogate performance (saved artifacts)")
    print("target | split | MAE | RMSE | R2 | sMAPE% | Bias | p90| pinball | q_cov | q_gap | overfit")

    for target in TARGETS:
        point_path = os.path.join(SURROGATE_MODEL_DIR, target, "point_regressor.joblib")
        point_model = joblib.load(point_path)

        y_tr = y_map[target][m_train].to_numpy()
        y_cb = y_map[target][m_calib].to_numpy()
        y_te = y_map[target][m_test].to_numpy()

        yhat_tr = point_model.predict(_model_input_frame(point_model, X_train))
        yhat_cb = point_model.predict(_model_input_frame(point_model, X_calib))
        yhat_te = point_model.predict(_model_input_frame(point_model, X_test))

        metrics_tr = _regression_metrics(y_tr, yhat_tr)
        metrics_cb = _regression_metrics(y_cb, yhat_cb)
        metrics_te = _regression_metrics(y_te, yhat_te)

        overfit = _detect_overfit(metrics_tr["mae"], metrics_cb["mae"], metrics_tr["r2"], metrics_cb["r2"])

        q_path = os.path.join(SURROGATE_MODEL_DIR, target, f"upper_quantile_regressor_tau{UPPER_TAU:.2f}.joblib")
        quantile_metrics = {
            "train": {"pinball": float("nan"), "empirical_coverage": float("nan"), "coverage_gap_to_tau": float("nan")},
            "calib": {"pinball": float("nan"), "empirical_coverage": float("nan"), "coverage_gap_to_tau": float("nan")},
            "test": {"pinball": float("nan"), "empirical_coverage": float("nan"), "coverage_gap_to_tau": float("nan")},
        }
        if os.path.exists(q_path):
            q_model = joblib.load(q_path)
            q_tr = q_model.predict(_model_input_frame(q_model, X_train))
            q_cb = q_model.predict(_model_input_frame(q_model, X_calib))
            q_te = q_model.predict(_model_input_frame(q_model, X_test))
            quantile_metrics["train"] = {
                "pinball": _pinball_loss(y_tr, q_tr, UPPER_TAU),
                "empirical_coverage": float(np.mean(y_tr <= q_tr)),
                "coverage_gap_to_tau": float(np.mean(y_tr <= q_tr) - UPPER_TAU),
            }
            quantile_metrics["calib"] = {
                "pinball": _pinball_loss(y_cb, q_cb, UPPER_TAU),
                "empirical_coverage": float(np.mean(y_cb <= q_cb)),
                "coverage_gap_to_tau": float(np.mean(y_cb <= q_cb) - UPPER_TAU),
            }
            quantile_metrics["test"] = {
                "pinball": _pinball_loss(y_te, q_te, UPPER_TAU),
                "empirical_coverage": float(np.mean(y_te <= q_te)),
                "coverage_gap_to_tau": float(np.mean(y_te <= q_te) - UPPER_TAU),
            }

        best_iter = int(
            getattr(
                point_model,
                "best_iteration_",
                getattr(point_model, "n_estimators", getattr(point_model, "n_iter_", -1)),
            )
        )
        overfit_flag = "YES" if overfit else "no"
        if overfit:
            results["overfit_targets"].append(target)

        def _print_row(split_name: str, metrics: dict, q_metrics: dict):
            print(
                f"{target:>5} | {split_name:<5} | {metrics['mae']:7.4f} | {metrics['rmse']:7.4f} | "
                f"{metrics['r2']:5.3f} | {metrics['smape_pct']:7.3f} | {metrics['bias']:6.3f} | "
                f"{metrics['abs_error_p90']:5.3f} | {q_metrics['pinball']:7.4f} | "
                f"{q_metrics['empirical_coverage']:5.3f} | {q_metrics['coverage_gap_to_tau']:6.3f} | {overfit_flag}"
            )

        _print_row("train", metrics_tr, quantile_metrics["train"])
        _print_row("calib", metrics_cb, quantile_metrics["calib"])
        _print_row("test", metrics_te, quantile_metrics["test"])

        results["targets"][target] = {
            "point_metrics": {
                "train": _json_metrics(metrics_tr),
                "calib": _json_metrics(metrics_cb),
                "test": _json_metrics(metrics_te),
            },
            "quantile_metrics": {
                "train": {k: _json_num(v) for k, v in quantile_metrics["train"].items()},
                "calib": {k: _json_num(v) for k, v in quantile_metrics["calib"].items()},
                "test": {k: _json_num(v) for k, v in quantile_metrics["test"].items()},
            },
            "best_iteration": int(best_iter),
            "overfit": bool(overfit),
            "generalization_gap": {
                "mae_gap_frac": _json_num((metrics_cb["mae"] - metrics_tr["mae"]) / max(metrics_cb["mae"], 1e-9)),
                "r2_gap": _json_num(metrics_tr["r2"] - metrics_cb["r2"]),
            },
        }

    if results["overfit_targets"]:
        print("\nOverfitting detected (heuristic):", ", ".join(results["overfit_targets"]))
    else:
        print("\nOverfitting detected (heuristic): none")

    if os.path.exists(CONFORMAL_BOUNDS_PATH):
        with open(CONFORMAL_BOUNDS_PATH, "r") as f:
            conformal_results = json.load(f)
        alpha_grid = conformal_results.get("alpha_grid", ALPHAS)
        conformal_summary = {}
        print("\nConformal calibration summary (saved bounds)")
        print("target | alpha | nominal | point_cov | point_gap | cqr_cov | cqr_gap")
        for target in TARGETS:
            cov_point = conformal_results["targets"].get(target, {}).get("point_test_coverage", None)
            cov_cqr = conformal_results["targets"].get(target, {}).get("cqr_test_coverage", None)
            if cov_point is None:
                continue
            cov_cqr = cov_cqr or [None] * len(alpha_grid)
            rows = []
            for alpha, cp, cq in zip(alpha_grid, cov_point, cov_cqr):
                nominal = 1.0 - alpha
                point_gap = float(cp - nominal)
                cqr_gap = None if cq is None else float(cq - nominal)
                rows.append(
                    {
                        "alpha": float(alpha),
                        "nominal_coverage": float(nominal),
                        "point_coverage": _json_num(cp),
                        "point_gap": _json_num(point_gap),
                        "cqr_coverage": _json_num(cq) if cq is not None else None,
                        "cqr_gap": _json_num(cqr_gap) if cqr_gap is not None else None,
                    }
                )
                cq_str = "  nan" if cq is None else f"{cq:7.3f}"
                cq_gap_str = "  nan" if cqr_gap is None else f"{cqr_gap:7.3f}"
                print(
                    f"{target:>5} | {alpha:5.3f} | {nominal:7.3f} | {cp:9.3f} | {point_gap:9.3f} | "
                    f"{cq_str} | {cq_gap_str}"
                )
            conformal_summary[target] = rows
        results["conformal_coverage"] = conformal_summary

    with open(SURROGATE_EVAL_REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved surrogate evaluation report:", SURROGATE_EVAL_REPORT_PATH)

    return results


def build_conformal_bounds() -> dict:
    out_path = CONFORMAL_BOUNDS_PATH
    prepared = load_prepared_feature_artifact(PREPARED_FEATURES_PATH)
    df = prepared["df"]
    X = prepared["X"]
    m_calib = np.asarray(prepared["mask_calib"], dtype=bool)
    m_test = np.asarray(prepared["mask_test"], dtype=bool)
    X_calib = X[m_calib]
    X_test = X[m_test]
    y_map = {t: df[t] for t in TARGETS}
    y_calib = {t: y_map[t][m_calib].to_numpy() for t in TARGETS}
    y_test = {t: y_map[t][m_test].to_numpy() for t in TARGETS}

    results = {"alpha_grid": ALPHAS, "targets": {}}

    for target in TARGETS:
        tgt_dir = os.path.join(SURROGATE_MODEL_DIR, target)
        point_model = joblib.load(os.path.join(tgt_dir, "point_regressor.joblib"))
        q_path = os.path.join(tgt_dir, f"upper_quantile_regressor_tau{UPPER_TAU:.2f}.joblib")
        has_cqr = os.path.exists(q_path)
        q_model = joblib.load(q_path) if has_cqr else None

        residuals = np.load(os.path.join(tgt_dir, "calibration_residuals.npy"))
        if residuals.shape[0] != y_calib[target].shape[0]:
            raise ValueError(f"Residual length mismatch for {target}.")

        yhat_test_point = point_model.predict(_model_input_frame(point_model, X_test))
        ub_point = []
        cov_point = []
        for alpha in ALPHAS:
            q = conformal_upper_quantile(residuals, alpha)
            ub = yhat_test_point + q
            coverage = float(np.mean(y_test[target] <= ub))
            ub_point.append(q)
            cov_point.append(coverage)

        ub_cqr = []
        cov_cqr = []
        if has_cqr:
            qhat_calib = q_model.predict(_model_input_frame(q_model, X_calib))
            score_calib = y_calib[target] - qhat_calib
            qhat_test = q_model.predict(_model_input_frame(q_model, X_test))
            for alpha in ALPHAS:
                q_score = conformal_upper_quantile(score_calib, alpha)
                ub = qhat_test + q_score
                coverage = float(np.mean(y_test[target] <= ub))
                ub_cqr.append(q_score)
                cov_cqr.append(coverage)
        else:
            ub_cqr = None
            cov_cqr = None

        results["targets"][target] = {
            "point_residual_quantiles": ub_point,
            "point_test_coverage": cov_point,
            "cqr_score_quantiles": ub_cqr,
            "cqr_test_coverage": cov_cqr,
        }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved conformal bounds:", out_path)
    print("\nTest coverage summary")
    for target in TARGETS:
        print(f"\n[{target}]")
        print(" alpha | 1-alpha | cov_point | cov_cqr")
        cov_cqr = results["targets"][target]["cqr_test_coverage"] or [None] * len(ALPHAS)
        for alpha, cov_point, cov_q in zip(ALPHAS, results["targets"][target]["point_test_coverage"], cov_cqr):
            one_minus_alpha = 1.0 - alpha
            cov_q_str = "   -" if cov_q is None else f"{cov_q:7.3f}"
            print(f" {alpha:5.3f} | {one_minus_alpha:7.3f} | {cov_point:8.3f} | {cov_q_str}")

    return results


def train_crc_breach_classifiers() -> dict:
    os.makedirs(CRC_MODEL_DIR, exist_ok=True)

    prepared = load_prepared_feature_artifact(PREPARED_FEATURES_PATH)
    df = prepared["df"]
    X = prepared["X"]
    m_train = np.asarray(prepared["mask_train"], dtype=bool)
    m_calib = np.asarray(prepared["mask_calib"], dtype=bool)
    target_feature_map = {k: list(v) for k, v in prepared.get("target_feature_map", {}).items()}

    train_df = df[m_train]
    X_train = X[m_train]
    X_calib = X[m_calib]

    proxy_limits = {
        "NOX": float(train_df["NOX"].quantile(0.95)),
        "TIT": float(train_df["TIT"].quantile(0.95)),
        "CO": float(train_df["CO"].quantile(0.95)),
    }

    breach_labels = {
        "NOX": (train_df["NOX"] > proxy_limits["NOX"]).astype(int).to_numpy(),
        "TIT": (train_df["TIT"] > proxy_limits["TIT"]).astype(int).to_numpy(),
        "CO": (train_df["CO"] > proxy_limits["CO"]).astype(int).to_numpy(),
    }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_features_path": PREPARED_FEATURES_PATH,
        "proxy_limits": proxy_limits,
        "classifier_kwargs": LGBM_CRC_CLASSIFIER_KW,
        "classifiers": {},
    }

    print("\nTraining CRC breach classifiers...")
    for target in TARGETS:
        target_features = target_feature_map.get(target, list(X.columns))
        X_train_target = X_train[target_features]
        X_calib_target = X_calib[target_features]
        clf = LGBMClassifier(**LGBM_CRC_CLASSIFIER_KW)
        clf.fit(X_train_target, breach_labels[target])

        model_path = os.path.join(CRC_MODEL_DIR, f"{target}_breach_probability_classifier.joblib")
        joblib.dump(clf, model_path)

        calib_scores = clf.predict_proba(X_calib_target)[:, 1]
        manifest["classifiers"][target] = {
            "model_path": model_path,
            "feature_columns": target_features,
            "train_prevalence": float(breach_labels[target].mean()),
            "calib_score_mean": float(np.mean(calib_scores)),
            "calib_score_std": float(np.std(calib_scores)),
        }
        print(
            f"[{target}] prevalence={manifest['classifiers'][target]['train_prevalence']:.4f} "
            f"calib_score_mean={manifest['classifiers'][target]['calib_score_mean']:.4f}"
        )

    with open(CRC_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Saved CRC classifier artifacts:", CRC_MODEL_DIR)
    return manifest


def main() -> None:
    maybe_install_dependencies()
    ensure_time_indexed_dataset(RAW_DATA_PATH, DATA_WITH_TIME_PATH)
    create_time_splits(DATA_WITH_TIME_PATH, SPLIT_DIR)
    build_and_save_prepared_features(
        data_path=DATA_WITH_TIME_PATH,
        split_dir=SPLIT_DIR,
        out_path=PREPARED_FEATURES_PATH,
        base_exog=BASE_EXOG,
    )
    train_surrogates()
    build_conformal_bounds()
    report_surrogate_performance()
    train_crc_breach_classifiers()
    export_journal_results_suite()
    print("\nSurrogate training complete.")
    print("When accuracy looks good, run RiskToCashFrontier.py.")


if __name__ == "__main__":
    main()
