from __future__ import annotations

import argparse
import os

from shared_config import RISK_TO_CASH_RESULTS_BUNDLE_PATH

# In-code plot saving toggle.
# Set SAVE_PLOTS=True to save plots automatically on every run.
SAVE_PLOTS = True
# Optional fixed save directory (None -> "<bundle_dir>/plots").
SAVE_PLOTS_DIR = None
# Default image format when saving plots.
SAVE_PLOTS_FORMAT = "png"


def _row_or_none(df: pd.DataFrame, idx) -> pd.DataFrame | None:
    try:
        return df.loc[[idx]]
    except (KeyError, TypeError, ValueError):
        return None


def _row_scalar(row: pd.DataFrame | None, col: str) -> float:
    """Safely extract a scalar from a single-row DataFrame."""
    if row is None or row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def compute_knee_point(df_alpha: pd.DataFrame, x_col: str, y_col: str):
    df_sorted = df_alpha.dropna(subset=[x_col, y_col]).sort_values(x_col)
    if len(df_sorted) < 3:
        return None
    a = df_sorted.iloc[0][[x_col, y_col]].to_numpy()
    b = df_sorted.iloc[-1][[x_col, y_col]].to_numpy()
    ab = b - a
    ab_norm = np.linalg.norm(ab)
    if ab_norm == 0:
        return None

    distances = []
    for idx, row in df_sorted.iterrows():
        p = np.array([row[x_col], row[y_col]])
        # 2D point-to-line distance using determinant form; avoids np.cross 2D deprecation.
        ap = p - a
        cross_mag = (ab[0] * ap[1]) - (ab[1] * ap[0])
        dist = np.abs(cross_mag) / ab_norm
        distances.append((idx, dist))

    knee_idx, _ = max(distances, key=lambda t: t[1])
    return df_sorted.loc[[knee_idx]]


def compute_nondominated_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Return globally non-dominated points for:
      - minimize x_col (emissions)
      - maximize y_col (money)
    """
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        return d

    x = d[x_col].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    dominated = np.zeros(len(d), dtype=bool)

    for i in range(len(d)):
        better_or_equal = (x <= x[i]) & (y >= y[i])
        strictly_better = (x < x[i]) | (y > y[i])
        if np.any(better_or_equal & strictly_better):
            dominated[i] = True

    return d.loc[~dominated].sort_values([x_col, y_col], ascending=[True, False])


def densify_heatmap(grid: pd.DataFrame, alpha_points: int = 200, epsilon_points: int = 400):
    if grid.empty:
        return None, None, None

    cols = grid.columns.to_numpy()
    rows = grid.index.to_numpy()
    vals = grid.to_numpy()

    eps_dense = np.linspace(cols.min(), cols.max(), epsilon_points)
    alpha_dense = np.linspace(rows.min(), rows.max(), alpha_points)

    vals_eps = np.array([np.interp(eps_dense, cols, row) for row in vals])
    vals_dense = np.array([np.interp(alpha_dense, rows, vals_eps[:, j]) for j in range(vals_eps.shape[1])]).T
    return vals_dense, alpha_dense, eps_dense


def _get_trace(simulation_traces: dict, alpha_val: float, epsilon_val: float, atol: float = 1e-9):
    direct = simulation_traces.get((alpha_val, epsilon_val))
    if direct is not None:
        return direct
    for key, trace in simulation_traces.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        a, e = key
        if np.isclose(float(a), float(alpha_val), atol=atol, rtol=0.0) and np.isclose(
            float(e), float(epsilon_val), atol=atol, rtol=0.0
        ):
            return trace
    return None


def _get_frontier_trace(frontier_traces: dict, alpha_val: float, atol: float = 1e-9):
    direct = frontier_traces.get(alpha_val)
    if direct is not None:
        return direct
    for key, trace in frontier_traces.items():
        try:
            if np.isclose(float(key), float(alpha_val), atol=atol, rtol=0.0):
                return trace
        except (TypeError, ValueError):
            continue
    return None


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    p = float(successes) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, center - radius), min(1.0, center + radius)


def _safe_pct_delta(series: pd.Series, baseline: float) -> pd.Series:
    if baseline == 0 or np.isnan(baseline):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return 100.0 * (series.astype(float) - float(baseline)) / float(abs(baseline))


def _energy_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = values.astype(float).fillna(0.0)
    weights = weights.astype(float).fillna(0.0)
    wsum = float(weights.sum())
    if wsum <= 0:
        return np.nan
    return float(np.dot(values, weights) / wsum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render risk-to-cash plots from saved simulation outputs.")
    parser.add_argument(
        "--bundle-path",
        default=RISK_TO_CASH_RESULTS_BUNDLE_PATH,
        help="Path to risk-to-cash results bundle generated by RiskToCashFrontier.py",
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save-plots",
        dest="save_plots_cli",
        action="store_true",
        help="Override in-code SAVE_PLOTS to True.",
    )
    save_group.add_argument(
        "--no-save-plots",
        dest="save_plots_cli",
        action="store_false",
        help="Override in-code SAVE_PLOTS to False.",
    )
    parser.set_defaults(save_plots_cli=None)
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory where plots are saved (default: SAVE_PLOTS_DIR or <bundle_dir>/plots).",
    )
    parser.add_argument(
        "--plot-format",
        default=None,
        choices=["png", "pdf", "svg"],
        help="Image format used when saving plots (default: SAVE_PLOTS_FORMAT).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global np, pd
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt

    if not os.path.exists(args.bundle_path):
        raise FileNotFoundError(
            "Results bundle not found. Run RiskToCashFrontier.py first.\n"
            f"Missing file: {args.bundle_path}"
        )

    bundle = joblib.load(args.bundle_path)
    risk_to_cash_df = bundle.get("risk_to_cash_df")
    pareto_eps_df = bundle.get("pareto_eps_df")
    pareto_by_alpha = bundle.get("pareto_by_alpha", {})
    simulation_traces = bundle.get("simulation_traces", {})
    frontier_simulation_traces = bundle.get("frontier_simulation_traces", {})

    if not isinstance(risk_to_cash_df, pd.DataFrame):
        raise ValueError("Bundle missing 'risk_to_cash_df' DataFrame.")
    if not isinstance(pareto_eps_df, pd.DataFrame):
        raise ValueError("Bundle missing 'pareto_eps_df' DataFrame.")

    alphas = bundle.get("alphas")
    if alphas is None:
        alphas = sorted(pd.unique(risk_to_cash_df["alpha"]).tolist())
    alphas = list(alphas)
    if not alphas:
        raise ValueError("Could not determine alpha grid from bundle.")

    plot_alpha = float(bundle.get("plot_alpha", alphas[0]))
    x_emission = str(bundle.get("x_emission_col", "energy_weighted_nox"))
    y_money = str(bundle.get("y_money_col", "total_economic_objective"))

    save_plots = bool(SAVE_PLOTS) if args.save_plots_cli is None else bool(args.save_plots_cli)
    default_plots_dir = SAVE_PLOTS_DIR if SAVE_PLOTS_DIR else os.path.join(os.path.dirname(os.path.abspath(args.bundle_path)), "plots")
    plots_dir = str(args.plots_dir) if args.plots_dir else str(default_plots_dir)
    plot_format = str(args.plot_format) if args.plot_format else str(SAVE_PLOTS_FORMAT)
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Plot saving enabled -> {plots_dir} (*.{plot_format})")
    saved_plot_counter = 0

    def _safe_plot_name(name: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(name)).strip("_")

    def _save_current_plot(name: str) -> None:
        nonlocal saved_plot_counter
        if not save_plots:
            return
        fig = plt.gcf()
        if fig is None:
            return
        saved_plot_counter += 1
        filename = f"{saved_plot_counter:02d}_{_safe_plot_name(name)}.{plot_format}"
        out_path = os.path.join(plots_dir, filename)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {out_path}")

    risk_summary_cols = bundle.get(
        "risk_summary_columns",
        [
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
    )
    epsilon_summary_cols = bundle.get(
        "epsilon_summary_columns",
        [
            "alpha",
            "epsilon_nox_cap",
            "mean_sim_nox",
            "mean_sim_nox_feasible",
            "total_economic_objective",
            "sim_any_breach_rate",
            "conf_any_violation_rate",
        ],
    )

    if risk_summary_cols:
        available = [c for c in risk_summary_cols if c in risk_to_cash_df.columns]
        if available:
            print("\n EMPC+CRC Risk→Cash Summary (per α) ")
            print(risk_to_cash_df[available].to_string(index=False))

    # Quick textual feasibility diagnostics for easier run-to-run interpretation.
    if {"alpha", "feasible_hours", "infeasible_hours", "feasible_fraction"}.issubset(risk_to_cash_df.columns):
        frontier_feas_df = risk_to_cash_df[
            ["alpha", "feasible_hours", "infeasible_hours", "feasible_fraction"]
        ].copy()
        frontier_feas_df["feasible_hours"] = frontier_feas_df["feasible_hours"].fillna(0).astype(int)
        frontier_feas_df["infeasible_hours"] = frontier_feas_df["infeasible_hours"].fillna(0).astype(int)
        frontier_feas_df["total_hours"] = (
            frontier_feas_df["feasible_hours"] + frontier_feas_df["infeasible_hours"]
        ).astype(int)
        frontier_feas_df["feasible_fraction"] = frontier_feas_df["feasible_fraction"].astype(float)
        frontier_feas_df = frontier_feas_df.sort_values("alpha")

        best_row = frontier_feas_df.loc[frontier_feas_df["feasible_fraction"].idxmax()]
        worst_row = frontier_feas_df.loc[frontier_feas_df["feasible_fraction"].idxmin()]
        print("\n Feasibility evaluation (frontier runs, per α) ")
        print(
            frontier_feas_df[
                ["alpha", "feasible_hours", "infeasible_hours", "total_hours", "feasible_fraction"]
            ].to_string(index=False)
        )
        print(
            "Best α by feasible share: "
            f"{float(best_row['alpha']):.3f} -> {float(best_row['feasible_fraction']):.2%} "
            f"({int(best_row['feasible_hours'])}/{int(best_row['total_hours'])} feasible hours)."
        )
        print(
            "Worst α by feasible share: "
            f"{float(worst_row['alpha']):.3f} -> {float(worst_row['feasible_fraction']):.2%} "
            f"({int(worst_row['feasible_hours'])}/{int(worst_row['total_hours'])} feasible hours)."
        )

    if {"alpha", "epsilon_nox_cap", "feasible_fraction"}.issubset(pareto_eps_df.columns):
        eps_feas = pareto_eps_df["feasible_fraction"].astype(float)
        n_points = int(len(eps_feas))
        n_zero = int((eps_feas <= 0.0).sum())
        per_alpha_best = (
            pareto_eps_df.groupby("alpha", as_index=False)["feasible_fraction"]
            .max()
            .sort_values("alpha")
        )
        print("\n Feasibility evaluation (ε-grid across all α) ")
        print(
            f"Across {n_points} α-ε points: min={float(eps_feas.min()):.4f}, "
            f"median={float(eps_feas.median()):.4f}, max={float(eps_feas.max()):.4f} feasible fraction."
        )
        print(f"Fully infeasible α-ε points (0 feasible hours): {n_zero}/{n_points}.")
        print("Best feasible fraction per α across ε:")
        print(per_alpha_best.to_string(index=False))

    def _format_case_key(case_key) -> str:
        if isinstance(case_key, tuple) and len(case_key) == 2:
            try:
                return f"(α={float(case_key[0]):.3f}, ε={float(case_key[1]):.2f})"
            except (TypeError, ValueError):
                return str(case_key)
        try:
            return f"α={float(case_key):.3f}"
        except (TypeError, ValueError):
            return str(case_key)

    def _print_blocker_breakdown(traces: dict, label: str, top_n: int = 10) -> None:
        if not isinstance(traces, dict) or not traces:
            return

        per_case_rows = []
        total_hours = 0
        total_infeasible = 0
        agg = {"crc_gate": 0, "nox_ub": 0, "tit_ub": 0, "co_ub": 0, "epsilon_gate": 0, "other": 0}

        for case_key, trace_df in traces.items():
            if not isinstance(trace_df, pd.DataFrame) or trace_df.empty:
                continue
            if "optimizer_status" not in trace_df.columns or "primary_blocker" not in trace_df.columns:
                continue

            n_hours = int(len(trace_df))
            infeasible_mask = trace_df["optimizer_status"].astype(str).eq("infeasible")
            n_infeasible = int(infeasible_mask.sum())
            total_hours += n_hours
            total_infeasible += n_infeasible

            if n_infeasible == 0:
                per_case_rows.append(
                    {
                        "case": _format_case_key(case_key),
                        "hours": n_hours,
                        "infeasible_hours": 0,
                        "infeasible_%": 0.0,
                        "crc_gate": 0,
                        "cp_limits_total": 0,
                        "epsilon_gate": 0,
                        "other": 0,
                    }
                )
                continue

            vc = (
                trace_df.loc[infeasible_mask, "primary_blocker"]
                .fillna("other")
                .astype(str)
                .value_counts()
            )
            c_crc = int(vc.get("crc_gate", 0))
            c_nox = int(vc.get("nox_ub", 0))
            c_tit = int(vc.get("tit_ub", 0))
            c_co = int(vc.get("co_ub", 0))
            c_eps = int(vc.get("epsilon_gate", 0))
            c_cp_total = c_nox + c_tit + c_co
            c_other = int(n_infeasible - (c_crc + c_cp_total + c_eps))

            agg["crc_gate"] += c_crc
            agg["nox_ub"] += c_nox
            agg["tit_ub"] += c_tit
            agg["co_ub"] += c_co
            agg["epsilon_gate"] += c_eps
            agg["other"] += c_other

            per_case_rows.append(
                {
                    "case": _format_case_key(case_key),
                    "hours": n_hours,
                    "infeasible_hours": n_infeasible,
                    "infeasible_%": 100.0 * n_infeasible / n_hours if n_hours else 0.0,
                    "crc_gate": c_crc,
                    "cp_limits_total": c_cp_total,
                    "epsilon_gate": c_eps,
                    "other": c_other,
                }
            )

        if total_hours == 0:
            return

        print(f"\n {label} ")
        print(
            f"Total infeasible hours: {total_infeasible}/{total_hours} "
            f"({(100.0 * total_infeasible / total_hours):.2f}%)."
        )
        if total_infeasible > 0:
            cp_total = int(agg["nox_ub"] + agg["tit_ub"] + agg["co_ub"])
            print(f" - crc_gate: {agg['crc_gate']} ({(100.0 * agg['crc_gate'] / total_infeasible):.2f}%)")
            print(f" - cp_limits_total: {cp_total} ({(100.0 * cp_total / total_infeasible):.2f}%)")
            print(f"   - nox_ub: {agg['nox_ub']}")
            print(f"   - tit_ub: {agg['tit_ub']}")
            print(f"   - co_ub: {agg['co_ub']}")
            print(
                f" - epsilon_gate: {agg['epsilon_gate']} "
                f"({(100.0 * agg['epsilon_gate'] / total_infeasible):.2f}%)"
            )
            print(f" - other: {agg['other']} ({(100.0 * agg['other'] / total_infeasible):.2f}%)")

        if per_case_rows:
            per_case_df = pd.DataFrame(per_case_rows)
            per_case_df = per_case_df.sort_values(["infeasible_%", "infeasible_hours"], ascending=False)
            show_n = min(int(top_n), len(per_case_df))
            print(f"Top {show_n} cases by infeasible fraction:")
            print(
                per_case_df[
                    ["case", "hours", "infeasible_hours", "infeasible_%", "crc_gate", "cp_limits_total", "epsilon_gate", "other"]
                ]
                .head(show_n)
                .to_string(index=False)
            )

    _print_blocker_breakdown(frontier_simulation_traces, "Blocker breakdown (frontier α runs)", top_n=max(5, len(alphas)))
    _print_blocker_breakdown(simulation_traces, "Blocker breakdown (ε-sweep runs)", top_n=10)

    # 1) 3x2 risk-to-cash summary plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()

    axes[0].plot(risk_to_cash_df["alpha"], risk_to_cash_df["total_economic_objective"], marker="o", linestyle="-")
    axes[0].set_xlabel("α (risk tolerance)")
    axes[0].set_ylabel("Total economic objective (arb. £ units)")
    axes[0].set_title("Risk→Cash Frontier: Economics vs α")
    axes[0].grid(True)

    axes[1].plot(risk_to_cash_df["alpha"], risk_to_cash_df["total_mwh"], marker="o", linestyle="-")
    axes[1].set_xlabel("α (risk tolerance)")
    axes[1].set_ylabel("Total MWh over test horizon")
    axes[1].set_title("Risk→Cash Frontier: Energy vs α")
    axes[1].grid(True)

    axes[2].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["alpha"],
        linestyle="--",
        label="Reference α (CRC calibrated on hourly any-breach rate)",
    )
    axes[2].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["sim_any_breach_indicator"],
        marker="o",
        linestyle="-",
        label="Realised any-breach indicator (per-trajectory, diagnostic)",
    )
    axes[2].set_xlabel("α (risk tolerance)")
    axes[2].set_ylabel("Any-breach indicator (fraction of trajectories)")
    axes[2].set_title("Per-trajectory any-breach indicator (diagnostic only)")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(
        risk_to_cash_df["alpha"],
        np.clip(
            risk_to_cash_df["alpha"].astype(float).to_numpy()
            * max(
                1,
                int(
                    sum(
                        int(c in risk_to_cash_df.columns)
                        for c in ["conf_nox_violation_rate", "conf_tit_violation_rate", "conf_co_violation_rate"]
                    )
                ),
            ),
            0.0,
            1.0,
        ),
        linestyle="--",
        label="Union reference m·α (Bonferroni upper bound)",
    )
    axes[3].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["conf_any_violation_rate"],
        marker="o",
        linestyle="-",
        label="Realised any-pollutant conformal violations (feasible hours)",
    )
    axes[3].set_xlabel("α (risk tolerance)")
    axes[3].set_ylabel("Violation rate (fraction of feasible hours)")
    axes[3].set_title("Conformal union risk (feasible hours)")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["sim_nox_breach_rate"],
        marker="x",
        linestyle="--",
        label="NOX limit breach",
    )
    axes[4].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["sim_tit_breach_rate"],
        marker="s",
        linestyle="--",
        label="TIT limit breach",
    )
    axes[4].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["sim_co_breach_rate"],
        marker="^",
        linestyle="--",
        label="CO limit breach",
    )
    axes[4].set_xlabel("α (risk tolerance)")
    axes[4].set_ylabel("Breach rate (fraction of feasible hours)")
    axes[4].set_title("Per-pollutant breach rates (feasible hours)")
    axes[4].grid(True)
    axes[4].legend()

    axes[5].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["conf_nox_violation_rate"],
        marker="x",
        linestyle="--",
        label="NOX: P(sim > UB)",
    )
    axes[5].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["conf_tit_violation_rate"],
        marker="s",
        linestyle="--",
        label="TIT: P(sim > UB)",
    )
    axes[5].plot(
        risk_to_cash_df["alpha"],
        risk_to_cash_df["conf_co_violation_rate"],
        marker="^",
        linestyle="--",
        label="CO: P(sim > UB)",
    )
    axes[5].plot(risk_to_cash_df["alpha"], risk_to_cash_df["alpha"], linestyle=":", label="Target α")
    axes[5].set_xlabel("α (risk tolerance)")
    axes[5].set_ylabel("Violation rate (fraction of feasible hours)")
    axes[5].set_title("Conformal safety: per-pollutant coverage violations (feasible hours)")
    axes[5].grid(True)
    axes[5].legend()

    plt.tight_layout()
    _save_current_plot("risk_to_cash_summary_3x2")
    plt.show(block=False)
    plt.pause(0.001)

    # 1b) Realised risk vs target alpha with 95% binomial confidence intervals
    risk_plot_df = risk_to_cash_df.sort_values("alpha").copy()
    if {"feasible_hours"}.issubset(risk_plot_df.columns):
        risk_plot_df["n_hours_eval"] = risk_plot_df["feasible_hours"].fillna(0).astype(float).astype(int)
    else:
        # fallback if counts were not stored in the bundle
        risk_plot_df["n_hours_eval"] = 0
        for idx, row in risk_plot_df.iterrows():
            a = float(row["alpha"])
            trace = _get_frontier_trace(frontier_simulation_traces, a)
            risk_plot_df.loc[idx, "n_hours_eval"] = int(len(trace.dropna(subset=["sim_TIT"]))) if trace is not None else 0

    metric_specs = [
        ("sim_any_breach_rate", "Any pollutant breach"),
        ("sim_nox_breach_rate", "NOX breach"),
        ("sim_tit_breach_rate", "TIT breach"),
        ("sim_co_breach_rate", "CO breach"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, metric_specs):
        if col not in risk_plot_df.columns:
            ax.set_visible(False)
            continue

        x_alpha = risk_plot_df["alpha"].astype(float).to_numpy()
        y_rate = risk_plot_df[col].astype(float).to_numpy()
        n_hours = risk_plot_df["n_hours_eval"].astype(int).to_numpy()

        ci_low = np.full_like(y_rate, np.nan, dtype=float)
        ci_high = np.full_like(y_rate, np.nan, dtype=float)
        for i, (p_hat, n) in enumerate(zip(y_rate, n_hours)):
            if n <= 0 or np.isnan(p_hat):
                continue
            s = int(round(float(p_hat) * int(n)))
            lo, hi = _wilson_interval(s, int(n))
            ci_low[i] = lo
            ci_high[i] = hi

        yerr_low = np.clip(y_rate - ci_low, 0.0, None)
        yerr_high = np.clip(ci_high - y_rate, 0.0, None)
        yerr = np.vstack([yerr_low, yerr_high])

        ax.plot(x_alpha, x_alpha, "k--", linewidth=1.2, label="Target line: realised risk = α")
        ax.errorbar(
            x_alpha,
            y_rate,
            yerr=yerr,
            fmt="o-",
            capsize=3,
            linewidth=1.8,
            label="Realised risk with 95% Wilson CI",
        )
        ax.set_title(label)
        ax.set_xlabel("Target risk tolerance α")
        ax.set_ylabel("Realised breach rate (feasible hours)")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")

    fig.suptitle("Risk control check: realised breach rates (feasible hours) vs target α", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    _save_current_plot("risk_control_wilson_ci")
    plt.show(block=False)
    plt.pause(0.001)

    # 1c) Risk budget utilisation: realised any-breach divided by alpha target
    if {"alpha", "sim_any_breach_rate"}.issubset(risk_to_cash_df.columns):
        util_df = risk_to_cash_df.sort_values("alpha").dropna(subset=["alpha", "sim_any_breach_rate"]).copy()
        if not util_df.empty:
            alpha_vals = util_df["alpha"].astype(float).to_numpy()
            breach_vals = util_df["sim_any_breach_rate"].astype(float).to_numpy()
            util_vals = np.full_like(alpha_vals, np.nan, dtype=float)
            valid_alpha = alpha_vals > 0.0
            util_vals[valid_alpha] = breach_vals[valid_alpha] / alpha_vals[valid_alpha]

            plt.figure(figsize=(8, 4.8))
            plt.axhline(1.0, color="k", linestyle="--", linewidth=1.2, label="Full risk-budget use (ratio=1)")
            plt.plot(alpha_vals, util_vals, marker="o", linewidth=1.8, label="Realised any-breach / α")
            plt.xlabel("α (risk tolerance)")
            plt.ylabel("Risk budget utilisation ratio")
            plt.title("CRC risk-budget utilisation across α")
            plt.grid(True, alpha=0.35)
            plt.legend(loc="best")
            plt.tight_layout()
            _save_current_plot("risk_budget_utilisation")
            plt.show(block=False)
            plt.pause(0.001)

            util_report = pd.DataFrame(
                {
                    "alpha": alpha_vals,
                    "sim_any_breach_rate": breach_vals,
                    "risk_budget_utilisation": util_vals,
                }
            )
            print("\n Risk budget utilisation (realised any-breach / α) ")
            print(util_report.to_string(index=False))

    # 1d) Shadow price of risk: marginal money/energy gains per +alpha
    if {"alpha", "total_economic_objective", "total_mwh"}.issubset(risk_to_cash_df.columns):
        shadow_df = risk_to_cash_df.sort_values("alpha").dropna(subset=["alpha", "total_economic_objective", "total_mwh"]).copy()
        if len(shadow_df) >= 2:
            alpha_vals = shadow_df["alpha"].astype(float).to_numpy()
            money_vals = shadow_df["total_economic_objective"].astype(float).to_numpy()
            mwh_vals = shadow_df["total_mwh"].astype(float).to_numpy()

            delta_alpha = np.diff(alpha_vals)
            alpha_mid = 0.5 * (alpha_vals[:-1] + alpha_vals[1:])
            valid = delta_alpha != 0.0

            dmoney_dalpha = np.full_like(alpha_mid, np.nan, dtype=float)
            dmwh_dalpha = np.full_like(alpha_mid, np.nan, dtype=float)
            dmoney_dalpha[valid] = np.diff(money_vals)[valid] / delta_alpha[valid]
            dmwh_dalpha[valid] = np.diff(mwh_vals)[valid] / delta_alpha[valid]

            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            axes[0].plot(alpha_mid, dmoney_dalpha, marker="o", linewidth=1.8, color="#1f77b4")
            axes[0].set_ylabel("Δ£ / Δα")
            axes[0].set_title("Shadow price of risk tolerance")
            axes[0].grid(True, alpha=0.35)

            axes[1].plot(alpha_mid, dmwh_dalpha, marker="o", linewidth=1.8, color="#ff7f0e")
            axes[1].set_xlabel("Midpoint α between adjacent runs")
            axes[1].set_ylabel("ΔMWh / Δα")
            axes[1].grid(True, alpha=0.35)

            plt.tight_layout()
            _save_current_plot("shadow_price_of_risk")
            plt.show(block=False)
            plt.pause(0.001)

            shadow_report = pd.DataFrame(
                {
                    "alpha_left": alpha_vals[:-1],
                    "alpha_right": alpha_vals[1:],
                    "alpha_mid": alpha_mid,
                    "delta_money_per_alpha": dmoney_dalpha,
                    "delta_mwh_per_alpha": dmwh_dalpha,
                }
            )
            print("\n Shadow price of risk increments ")
            print(shadow_report.to_string(index=False))

    # 2) Pareto fronts (use a single emission metric consistently)
    emission_axis_labels = {
        "energy_weighted_nox": "Energy-weighted NOX (mg/m³, proxy, TEY-weighted)",
        "mean_sim_nox": "Average simulated NOX (mg/m³, proxy)",
        "mean_sim_nox_feasible": "Average simulated NOX (mg/m³, proxy) — feasible hours only",
    }
    emission_axis_label = emission_axis_labels.get(x_emission, x_emission)

    plt.figure(figsize=(8, 6))
    for alpha_val in alphas:
        df_alpha = (
            pareto_eps_df[pareto_eps_df["alpha"] == alpha_val]
            .dropna(subset=[x_emission, y_money])
            .sort_values(x_emission)
        )
        if df_alpha.empty:
            continue
        plt.plot(
            df_alpha[x_emission],
            df_alpha[y_money],
            marker="o",
            linestyle="-",
            label=f"α = {alpha_val}",
        )
        for _, row in df_alpha.iterrows():
            plt.annotate(
                f"ε={row['epsilon_nox_cap']:.1f}",
                (row[x_emission], row[y_money]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    plt.xlabel(emission_axis_label)
    plt.ylabel("Total economic objective (arb. £ units)")
    plt.title("ε-constraint Pareto frontiers across α (EMPC+CRC)")
    plt.grid(True)
    plt.legend(title="α (risk tolerance)")
    plt.tight_layout()
    _save_current_plot("pareto_frontiers_all")
    plt.show(block=False)
    plt.pause(0.001)

    plt.figure(figsize=(8, 6))
    for alpha_val in alphas:
        df_alpha_feas = pareto_eps_df[pareto_eps_df["alpha"] == alpha_val].copy()
        if "feasible_hours" in df_alpha_feas.columns:
            df_alpha_feas = df_alpha_feas[df_alpha_feas["feasible_hours"].fillna(0).astype(float) > 0]
        elif "feasible_fraction" in df_alpha_feas.columns:
            df_alpha_feas = df_alpha_feas[df_alpha_feas["feasible_fraction"].fillna(0.0).astype(float) > 0.0]

        df_alpha_feas = (
            df_alpha_feas
            .dropna(subset=[x_emission, y_money])
            .sort_values(x_emission)
        )
        if df_alpha_feas.empty:
            continue
        plt.plot(
            df_alpha_feas[x_emission],
            df_alpha_feas[y_money],
            marker="o",
            linestyle="-",
            label=f"α = {alpha_val}",
        )
        for _, row in df_alpha_feas.iterrows():
            plt.annotate(
                f"ε={row['epsilon_nox_cap']:.1f}",
                (row[x_emission], row[y_money]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    plt.xlabel(f"{emission_axis_label} — rows with feasible operation")
    plt.ylabel("Total economic objective (arb. £ units)")
    plt.title("ε-constraint Pareto (feasible-operation rows)")
    plt.grid(True)
    plt.legend(title="α (risk tolerance)")
    plt.tight_layout()
    _save_current_plot("pareto_frontiers_feasible_rows")
    plt.show(block=False)
    plt.pause(0.001)

    # 2a) Global Pareto cloud (all α,ε points) + non-dominated envelope
    if {"alpha", x_emission, y_money}.issubset(pareto_eps_df.columns):
        cloud_df = pareto_eps_df.dropna(subset=["alpha", x_emission, y_money]).copy()
        if not cloud_df.empty:
            nd_df = compute_nondominated_front(cloud_df, x_emission, y_money)

            if "feasible_fraction" in cloud_df.columns:
                sizes = 30.0 + 140.0 * cloud_df["feasible_fraction"].astype(float).clip(0.0, 1.0).to_numpy()
            else:
                sizes = np.full(len(cloud_df), 55.0, dtype=float)

            plt.figure(figsize=(9, 6))
            sc = plt.scatter(
                cloud_df[x_emission],
                cloud_df[y_money],
                c=cloud_df["alpha"].astype(float),
                s=sizes,
                cmap="viridis",
                alpha=0.55,
                edgecolors="none",
                label="All α-ε operating points",
            )
            if not nd_df.empty:
                plt.plot(
                    nd_df[x_emission],
                    nd_df[y_money],
                    color="black",
                    linewidth=2.0,
                    label="Global non-dominated envelope",
                )
                plt.scatter(nd_df[x_emission], nd_df[y_money], color="black", s=20)

            cbar = plt.colorbar(sc)
            cbar.set_label("α (risk tolerance)")
            plt.xlabel(emission_axis_label)
            plt.ylabel("Total economic objective (arb. £ units)")
            plt.title("Global α-ε Pareto cloud with non-dominated envelope")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()
            _save_current_plot("global_pareto_cloud_nondominated")
            plt.show(block=False)
            plt.pause(0.001)

            if not nd_df.empty:
                print(
                    "\n Global Pareto envelope summary: "
                    f"{len(nd_df)} non-dominated points out of {len(cloud_df)} total α-ε points."
                )

    if epsilon_summary_cols:
        avail_eps_cols = [c for c in epsilon_summary_cols if c in pareto_eps_df.columns]
        if avail_eps_cols:
            print("\n ε-constraint summary (all α, sorted by α and ε) ")
            print(pareto_eps_df[avail_eps_cols].to_string(index=False))

    # 2b) Feasibility map over alpha-epsilon
    if {"alpha", "epsilon_nox_cap", "feasible_fraction"}.issubset(pareto_eps_df.columns):
        feasibility_grid = pareto_eps_df.pivot(index="alpha", columns="epsilon_nox_cap", values="feasible_fraction")
        feasibility_grid = feasibility_grid.sort_index().sort_index(axis=1)
        if not feasibility_grid.empty:
            plt.figure(figsize=(9, 5.5))
            plt.imshow(
                feasibility_grid.values,
                aspect="auto",
                origin="lower",
                extent=[
                    feasibility_grid.columns.min(),
                    feasibility_grid.columns.max(),
                    feasibility_grid.index.min(),
                    feasibility_grid.index.max(),
                ],
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            plt.colorbar(label="Feasible-hour fraction (0 = none feasible, 1 = all feasible)")
            plt.xlabel("ε NOX cap (mg/m³, proxy)")
            plt.ylabel("α (risk tolerance)")
            plt.title("Feasibility map over α–ε grid")
            plt.tight_layout()
            _save_current_plot("feasibility_map_alpha_epsilon")
            plt.show(block=False)
            plt.pause(0.001)

    # 3) Derived analytics: knee + Form B + MAC
    if not pareto_by_alpha:
        pareto_by_alpha = {a: pareto_eps_df[pareto_eps_df["alpha"] == a].copy() for a in alphas}

    knee_summary_rows = []
    plt.figure(figsize=(8, 6))
    for alpha_val in alphas:
        df_alpha = pareto_by_alpha.get(alpha_val)
        if df_alpha is None or df_alpha.empty:
            continue
        df_alpha = df_alpha.dropna(subset=[x_emission, y_money, "epsilon_nox_cap"]).sort_values(x_emission)
        if df_alpha.empty:
            continue
        plt.plot(df_alpha[x_emission], df_alpha[y_money], marker="o", linestyle="-", label=f"α = {alpha_val}")

        baseline_row = _row_or_none(df_alpha, df_alpha["epsilon_nox_cap"].idxmax()) if not df_alpha.empty else None
        clean_row = _row_or_none(df_alpha, df_alpha[x_emission].idxmin()) if not df_alpha.empty else None
        knee_row = compute_knee_point(df_alpha, x_emission, y_money)

        for row, marker, label in [
            (baseline_row, "s", "baseline (ε max)"),
            (clean_row, "v", "clean (min emission)"),
            (knee_row, "D", "knee"),
        ]:
            if row is None or row.empty:
                continue
            plt.scatter(row[x_emission], row[y_money], marker=marker, s=80, label=f"α={alpha_val} {label}")

        if knee_row is not None and not knee_row.empty and baseline_row is not None and not baseline_row.empty:
            base_money = _row_scalar(baseline_row, y_money)
            base_emis = _row_scalar(baseline_row, x_emission)
            knee_money = _row_scalar(knee_row, y_money)
            knee_emis = _row_scalar(knee_row, x_emission)
            clean_emis = _row_scalar(clean_row, x_emission)
            knee_summary_rows.append(
                {
                    "alpha": alpha_val,
                    "knee_energy_weighted_nox": knee_emis,
                    "knee_money": knee_money,
                    "money_loss_vs_baseline_%": 100.0 * (base_money - knee_money) / base_money if base_money else np.nan,
                    "emission_reduction_vs_baseline_%": 100.0 * (base_emis - knee_emis) / base_emis if base_emis else np.nan,
                    "emission_reduction_vs_min_%": 100.0 * (knee_emis - clean_emis) / clean_emis if clean_emis else np.nan,
                }
            )

    plt.xlabel("Energy-weighted NOX (mg/m³, proxy, TEY-weighted)")
    plt.ylabel("Total economic objective (arb. £ units)")
    plt.title("Pareto fronts (ε-constraint) with knee/baseline/clean markers")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _save_current_plot("pareto_knee_baseline_clean")
    plt.show(block=False)
    plt.pause(0.001)

    if knee_summary_rows:
        knee_summary_df = pd.DataFrame(knee_summary_rows)
        print("\n Knee point summary (energy-weighted NOX) ")
        print(knee_summary_df.to_string(index=False))

    form_b_rows = []
    money_floor_fracs = [0.90, 0.95, 0.99]
    for alpha_val in alphas:
        df_alpha = pareto_by_alpha.get(alpha_val, pd.DataFrame()).dropna(subset=[x_emission, y_money, "epsilon_nox_cap"])
        if df_alpha.empty:
            continue
        baseline_row = _row_or_none(df_alpha, df_alpha["epsilon_nox_cap"].idxmax())
        if baseline_row is None or baseline_row.empty:
            continue
        base_money = _row_scalar(baseline_row, y_money)
        for frac in money_floor_fracs:
            floor = base_money * frac
            feasible = df_alpha[df_alpha[y_money] >= floor]
            if feasible.empty:
                continue
            best = _row_or_none(feasible, feasible[x_emission].idxmin())
            if best is None or best.empty:
                continue
            form_b_rows.append(
                {
                    "alpha": alpha_val,
                    "money_floor_frac": frac,
                    "money_floor_abs": floor,
                    "chosen_energy_weighted_nox": _row_scalar(best, x_emission),
                    "chosen_money": _row_scalar(best, y_money),
                    "epsilon_used": _row_scalar(best, "epsilon_nox_cap"),
                }
            )
    if form_b_rows:
        form_b_df = pd.DataFrame(form_b_rows)
        print("\n Form B (emissions-first) selections from ε-frontiers ")
        print(form_b_df.to_string(index=False))

    mac_alpha = plot_alpha if plot_alpha in alphas else alphas[0]
    df_mac = pareto_by_alpha.get(mac_alpha, pd.DataFrame()).dropna(subset=[x_emission, y_money])
    df_mac = df_mac.sort_values(x_emission, ascending=False)

    mac_rows = []
    if not df_mac.empty and len(df_mac) > 1:
        baseline_emission = float(df_mac.iloc[0][x_emission])
        baseline_money = float(df_mac.iloc[0][y_money])
        for i in range(len(df_mac) - 1):
            e0, e1 = float(df_mac.iloc[i][x_emission]), float(df_mac.iloc[i + 1][x_emission])
            m0, m1 = float(df_mac.iloc[i][y_money]), float(df_mac.iloc[i + 1][y_money])
            delta_e = e1 - e0
            delta_m = m1 - m0
            if delta_e == 0:
                continue
            mac = -delta_m / delta_e
            cumulative_reduction = baseline_emission - e1
            mac_rows.append(
                {
                    "segment_start_emission": e0,
                    "segment_end_emission": e1,
                    "money_start": m0,
                    "money_end": m1,
                    "mac": mac,
                    "cumulative_reduction": cumulative_reduction,
                }
            )

    if mac_rows:
        mac_df = pd.DataFrame(mac_rows)
        print(f"\n Marginal abatement cost (α={mac_alpha}) ")
        print(mac_df.to_string(index=False))

        plt.figure(figsize=(8, 4))
        plt.plot(mac_df["cumulative_reduction"], mac_df["mac"], marker="o")
        plt.xlabel("Cumulative emission reduction from baseline (mg/m³, proxy, energy-weighted)")
        plt.ylabel("Marginal abatement cost (Δ£ / Δ emission)")
        plt.title(f"MAC along Pareto frontier (α={mac_alpha})")
        plt.grid(True)
        plt.tight_layout()
        _save_current_plot(f"mac_curve_alpha_{mac_alpha}")
        plt.show(block=False)
        plt.pause(0.001)

        plt.figure(figsize=(8, 4))
        df_cum = df_mac.copy()
        baseline_emission = float(df_mac.iloc[0][x_emission])
        baseline_money = float(df_mac.iloc[0][y_money])
        df_cum["delta_E"] = baseline_emission - df_cum[x_emission]
        df_cum["delta_money"] = baseline_money - df_cum[y_money]
        plt.plot(df_cum["delta_E"], df_cum["delta_money"], marker="o")
        plt.xlabel("Cumulative emission reduction ΔE (mg/m³, proxy, energy-weighted)")
        plt.ylabel("Cumulative money loss Δ£ (arb. £ units)")
        plt.title(f"Cumulative trade-off (α={mac_alpha})")
        plt.grid(True)
        plt.tight_layout()
        _save_current_plot(f"cumulative_tradeoff_alpha_{mac_alpha}")
        plt.show(block=False)
        plt.pause(0.001)

    # 4) Heatmaps
    money_grid = pareto_eps_df.pivot(index="alpha", columns="epsilon_nox_cap", values=y_money)
    money_grid = money_grid.sort_index().sort_index(axis=1)
    if not money_grid.empty:
        plt.figure(figsize=(8, 5))
        money_dense, alphas_dense, eps_dense = densify_heatmap(money_grid)
        money_vals = money_dense if money_dense is not None else money_grid.values
        eps_axis = eps_dense if eps_dense is not None else money_grid.columns.to_numpy()
        alpha_axis = alphas_dense if alphas_dense is not None else money_grid.index.to_numpy()
        plt.imshow(
            money_vals,
            aspect="auto",
            origin="lower",
            extent=[eps_axis.min(), eps_axis.max(), alpha_axis.min(), alpha_axis.max()],
            cmap="viridis",
            interpolation="lanczos",
        )
        plt.colorbar(label="Total economic objective (arb. £ units)")
        plt.xlabel("ε NOX cap (mg/m³, proxy)")
        plt.ylabel("α (risk tolerance)")
        plt.title("Money heatmap over α–ε grid")
        if alpha_axis.size > 1 and eps_axis.size > 1:
            Xg, Yg = np.meshgrid(eps_axis, alpha_axis)
            plt.contour(Xg, Yg, money_vals, colors="white", linewidths=0.7, alpha=0.8)
        plt.tight_layout()
        _save_current_plot("money_heatmap_alpha_epsilon")
        plt.show(block=False)
        plt.pause(0.001)

    emission_grid = pareto_eps_df.pivot(index="alpha", columns="epsilon_nox_cap", values=x_emission)
    emission_grid = emission_grid.sort_index().sort_index(axis=1)
    if not emission_grid.empty:
        plt.figure(figsize=(8, 5))
        emission_dense, alphas_dense, eps_dense = densify_heatmap(emission_grid)
        emission_vals = emission_dense if emission_dense is not None else emission_grid.values
        eps_axis = eps_dense if eps_dense is not None else emission_grid.columns.to_numpy()
        alpha_axis = alphas_dense if alphas_dense is not None else emission_grid.index.to_numpy()
        plt.imshow(
            emission_vals,
            aspect="auto",
            origin="lower",
            extent=[eps_axis.min(), eps_axis.max(), alpha_axis.min(), alpha_axis.max()],
            cmap="magma",
            interpolation="lanczos",
        )
        plt.colorbar(label="Energy-weighted NOX (mg/m³, proxy)")
        plt.xlabel("ε NOX cap (mg/m³, proxy)")
        plt.ylabel("α (risk tolerance)")
        plt.title("Emission heatmap over α–ε grid")
        if alpha_axis.size > 1 and eps_axis.size > 1:
            Xg, Yg = np.meshgrid(eps_axis, alpha_axis)
            plt.contour(Xg, Yg, emission_vals, colors="white", linewidths=0.7, alpha=0.8)
        plt.tight_layout()
        _save_current_plot("emission_heatmap_alpha_epsilon")
        plt.show(block=False)
        plt.pause(0.001)

    # 4b) Baseline-relative uplift plots (%Δ£, %ΔMWh, %ΔNOX)
    uplift_df = risk_to_cash_df.sort_values("alpha").copy()
    if not uplift_df.empty and {"alpha", "total_economic_objective", "total_mwh"}.issubset(uplift_df.columns):
        strict_alpha = float(uplift_df["alpha"].min())

        ew_nox_by_alpha = {}
        for alpha_val in uplift_df["alpha"].tolist():
            trace = _get_frontier_trace(frontier_simulation_traces, float(alpha_val))
            if trace is None or trace.empty or not {"sim_NOX", "optimal_TEY"}.issubset(trace.columns):
                ew_nox_by_alpha[float(alpha_val)] = np.nan
            else:
                ew_nox_by_alpha[float(alpha_val)] = _energy_weighted_mean(trace["sim_NOX"], trace["optimal_TEY"])
        uplift_df["energy_weighted_nox"] = uplift_df["alpha"].astype(float).map(ew_nox_by_alpha)

        # Strict baseline can be degenerate (e.g., zero MWh / NaN NOX). If so, use
        # the first alpha with finite non-zero baseline terms so uplift curves exist.
        strict_row = uplift_df[uplift_df["alpha"] == strict_alpha].iloc[0]
        strict_money = float(strict_row["total_economic_objective"])
        strict_mwh = float(strict_row["total_mwh"])
        strict_nox = float(strict_row["energy_weighted_nox"])

        valid_baseline_mask = (
            uplift_df["total_economic_objective"].astype(float).ne(0.0)
            & uplift_df["total_mwh"].astype(float).ne(0.0)
            & uplift_df["energy_weighted_nox"].astype(float).notna()
            & uplift_df["energy_weighted_nox"].astype(float).ne(0.0)
        )
        if (
            strict_money == 0.0
            or strict_mwh == 0.0
            or not np.isfinite(strict_nox)
            or strict_nox == 0.0
        ) and valid_baseline_mask.any():
            baseline_alpha = float(uplift_df.loc[valid_baseline_mask, "alpha"].astype(float).min())
            baseline_note = (
                f"strict α={strict_alpha:.3f} is degenerate; using α={baseline_alpha:.3f} as uplift baseline"
            )
        else:
            baseline_alpha = strict_alpha
            baseline_note = f"baseline = strict α={strict_alpha:.3f}"

        baseline_row = uplift_df[uplift_df["alpha"] == baseline_alpha].iloc[0]
        baseline_money = float(baseline_row["total_economic_objective"])
        baseline_mwh = float(baseline_row["total_mwh"])
        baseline_nox = float(baseline_row["energy_weighted_nox"])

        uplift_df["pct_delta_money"] = _safe_pct_delta(uplift_df["total_economic_objective"], baseline_money)
        uplift_df["pct_delta_mwh"] = _safe_pct_delta(uplift_df["total_mwh"], baseline_mwh)
        uplift_df["pct_delta_nox"] = _safe_pct_delta(uplift_df["energy_weighted_nox"], baseline_nox)

        metric_plot_specs = [
            ("pct_delta_money", "%Δ Economic objective vs baseline"),
            ("pct_delta_mwh", "%Δ MWh vs baseline"),
            ("pct_delta_nox", "%Δ Energy-weighted NOX vs baseline"),
        ]
        n_valid_series = int(
            sum(int(uplift_df[col].notna().any()) for col, _ in metric_plot_specs if col in uplift_df.columns)
        )
        if n_valid_series > 0:
            fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
            for ax, (metric_col, ylab) in zip(axes, metric_plot_specs):
                ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, label="Baseline level")
                ax.plot(
                    uplift_df["alpha"],
                    uplift_df[metric_col],
                    marker="o",
                    linewidth=1.8,
                    label="Relative change",
                )
                ax.set_ylabel(ylab)
                ax.grid(True, alpha=0.35)
                ax.legend(loc="best")

            axes[-1].set_xlabel("α (risk tolerance)")
            axes[0].set_title(f"Baseline-relative uplifts across α ({baseline_note})")
            plt.tight_layout()
            _save_current_plot("baseline_relative_uplifts")
            plt.show(block=False)
            plt.pause(0.001)
        else:
            print(
                "\nSkipping uplift plot: all uplift metrics are undefined with current baseline and traces."
            )

    # 4c) Constraint-binding decomposition over epsilon (representative alpha)
    bind_alpha = plot_alpha if plot_alpha in alphas else alphas[0]
    bind_categories = ["nox_ub", "tit_ub", "co_ub", "crc_gate", "epsilon_gate", "infeasible", "other"]
    bind_labels = {
        "nox_ub": "NOX UB binding",
        "tit_ub": "TIT UB binding",
        "co_ub": "CO UB binding",
        "crc_gate": "CRC gate binding",
        "epsilon_gate": "ε gate binding",
        "infeasible": "No feasible candidate",
        "other": "Other / tie / unclassified",
    }
    bind_colors = {
        "nox_ub": "#1b9e77",
        "tit_ub": "#d95f02",
        "co_ub": "#7570b3",
        "crc_gate": "#e7298a",
        "epsilon_gate": "#66a61e",
        "infeasible": "#666666",
        "other": "#a6761d",
    }

    rows = []
    if {"alpha", "epsilon_nox_cap"}.issubset(pareto_eps_df.columns):
        eps_values = (
            pareto_eps_df[pareto_eps_df["alpha"] == bind_alpha]["epsilon_nox_cap"]
            .astype(float)
            .sort_values()
            .tolist()
        )
        for eps_val in eps_values:
            trace = _get_trace(simulation_traces, bind_alpha, eps_val)
            if trace is None or trace.empty:
                continue
            dom = trace.get("dominant_constraint", pd.Series(["other"] * len(trace))).fillna("other").astype(str)
            dom = dom.where(dom.isin(bind_categories), "other")
            total = len(dom)
            if total <= 0:
                continue
            shares = {cat: float((dom == cat).mean()) for cat in bind_categories}
            shares.update({"alpha": float(bind_alpha), "epsilon_nox_cap": float(eps_val)})
            rows.append(shares)

    if rows:
        bind_df = pd.DataFrame(rows).sort_values("epsilon_nox_cap")
        x = bind_df["epsilon_nox_cap"].to_numpy()
        y_stack = [bind_df[c].to_numpy() for c in bind_categories]

        plt.figure(figsize=(11, 5.8))
        plt.stackplot(
            x,
            *y_stack,
            labels=[bind_labels[c] for c in bind_categories],
            colors=[bind_colors[c] for c in bind_categories],
            alpha=0.9,
        )
        plt.ylim(0.0, 1.0)
        plt.xlabel("ε NOX cap (mg/m³, proxy)")
        plt.ylabel("Share of hours where each constraint is dominant")
        plt.title(
            f"Constraint-binding decomposition along ε-frontier (α = {float(bind_alpha):.3f})"
        )
        plt.grid(True, alpha=0.25)
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, title="Dominant constraint")
        plt.tight_layout()
        _save_current_plot(f"constraint_binding_decomposition_alpha_{bind_alpha}")
        plt.show(block=False)
        plt.pause(0.001)

    # 4d) Alpha vs epsilon involvement ratio diagnostics
    ratio_metrics = [
        ("feasible_fraction", "Feasible fraction"),
        (y_money, "Economic objective"),
        ("sim_any_breach_rate", "Any breach rate"),
        ("conf_any_violation_rate", "Any conformal violation"),
        (x_emission, "Energy-weighted NOX"),
    ]
    ratio_frontier_df = risk_to_cash_df.copy()
    if x_emission not in ratio_frontier_df.columns:
        ew_nox_by_alpha_ratio = {}
        for alpha_val in ratio_frontier_df.get("alpha", pd.Series(dtype=float)).tolist():
            trace = _get_frontier_trace(frontier_simulation_traces, float(alpha_val))
            if trace is None or trace.empty or not {"sim_NOX", "optimal_TEY"}.issubset(trace.columns):
                ew_nox_by_alpha_ratio[float(alpha_val)] = np.nan
            else:
                ew_nox_by_alpha_ratio[float(alpha_val)] = _energy_weighted_mean(
                    trace["sim_NOX"], trace["optimal_TEY"]
                )
        ratio_frontier_df[x_emission] = ratio_frontier_df["alpha"].astype(float).map(ew_nox_by_alpha_ratio)

    ratio_rows = []
    if {"alpha", "epsilon_nox_cap"}.issubset(pareto_eps_df.columns):
        ratio_eps_df = pareto_eps_df.sort_values(["alpha", "epsilon_nox_cap"]).copy()
        for metric_col, metric_label in ratio_metrics:
            if metric_col not in ratio_frontier_df.columns or metric_col not in ratio_eps_df.columns:
                continue

            frontier_vals = ratio_frontier_df[metric_col].astype(float)
            if frontier_vals.notna().sum() < 2:
                continue
            alpha_span = float(frontier_vals.max() - frontier_vals.min())

            eps_spans = []
            dec_steps = []
            for _, grp in ratio_eps_df.groupby("alpha"):
                series = grp.sort_values("epsilon_nox_cap")[metric_col].astype(float)
                y = series.to_numpy()
                y = y[np.isfinite(y)]
                if len(y) < 2:
                    continue
                eps_spans.append(float(np.max(y) - np.min(y)))
                dec_steps.append(int(np.sum(np.diff(y) < 0)))

            if not eps_spans:
                continue

            mean_eps_span = float(np.mean(eps_spans))
            ratio_val = np.nan if mean_eps_span == 0.0 else float(alpha_span / mean_eps_span)
            ratio_rows.append(
                {
                    "metric": metric_col,
                    "metric_label": metric_label,
                    "alpha_span": alpha_span,
                    "mean_epsilon_span": mean_eps_span,
                    "alpha_over_epsilon_ratio": ratio_val,
                    "mean_epsilon_decrease_steps": float(np.mean(dec_steps)) if dec_steps else np.nan,
                }
            )

    if ratio_rows:
        ratio_df = pd.DataFrame(ratio_rows).sort_values("alpha_over_epsilon_ratio", ascending=False)
        print("\n Alpha vs epsilon involvement ratio diagnostics ")
        print(
            ratio_df[
                [
                    "metric",
                    "alpha_span",
                    "mean_epsilon_span",
                    "alpha_over_epsilon_ratio",
                    "mean_epsilon_decrease_steps",
                ]
            ].to_string(index=False)
        )

        plt.figure(figsize=(10.5, 4.8))
        x_labels = ratio_df["metric_label"].tolist()
        ratio_vals = ratio_df["alpha_over_epsilon_ratio"].astype(float).to_numpy()
        bars = plt.bar(x_labels, ratio_vals, color="#4c78a8", alpha=0.9)
        plt.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="Equal leverage (ratio=1)")
        plt.axhline(2.0, color="#666666", linestyle=":", linewidth=1.0, label="Alpha 2x epsilon")
        plt.ylabel("Involvement ratio = alpha span / mean epsilon span")
        plt.title("Alpha vs epsilon involvement ratio by metric")
        plt.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=15, ha="right")
        for bar, val in zip(bars, ratio_vals):
            if np.isfinite(val):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        plt.legend(loc="upper right")
        plt.tight_layout()
        _save_current_plot("alpha_vs_epsilon_involvement_ratio")
        plt.show(block=False)
        plt.pause(0.001)

    # 5) Baseline vs knee vs clean traces
    ts_alpha = mac_alpha
    df_alpha_ts = pareto_by_alpha.get(ts_alpha)
    if df_alpha_ts is not None and not df_alpha_ts.empty:
        df_alpha_ts = df_alpha_ts.dropna(subset=[x_emission, y_money, "epsilon_nox_cap"])
        if not df_alpha_ts.empty:
            baseline_eps = float(df_alpha_ts["epsilon_nox_cap"].max())
            clean_eps = float(df_alpha_ts.loc[df_alpha_ts[x_emission].idxmin(), "epsilon_nox_cap"])
            knee_row = compute_knee_point(df_alpha_ts, x_emission, y_money)
            knee_eps = float(knee_row.iloc[0]["epsilon_nox_cap"]) if knee_row is not None and not knee_row.empty else baseline_eps

            baseline_trace = _get_trace(simulation_traces, ts_alpha, baseline_eps)
            knee_trace = _get_trace(simulation_traces, ts_alpha, knee_eps)
            clean_trace = _get_trace(simulation_traces, ts_alpha, clean_eps)

            if baseline_trace is not None and knee_trace is not None and clean_trace is not None:
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                labels = [("Baseline (ε max)", baseline_trace), ("Knee", knee_trace), ("Clean (min emission)", clean_trace)]
                for label, trace in labels:
                    axes[0].plot(trace["hour_idx"], trace["optimal_TEY"], label=label)
                    axes[1].plot(trace["hour_idx"], trace["sim_NOX"], label=label)
                    axes[2].plot(trace["hour_idx"], trace["sim_CO"], label=label)
                    axes[3].plot(trace["hour_idx"], trace["sim_nox_breach"], label=label)

                axes[0].set_ylabel("TEY")
                axes[1].set_ylabel("NOX (mg/m³, proxy)")
                axes[2].set_ylabel("CO (mg/m³, proxy)")
                axes[3].set_ylabel("NOX breach (0/1)")
                axes[3].set_xlabel("hour_idx")
                axes[0].set_title(f"Policy traces at α={ts_alpha}: baseline vs knee vs clean")
                for ax in axes:
                    ax.grid(True)
                    ax.legend()
                plt.tight_layout()
                _save_current_plot(f"policy_traces_alpha_{ts_alpha}")
                plt.show(block=False)
                plt.pause(0.001)
            else:
                print(f"\nTime-series traces missing for α={ts_alpha}; skipping baseline/knee/clean plot.")

    if plt.get_fignums():
        print("Plots generated; close the figure windows to exit.")
        plt.show()

    if save_plots:
        print(f"Saved {saved_plot_counter} plot file(s) to: {plots_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
