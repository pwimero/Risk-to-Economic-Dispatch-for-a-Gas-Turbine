from __future__ import annotations

import argparse
import os

from shared_config import RISK_TO_CASH_FRONTIER_CSV_PATH, RISK_TO_CASH_RESULTS_DIR, TARGETS

# In-code defaults for quick local reruns.
SAVE_PLOT = True
DEFAULT_OUTPUT_PATH = os.path.join(
    RISK_TO_CASH_RESULTS_DIR,
    "plots",
    "target_risk_to_cash_frontier.png",
)

TARGET_COLORS = {
    "NOX": "#D55E00",
    "TIT": "#0072B2",
    "CO": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target-wise risk-to-cash frontier charts from the saved frontier CSV."
    )
    parser.add_argument(
        "--csv-path",
        default=RISK_TO_CASH_FRONTIER_CSV_PATH,
        help="Path to the saved risk-to-cash frontier CSV.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path when saving is enabled.",
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save",
        dest="save_plot_cli",
        action="store_true",
        help="Override in-code SAVE_PLOT to True.",
    )
    save_group.add_argument(
        "--no-save",
        dest="save_plot_cli",
        action="store_false",
        help="Override in-code SAVE_PLOT to False.",
    )
    parser.set_defaults(save_plot_cli=None)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after rendering.",
    )
    return parser.parse_args()


def _sim_rate_col(target: str) -> str:
    return f"sim_{target.lower()}_breach_rate"


def _conf_rate_col(target: str) -> str:
    return f"conf_{target.lower()}_violation_rate"


def _blend_with_white(color: str, amount: float = 0.35) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(color)
    return tuple((1.0 - amount) * ch + amount for ch in rgb)


def _money_formatter(value: float, _pos: int) -> str:
    return f"£{value / 1_000_000:.1f}M"


def load_frontier(csv_path: str):
    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Risk-to-cash frontier CSV not found. Run RiskToCashFrontier.py first.\n"
            f"Missing file: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    required_cols = {
        "alpha",
        "total_economic_objective",
        "feasible_fraction",
        "sim_any_breach_rate",
        "conf_any_violation_rate",
    }
    for target in TARGETS:
        required_cols.add(_sim_rate_col(target))
        required_cols.add(_conf_rate_col(target))

    missing = [col for col in sorted(required_cols) if col not in df.columns]
    if missing:
        raise ValueError(f"Frontier CSV is missing required columns: {missing}")

    df = df.sort_values("alpha").reset_index(drop=True)
    return df


def _decorate_target_axis(ax, df, target: str, money_formatter, percent_formatter) -> None:
    color = TARGET_COLORS.get(target, "#333333")
    conf_color = _blend_with_white(color, amount=0.40)
    cash = df["total_economic_objective"].astype(float)
    sim_rate = df[_sim_rate_col(target)].astype(float)
    conf_rate = df[_conf_rate_col(target)].astype(float)

    ax.plot(
        sim_rate,
        cash,
        color=color,
        linewidth=2.6,
        marker="o",
        markersize=7,
        label="Simulated breach rate",
    )
    ax.plot(
        conf_rate,
        cash,
        color=conf_color,
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=6,
        label="Conformal violation rate",
    )

    for row in df.itertuples(index=False):
        ax.annotate(
            f"α={row.alpha:.2f}",
            (float(getattr(row, _conf_rate_col(target))), float(row.total_economic_objective)),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color="#2f2f2f",
        )

    x_max = max(float(sim_rate.max()), float(conf_rate.max()))
    ax.set_xlim(0.0, x_max * 1.18 if x_max > 0 else 0.05)
    ax.set_title(f"{target} Target Frontier", fontsize=12, fontweight="bold")
    ax.set_xlabel("Risk rate")
    ax.set_ylabel("Total economic objective")
    ax.xaxis.set_major_formatter(percent_formatter)
    ax.yaxis.set_major_formatter(money_formatter)
    ax.grid(True, alpha=0.22)


def build_figure(df):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    percent_formatter = PercentFormatter(1.0, decimals=0)
    money_formatter = FuncFormatter(_money_formatter)

    for ax, target in zip(axes.flat[:3], TARGETS):
        _decorate_target_axis(ax, df, target, money_formatter, percent_formatter)

    context_ax = axes.flat[3]
    alphas = df["alpha"].astype(float)
    feasible_fraction = df["feasible_fraction"].astype(float)
    cash = df["total_economic_objective"].astype(float)

    bar_colors = ["#d7ebd0" if frac >= 0.75 else "#f5e7b8" if frac >= 0.50 else "#f2c4b4" for frac in feasible_fraction]
    context_ax.bar(alphas, feasible_fraction, width=0.018, color=bar_colors, edgecolor="#748c69", alpha=0.9)
    context_ax.set_title("Frontier Context by Risk Tolerance", fontsize=12, fontweight="bold")
    context_ax.set_xlabel("Alpha risk budget")
    context_ax.set_ylabel("Feasible fraction")
    context_ax.yaxis.set_major_formatter(percent_formatter)
    context_ax.set_ylim(0.0, 1.05)
    context_ax.set_xlim(float(alphas.min()) - 0.015, float(alphas.max()) + 0.015)
    context_ax.set_xticks(alphas.tolist())
    context_ax.grid(True, axis="y", alpha=0.22)

    cash_ax = context_ax.twinx()
    cash_ax.plot(
        alphas,
        cash,
        color="#1f1f1f",
        marker="o",
        linewidth=2.4,
        markersize=6,
        label="Economic objective",
    )
    cash_ax.set_ylabel("Total economic objective")
    cash_ax.yaxis.set_major_formatter(money_formatter)
    cash_ax.spines["top"].set_visible(False)

    for alpha_val, feasible_val, cash_val in zip(alphas, feasible_fraction, cash):
        context_ax.annotate(
            f"{feasible_val:.0%}",
            (float(alpha_val), float(feasible_val)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#364035",
        )
        cash_ax.annotate(
            f"£{cash_val / 1_000_000:.1f}M",
            (float(alpha_val), float(cash_val)),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#1f1f1f",
        )

    style_handles = [
        plt.Line2D([], [], color="#2b2b2b", marker="o", linewidth=2.6, label="Simulated breach rate"),
        plt.Line2D([], [], color="#6f6f6f", marker="s", linewidth=2.2, linestyle="--", label="Conformal violation rate"),
    ]

    fig.legend(
        style_handles,
        [h.get_label() for h in style_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle("Target-wise Risk-to-Cash Frontier", fontsize=16, fontweight="bold", y=1.03)
    return fig


def main() -> int:
    args = parse_args()
    save_plot = bool(SAVE_PLOT) if args.save_plot_cli is None else bool(args.save_plot_cli)

    df = load_frontier(args.csv_path)
    fig = build_figure(df)

    if save_plot:
        out_dir = os.path.dirname(os.path.abspath(args.output_path))
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
        print(f"Saved plot: {args.output_path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
