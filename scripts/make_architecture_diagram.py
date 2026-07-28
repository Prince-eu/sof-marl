"""Generate the SOF-MARL system architecture diagram -> reports/figures/architecture.png.

A static, reproducible rendering of the design in docs/ARCHITECTURE.md: four agents
acting on one shared treasury environment (a calibrated simulation), the real SBA
7(a) credit model feeding the credit-risk agent, and the reward/critic structure
that distinguishes IPPO (independent) from MAPPO (coordinated). The agent order here
is illustrative -- the credit-risk agent is drawn next to its real-data source for
clarity, not to imply an ordering in code.

Regenerate with:
    python scripts/make_architecture_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Print-friendly muted palette.
ENV_FILL, ENV_EDGE = "#dce9f5", "#3b6ea5"
REAL_FILL, REAL_EDGE = "#d7ead0", "#4a7a3a"
AGENT_FILL, AGENT_EDGE = "#eeeeee", "#555555"
CREDIT_AGENT_FILL = "#e5efdd"  # subtle green tint links it to its data source
REWARD_FILL, REWARD_EDGE = "#f5ecd7", "#a5843b"


def box(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    title: str,
    body: str = "",
    fill: str = "#ffffff",
    edge: str = "#333333",
    lw: float = 1.4,
    title_size: int = 11,
    body_size: int = 8,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=lw,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    cx = (x0 + x1) / 2
    if body:
        ax.text(cx, y1 - 0.34, title, ha="center", va="top", fontsize=title_size, fontweight="bold")
        # push the body clear of multi-line titles
        title_lines = title.count("\n") + 1
        body_y = y1 - 0.34 - 0.30 * title_lines - 0.16
        ax.text(cx, body_y, body, ha="center", va="top", fontsize=body_size, linespacing=1.35)
    else:
        ax.text(cx, (y0 + y1) / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold")


def arrow(
    ax: plt.Axes,
    xy_from: tuple[float, float],
    xy_to: tuple[float, float],
    style: str = "-|>",
    color: str = "#333333",
    lw: float = 1.3,
    ls: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            xy_from,
            xy_to,
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> int:
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11.4)
    ax.axis("off")

    ax.text(8, 11.05, "SOF-MARL system architecture", ha="center", fontsize=15, fontweight="bold")
    ax.text(
        8,
        10.62,
        "Four agents over one shared cash pool; the credit-risk function is grounded in a real SBA 7(a) model.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    # Shared environment.
    box(
        ax,
        0.5,
        8.5,
        15.5,
        10.3,
        "Shared treasury environment  —  calibrated simulation (52-week episode)",
        "State: cash pool | AR aging | AP schedule | revolving credit + term debt | seasonal revenue/expenses | reserve | invested | pending credit exposures\n"
        "Dynamics close the books every week (cash_{t+1} = cash_t + inflows − outflows), verified by unit tests.",
        fill=ENV_FILL,
        edge=ENV_EDGE,
        title_size=12,
    )

    # Real credit model (far left, emphasized).
    box(
        ax,
        0.4,
        4.3,
        2.5,
        6.7,
        "SBA 7(a)\ncredit model",
        "REAL public data\nXGBoost, isotonic-\ncalibrated\n(ROC-AUC 0.94)",
        fill=REAL_FILL,
        edge=REAL_EDGE,
        lw=2.2,
        title_size=10,
        body_size=7.5,
    )

    # Obs/action legend note in the clear space above the credit model.
    ax.text(
        1.45,
        7.9,
        "Each agent observes its\nown state slice and acts\non the shared cash pool ( ↕ )",
        ha="center",
        va="top",
        fontsize=7.5,
        color=ENV_EDGE,
        linespacing=1.4,
    )

    # Four agents (credit-risk drawn next to its real-data source; order illustrative).
    agents = [
        (3.4, 6.0, "Credit-risk", "deny / approve /\napprove-with-premium", CREDIT_AGENT_FILL),
        (6.4, 9.0, "Liquidity", "buffer, draw/repay,\ndefer AP, accel. AR", AGENT_FILL),
        (9.4, 12.0, "Expenditure", "spend now vs. defer,\nessential bias", AGENT_FILL),
        (12.4, 15.0, "Capital alloc.", "debt / reserve /\ninvest (simplex)", AGENT_FILL),
    ]
    for x0, x1, title, action, fill in agents:
        box(ax, x0, 4.3, x1, 6.7, title, action, fill=fill, edge=AGENT_EDGE, title_size=11, body_size=7.5)
        cx = (x0 + x1) / 2
        # env <-> agent (observation down, action up)
        arrow(ax, (cx, 8.5), (cx, 6.7), style="<->", color=ENV_EDGE, lw=1.3)
        # agent -> learning signal
        arrow(ax, (cx, 4.3), (cx, 3.3), style="-|>", color="#777777", lw=1.1)

    # Real model -> credit agent, with the label above the arrow in the clear gap.
    arrow(ax, (2.5, 5.4), (3.4, 5.4), style="-|>", color=REAL_EDGE, lw=1.8)
    ax.text(2.95, 6.15, "P(default)\nper exposure", ha="center", va="bottom", fontsize=7, color=REAL_EDGE, linespacing=1.3)

    # Learning-signal band.
    box(ax, 0.5, 1.5, 15.5, 3.3, "", fill=REWARD_FILL, edge=REWARD_EDGE)
    ax.text(8, 3.02, "Learning signal (PPO)", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(
        3.4,
        2.42,
        "Per-agent reward R_i\n(function-specific)",
        ha="center",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
    )
    ax.text(
        8,
        2.42,
        "+  λ · R_shared\nR_shared = α·Δ(firm value) − β·(solvency breach)",
        ha="center",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
    )
    ax.text(
        12.7,
        2.42,
        "Critic:\ncentralized over joint obs (MAPPO)\nvs. independent per-agent (IPPO)",
        ha="center",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
    )

    ax.text(
        8,
        0.75,
        "MAPPO (coordinated, Exhibit C.2 design): λ > 0 with a centralized critic.      "
        "IPPO (baseline): λ = 0 with independent critics — isolates the value of coordination.",
        ha="center",
        fontsize=8.5,
        color="#333333",
    )

    out_path = Path("reports/figures/architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
