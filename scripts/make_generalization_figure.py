"""Cross-sector generalization figure (report section 5.6).

Reads reports/results/generalization_metrics.json and draws, per sector, the
combined objective J for each policy under two arms: zero-shot (baseline policies
trained on the general profile only) vs domain-randomized (retrained with the sector
sampled each episode). J is compared only *within* a sector (V_0 differs across
sectors), so each sector is its own panel with its own y-scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("reports/results/generalization_metrics.json")
OUT = Path("reports/figures/generalization.png")
SECTORS = ["general", "retail", "manufacturing", "services"]
POLICIES = ["rule_based", "single_agent_ppo", "ippo", "mappo"]
LABELS = ["Rule-based", "Single-agent", "IPPO", "MAPPO"]


def main() -> int:
    data = json.loads(RESULTS.read_text())
    zs, dr = data["zero_shot"], data["domain_randomized"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    x = np.arange(len(POLICIES))
    width = 0.38
    for ax, sector in zip(axes, SECTORS, strict=True):
        zvals = [zs[sector][p]["combined_objective"] for p in POLICIES]
        dvals = [dr[sector][p]["combined_objective"] for p in POLICIES]
        ax.bar(x - width / 2, zvals, width, label="Zero-shot", color="#4C72B0")
        ax.bar(
            x + width / 2, dvals, width, label="Domain-randomized",
            color="#DD8452", hatch="//", edgecolor="white",
        )
        ax.set_title(sector.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if sector == SECTORS[0]:
            ax.set_ylabel("Combined objective J")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Cross-sector generalization: zero-shot vs domain-randomized (mean over 5 seeds x 100 episodes)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
