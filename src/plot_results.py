"""Save dealer bust chart and ΔE heatmap from exact model."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exact import UPCARDS, E_hit, E_stand, dealer_table

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def main() -> None:
    rows = dealer_table()
    ups = [r["upcard"] for r in rows]
    bust = [100 * r["bust"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ups, bust, "-o", color="#1f4e79", lw=2)
    for x, y in zip(ups, bust):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Dealer upcard")
    ax.set_ylabel("Bust probability (%)")
    ax.set_title("Dealer bust rate by upcard (exact, infinite-deck S17)")
    ax.set_ylim(0, 50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p1 = FIG / "dealer_bust.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    Hs = list(range(12, 17))
    M = np.array([[E_hit(H, u) - E_stand(H, u) for u in UPCARDS] for H in Hs])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(M, cmap="RdBu", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(range(len(UPCARDS)))
    ax.set_xticklabels(UPCARDS)
    ax.set_yticks(range(len(Hs)))
    ax.set_yticklabels(Hs)
    ax.set_xlabel("Dealer upcard")
    ax.set_ylabel("Player hard total")
    ax.set_title("ΔE = E(hit) − E(stand); blue → hit better, red → stand better")
    for i in range(len(Hs)):
        for j in range(len(UPCARDS)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p2 = FIG / "delta_E_heatmap.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    print("saved", p1)
    print("saved", p2)


if __name__ == "__main__":
    main()
