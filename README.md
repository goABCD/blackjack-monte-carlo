# Blackjack Experiment: Exact Hit/Stand + Monte Carlo

Macau-style blackjack experiment under a **locked** rule set:

- continuous shuffle / infinite-deck draw probabilities (fixed \(4/52\), \(16/52\) for tens)
- dealer **stands on soft 17 (S17)**
- no-hole-card timing for the player decision model
- player actions studied: **hit vs stand** on **hard totals 12–16**

The repo pairs an **exact conditional-probability recursion** with a **Monte Carlo** check.

## Question

How does the dealer’s bust probability depend on the upcard, and when should a player hit or stand on hard 12–16?

## Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd src
python exact.py
python plot_results.py
python monte_carlo.py --n-dealer 200000 --n-ev 80000
```

## Outputs

| Path | Content |
|---|---|
| `figures/dealer_bust.png` | Exact dealer bust % by upcard |
| `figures/delta_E_heatmap.png` | ΔE = E(hit)−E(stand) heatmap |
| `data/mc_dealer_bust.csv` | Exact vs MC bust rates |
| `data/mc_ev_focus.csv` | Focus cells with MC EV |

## Method (short)

1. **Dealer chain**: state `(total_low, has_ace)` until stand (≥17) or bust; infinite-deck transitions.
2. **Stand EV** on hard ≤16 collapses to \(2B_u - 1\) (win only if dealer busts).
3. **Hit EV**: one card, then continue optimally (hard totals).
4. **Monte Carlo**: resample the same probabilities; compare bust rates and a few boundary cells.

## Snapshot results (exact)

Dealer bust rises from about **11.5%** (upcard A) to about **42.3%** (upcard 6), then falls toward ten-value upcards. Hard 12–16 therefore tends to **stand vs weak upcards 2–6** and **hit vs strong upcards 7–A**, with a few near-zero ΔE boundary cells (e.g. 12 vs 4, 16 vs 10) that MC is meant to stress-test.

## Limitations

- Soft totals, doubles, splits, and surrender are out of scope.
- Infinite-deck / CSM is a within-shoe approximation of continuous shuffle.
- Natural blackjack edge cases follow the ENHC-style treatment coded in `exact.py`.

## Stack

Python · NumPy · Matplotlib
