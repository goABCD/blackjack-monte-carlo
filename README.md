# Blackjack Monte Carlo · Math IA Lab

Macau-style blackjack under a **locked** rule set (CSM / infinite-deck, S17, hard hit/stand 12–16): exact recursion + Monte Carlo + interactive lab.

---

## Quick start (Windows)

1. Download / unzip this repo  
2. Double-click **`start_lab.bat`**  
3. Browser opens **http://localhost:8501**

Or in a terminal:

```bat
cd blackjack-monte-carlo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

> Do **not** open `.py` with “Python default”. This is a Streamlit web app.

### Lab tabs

| Tab | What it does |
|---|---|
| Practice | Deal / Hit / Stand table + IA tip |
| Custom strategy | Edit hard 12–16 H/S matrix |
| Batch Monte Carlo | Up to 1e6 hands; bust curve, EV bars, ΔE heatmap |
| Exact model | Dealer terminal table + decision grid |

---

## Question

How does the dealer’s bust probability depend on the upcard, and when should a player hit or stand on hard 12–16?

## Folder map

| Path | Role |
|---|---|
| `app/streamlit_app.py` | Interactive Math IA panel |
| `src/exact.py` | Exact dealer dist + optimal hit/stand |
| `src/monte_carlo.py` | Monte Carlo validation CLI |
| `src/plot_results.py` | Save static figures |
| `data/` | CSV outputs |
| `figures/` | Pre-rendered charts |
| `start_lab.bat` | One-click Windows launcher |

## CLI (optional)

```bash
pip install -r requirements.txt
cd src
python exact.py
python plot_results.py
python monte_carlo.py --n-dealer 200000 --n-ev 80000
```

## Method (short)

1. **Dealer chain**: state `(total_low, has_ace)` until stand (≥17) or bust.  
2. **Stand EV** on hard ≤16 collapses to `2B_u − 1`.  
3. **Hit EV**: draw one card, then continue optimally (hard totals).  
4. **Monte Carlo**: resample the same probabilities; compare bust rates and strategy EV.

## Snapshot results (exact)

Dealer bust rises from about **11.5%** (upcard A) to about **42.3%** (upcard 6). Hard 12–16 tends to **stand vs 2–6** and **hit vs 7–A**, with near-zero ΔE boundary cells (e.g. 12 vs 4, 16 vs 10).

## Limitations

- Soft totals, doubles, splits, surrender out of scope.  
- Infinite-deck / CSM approximates continuous shuffle.  
- Lab payout is 1:1 (no 3:2 blackjack).

## Stack

Python · NumPy · Matplotlib · Pandas · Streamlit
