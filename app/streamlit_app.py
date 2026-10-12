# -*- coding: utf-8 -*-
"""Macau-style Blackjack · Math IA Lab (Streamlit)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exact import (  # noqa: E402
    E_hit,
    E_stand,
    UPCARDS,
    VAL,
    dealer_from_upcard,
    dealer_table,
    optimal_action,
)
from monte_carlo import draw, mc_dealer_bust, play_dealer  # noqa: E402


def setup_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.sans-serif"] = candidates
    matplotlib.rcParams["axes.unicode_minus"] = False


setup_fonts()

st.set_page_config(
    page_title="Blackjack Math IA Lab",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "balance": 10000,
        "bet": 25,
        "stats": {"played": 0, "wins": 0, "losses": 0, "pushes": 0},
        "hand": None,
        "rng_seed": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def rng() -> np.random.Generator:
    st.session_state.rng_seed += 1
    return np.random.default_rng(st.session_state.rng_seed)


def hand_total(cards: list[str]) -> tuple[int, bool]:
    low = sum(VAL[c] for c in cards)
    aces = sum(1 for c in cards if c == "A")
    soft = bool(aces and low + 10 <= 21)
    return (low + 10 if soft else low), soft


def render_cards(cards: list[str]) -> str:
    return "  ".join(f"`{c}`" for c in cards) if cards else "—"


def default_strategy_matrix() -> pd.DataFrame:
    """Columns: Hard, then upcards. Integer Hard column (no fragile index)."""
    rows = []
    for H in range(12, 17):
        row = {"Hard": H}
        for u in UPCARDS:
            row[u] = optimal_action(H, u)
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_strategy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Hard" not in out.columns:
        # recover from old index-based editor state
        out = out.reset_index()
        if "Hard" not in out.columns:
            first = out.columns[0]
            out = out.rename(columns={first: "Hard"})
    out["Hard"] = pd.to_numeric(out["Hard"], errors="coerce").fillna(12).astype(int)
    for u in UPCARDS:
        if u not in out.columns:
            out[u] = "S"
        out[u] = (
            out[u]
            .astype(str)
            .str.upper()
            .str.strip()
            .map(lambda x: "S" if str(x).startswith("S") else "H")
        )
    cols = ["Hard"] + UPCARDS
    out = out[cols].drop_duplicates(subset=["Hard"]).sort_values("Hard")
    # ensure all hard totals present
    base = default_strategy_matrix()
    merged = base[["Hard"]].merge(out, on="Hard", how="left")
    for u in UPCARDS:
        merged[u] = merged[u].fillna(base.set_index("Hard")[u]).astype(str)
        merged[u] = merged[u].map(lambda x: "S" if str(x).upper().startswith("S") else "H")
    return merged.reset_index(drop=True)


def strategy_action(matrix: pd.DataFrame, H: int, u: str) -> str:
    if H < 12:
        return "H"
    if H >= 17:
        return "S"
    m = normalize_strategy(matrix)
    row = m.loc[m["Hard"] == H]
    if row.empty or u not in m.columns:
        return optimal_action(min(max(H, 12), 16), u)
    val = str(row.iloc[0][u]).upper().strip()
    return "S" if val.startswith("S") else "H"


def finish_dealer(up: str, r: np.random.Generator) -> tuple[list[str], object]:
    hole = draw(r)
    cards = [up, hole]
    low = sum(VAL[c] for c in cards)
    has_ace = any(c == "A" for c in cards)
    while True:
        T = low + (10 if has_ace and low + 10 <= 21 else 0)
        if T > 21:
            return cards, "bust"
        if T >= 17:
            return cards, T
        c = draw(r)
        cards.append(c)
        low += VAL[c]
        has_ace = has_ace or (c == "A")


def settle(player_total: int, dealer_outcome, bet: int) -> tuple[int, str]:
    if player_total > 21:
        return -bet, "Player bust — dealer wins"
    if dealer_outcome == "bust":
        return bet, "Dealer bust — player wins"
    d = int(dealer_outcome)
    if player_total > d:
        return bet, f"Player {player_total} > dealer {d}"
    if player_total < d:
        return -bet, f"Player {player_total} < dealer {d}"
    return 0, f"Push {player_total}"


def apply_pnl(pnl: int) -> None:
    bet = st.session_state.hand["bet"]
    st.session_state.balance += bet + pnl
    st.session_state.stats["played"] += 1
    if pnl > 0:
        st.session_state.stats["wins"] += 1
    elif pnl < 0:
        st.session_state.stats["losses"] += 1
    else:
        st.session_state.stats["pushes"] += 1


def mc_custom_strategy(matrix: pd.DataFrame, n_hands: int, seed: int, progress_cb=None) -> dict:
    r = np.random.default_rng(seed)
    matrix = normalize_strategy(matrix)
    pay = 0.0
    wins = losses = pushes = busts_p = 0
    chunk = max(n_hands // 100, 1)
    for i in range(n_hands):
        cards = [draw(r), draw(r)]
        up = draw(r)
        while True:
            total, soft = hand_total(cards)
            if total > 21:
                pay -= 1
                losses += 1
                busts_p += 1
                break
            if soft:
                act = "S" if total >= 17 else "H"
            else:
                act = strategy_action(matrix, total, up)
            if act == "S":
                d = play_dealer(up, r)
                if d == "bust" or (isinstance(d, int) and d < total):
                    pay += 1
                    wins += 1
                elif isinstance(d, int) and d > total:
                    pay -= 1
                    losses += 1
                else:
                    pushes += 1
                break
            cards.append(draw(r))
        if progress_cb and (i + 1) % chunk == 0:
            progress_cb((i + 1) / n_hands)
    if progress_cb:
        progress_cb(1.0)
    return {
        "n": n_hands,
        "ev": pay / n_hands,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "player_busts": busts_p,
        "win_rate": wins / n_hands,
        "name": "",
    }


def fig_dealer_bust(exact_rows, mc_map=None):
    ups = [row["upcard"] for row in exact_rows]
    exact = [100 * row["bust"] for row in exact_rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ups, exact, "-o", color="#1f4e79", lw=2, label="Exact bust %")
    if mc_map:
        ax.plot(ups, [100 * mc_map[u] for u in ups], "--s", color="#c0392b", lw=1.6, label="Monte Carlo bust %")
    ax.set_xlabel("Dealer upcard")
    ax.set_ylabel("Bust probability (%)")
    ax.set_title("Dealer bust rate by upcard (IA core chart)")
    ax.set_ylim(0, 50)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def fig_delta_heatmap():
    Hs = list(range(12, 17))
    M = np.array([[E_hit(H, u) - E_stand(H, u) for u in UPCARDS] for H in Hs])
    fig, ax = plt.subplots(figsize=(10, 4.0))
    im = ax.imshow(M, cmap="RdBu", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(range(len(UPCARDS)))
    ax.set_xticklabels(UPCARDS)
    ax.set_yticks(range(len(Hs)))
    ax.set_yticklabels(Hs)
    ax.set_xlabel("Dealer upcard")
    ax.set_ylabel("Player hard total")
    ax.set_title("Delta E = E(hit) - E(stand) (IA core chart)")
    for i in range(len(Hs)):
        for j in range(len(UPCARDS)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_strategy_grid(custom: pd.DataFrame, base: pd.DataFrame):
    c = normalize_strategy(custom).set_index("Hard")[UPCARDS]
    b = normalize_strategy(base).set_index("Hard")[UPCARDS]
    diff = (b.astype(str).values != c.astype(str).values).astype(float)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.imshow(diff, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(UPCARDS)))
    ax.set_xticklabels(UPCARDS)
    ax.set_yticks(range(len(c.index)))
    ax.set_yticklabels(list(c.index))
    ax.set_title("Custom strategy (red = differs from exact optimal)")
    ax.set_xlabel("Dealer upcard")
    ax.set_ylabel("Hard total")
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            ax.text(j, i, str(c.values[i, j]), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def fig_ev_bars(rows: list[dict]):
    labels = [r["name"] for r in rows]
    evs = [r["ev"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1f4e79" if e >= 0 else "#c0392b" for e in evs]
    ax.bar(labels, evs, color=colors)
    ax.axhline(0, color="gray", lw=1)
    ax.set_ylabel("EV per unit bet")
    ax.set_title("Strategy EV comparison (Monte Carlo)")
    for i, v in enumerate(evs):
        ax.text(i, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    return fig


init_state()

st.title("🃏 Blackjack · Math IA Lab")
st.caption(
    "IB Math AA HL Exploration tool under locked Macau-style rules "
    "(CSM / infinite-deck, S17, hard hit/stand). Practice table + editable strategy + large-N Monte Carlo + IA charts."
)

with st.sidebar:
    st.header("World Lock")
    st.markdown(
        """
**L1** six-deck composition ratios  
**L2** CSM = fixed with-replacement probs  
**L3** dealer S17  
**L4** player acts first  
**L5** Hit / Stand only  
**L6** +1 / −1 / 0  
"""
    )
    st.divider()
    st.session_state.bet = st.select_slider(
        "Practice bet", options=[5, 10, 25, 50, 100], value=int(st.session_state.bet)
    )
    if st.button("Reset bankroll & stats", width="stretch"):
        st.session_state.balance = 10000
        st.session_state.stats = {"played": 0, "wins": 0, "losses": 0, "pushes": 0}
        st.session_state.hand = None
        st.rerun()

if "strategy_df" not in st.session_state:
    st.session_state.strategy_df = default_strategy_matrix()
else:
    st.session_state.strategy_df = normalize_strategy(st.session_state.strategy_df)

tab_play, tab_strategy, tab_batch, tab_theory = st.tabs(
    ["🎮 Practice", "🧩 Custom strategy", "📈 Batch Monte Carlo", "📐 Exact model / IA charts"]
)

with tab_play:
    s = st.session_state.stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Balance", f"{st.session_state.balance:,}")
    m2.metric("Hands", s["played"])
    m3.metric("W / L / P", f"{s['wins']} / {s['losses']} / {s['pushes']}")
    m4.metric("Win rate", f"{(100 * s['wins'] / s['played']) if s['played'] else 0:.1f}%")
    st.info("Practice builds intuition. For IA numbers use **Batch Monte Carlo** and **Exact model**.")

    hand = st.session_state.hand
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.subheader("Table")
        if hand is None:
            st.write("Click **Deal** to start.")
        else:
            if hand["done"]:
                st.markdown(f"**Dealer**  {render_cards(hand['dealer_cards'])}  → `{hand['dealer_final']}`")
            else:
                st.markdown(f"**Dealer**  `{hand['dealer_up']}`  🂠")
            pt, soft = hand_total(hand["player"])
            st.markdown(f"**You**  {render_cards(hand['player'])}")
            st.caption(f"Total {pt} ({'soft' if soft else 'hard'}) · bet {hand['bet']}")
            if hand["done"]:
                st.success(hand["message"])

        b1, b2, b3, b4 = st.columns(4)
        can_deal = hand is None or hand.get("done")
        if b1.button("Deal", type="primary", width="stretch", disabled=not can_deal):
            if st.session_state.balance < st.session_state.bet:
                st.error("Not enough balance")
            else:
                r = rng()
                st.session_state.balance -= st.session_state.bet
                st.session_state.hand = {
                    "player": [draw(r), draw(r)],
                    "dealer_up": draw(r),
                    "dealer_cards": [],
                    "dealer_final": None,
                    "bet": st.session_state.bet,
                    "done": False,
                    "message": "",
                }
                st.rerun()
        if hand and not hand["done"]:
            if b2.button("Hit", width="stretch"):
                hand["player"].append(draw(rng()))
                pt, _ = hand_total(hand["player"])
                if pt > 21:
                    hand["done"] = True
                    hand["dealer_cards"] = [hand["dealer_up"]]
                    hand["dealer_final"] = "(no draw)"
                    hand["message"] = "Player bust — dealer wins"
                    apply_pnl(-hand["bet"])
                st.rerun()
            if b3.button("Stand", width="stretch"):
                cards, final = finish_dealer(hand["dealer_up"], rng())
                hand["dealer_cards"] = cards
                hand["dealer_final"] = final
                pt, _ = hand_total(hand["player"])
                pnl, msg = settle(pt, final, hand["bet"])
                hand["done"] = True
                hand["message"] = msg
                apply_pnl(pnl)
                st.rerun()
            if b4.button("IA tip", width="stretch"):
                pt, soft = hand_total(hand["player"])
                if soft:
                    st.warning("Soft hands are outside the IA main table; rule of thumb: stand ≥17 else hit.")
                elif 12 <= pt <= 16:
                    act = optimal_action(pt, hand["dealer_up"])
                    dE = E_hit(pt, hand["dealer_up"]) - E_stand(pt, hand["dealer_up"])
                    st.info(f"Exact optimal: **{'Stand' if act == 'S' else 'Hit'}** (ΔE={dE:+.3f})")
                else:
                    st.info("Hard <12 → usually hit; ≥17 → usually stand.")
        if hand and hand["done"]:
            if st.button("Next hand", width="stretch"):
                st.session_state.hand = None
                st.rerun()
    with col_b:
        st.subheader("Rules")
        st.markdown(
            """
| Item | Lab setting |
|---|---|
| Draws | CSM fixed probs |
| Dealer | S17 |
| Payout | 1:1 (simplified) |
| IA focus | bust% · hard 12–16 |
"""
        )

with tab_strategy:
    st.subheader("Hard 12–16 Hit/Stand matrix (editable)")
    st.caption("Cells: `H` = hit, `S` = stand. Default = exact optimal. Used by Batch Monte Carlo.")
    base = default_strategy_matrix()
    edited = st.data_editor(
        st.session_state.strategy_df,
        width="stretch",
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Hard": st.column_config.NumberColumn("Hard", disabled=True, min_value=12, max_value=16),
            **{u: st.column_config.SelectboxColumn(u, options=["H", "S"], required=True) for u in UPCARDS},
        },
        key="strategy_editor_v2",
    )
    cleaned = normalize_strategy(edited)
    st.session_state.strategy_df = cleaned

    c1, c2, c3 = st.columns(3)
    if c1.button("Reset to exact optimal", width="stretch"):
        st.session_state.strategy_df = default_strategy_matrix()
        st.rerun()
    if c2.button("All Stand (S)", width="stretch"):
        df = default_strategy_matrix()
        for u in UPCARDS:
            df[u] = "S"
        st.session_state.strategy_df = df
        st.rerun()
    if c3.button("All Hit (H)", width="stretch"):
        df = default_strategy_matrix()
        for u in UPCARDS:
            df[u] = "H"
        st.session_state.strategy_df = df
        st.rerun()

    st.pyplot(fig_strategy_grid(cleaned, base), clear_figure=True)
    bmat = normalize_strategy(base).set_index("Hard")[UPCARDS]
    cmat = normalize_strategy(cleaned).set_index("Hard")[UPCARDS]
    diffs = int((bmat.astype(str).values != cmat.astype(str).values).sum())
    st.write(f"Cells differing from exact optimal: **{diffs}** / {bmat.size}")

with tab_batch:
    st.subheader("Large-N Monte Carlo")
    st.markdown("Compare: (1) exact optimal (2) your custom matrix (3) naive threshold stand.")
    n_hands = st.select_slider(
        "Hands per strategy",
        options=[10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000],
        value=100_000,
    )
    n_bust = st.select_slider(
        "Dealer bust check: hands per upcard",
        options=[20_000, 50_000, 100_000, 200_000],
        value=50_000,
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    threshold = st.slider("Naive threshold k (stand if hard ≥ k)", 12, 17, 17)

    if st.button("Run simulation & charts", type="primary"):
        prog = st.progress(0.0)

        status = st.empty()
        status.write("Checking dealer bust rates…")
        mc_bust = mc_dealer_bust(int(n_bust), seed=int(seed))
        exact_rows = dealer_table()
        st.pyplot(fig_dealer_bust(exact_rows, mc_bust), clear_figure=True)

        bust_df = pd.DataFrame(
            [
                {
                    "upcard": u,
                    "exact_bust": round(dealer_from_upcard(u).get("bust", 0.0), 6),
                    "mc_bust": round(mc_bust[u], 6),
                    "abs_err": round(abs(mc_bust[u] - dealer_from_upcard(u).get("bust", 0.0)), 6),
                }
                for u in UPCARDS
            ]
        )
        st.dataframe(bust_df, width="stretch")

        opt = default_strategy_matrix()
        custom = normalize_strategy(st.session_state.strategy_df)
        naive = default_strategy_matrix()
        for i, H in enumerate(naive["Hard"]):
            for u in UPCARDS:
                naive.loc[i, u] = "S" if int(H) >= threshold else "H"

        results = []
        specs = [
            ("Exact optimal", opt, 0.05, 0.35),
            ("Custom strategy", custom, 0.35, 0.65),
            (f"Threshold ≥{threshold}", naive, 0.65, 0.95),
        ]
        for name, mat, a, b in specs:
            status.write(f"Simulating: {name}")

            def cb(p, a=a, b=b):
                prog.progress(min(a + (b - a) * p, 1.0))

            out = mc_custom_strategy(mat, int(n_hands), seed=int(seed) + (abs(hash(name)) % 10000), progress_cb=cb)
            out["name"] = name
            results.append(out)

        prog.progress(1.0)
        status.write("Done.")
        res_df = pd.DataFrame(
            [
                {
                    "strategy": r["name"],
                    "hands": r["n"],
                    "EV": round(r["ev"], 5),
                    "win_rate": round(r["win_rate"], 4),
                    "wins": r["wins"],
                    "losses": r["losses"],
                    "pushes": r["pushes"],
                    "player_busts": r["player_busts"],
                }
                for r in results
            ]
        )
        st.dataframe(res_df, width="stretch")
        st.pyplot(fig_ev_bars(results), clear_figure=True)
        st.pyplot(fig_delta_heatmap(), clear_figure=True)
        st.success("Finished. Screenshot charts for IA Validation / Results.")

with tab_theory:
    st.subheader("Exact recursion (same as src/exact.py)")
    rows = dealer_table()
    exact_df = pd.DataFrame(
        [
            {
                "upcard": r["upcard"],
                "bust%": round(100 * r["bust"], 2),
                "17%": round(100 * r["17"], 2),
                "18%": round(100 * r["18"], 2),
                "19%": round(100 * r["19"], 2),
                "20%": round(100 * r["20"], 2),
                "21%": round(100 * r["21"], 2),
            }
            for r in rows
        ]
    )
    st.dataframe(exact_df, width="stretch")
    st.pyplot(fig_dealer_bust(rows), clear_figure=True)
    st.pyplot(fig_delta_heatmap(), clear_figure=True)

    st.markdown("#### Hard 12–16 optimal action & ΔE")
    grid = []
    for H in range(12, 17):
        row = {"H": H}
        for u in UPCARDS:
            dE = E_hit(H, u) - E_stand(H, u)
            row[u] = f"{optimal_action(H, u)}({dE:+.3f})"
        grid.append(row)
    st.dataframe(pd.DataFrame(grid), width="stretch", hide_index=True)

    st.markdown(
        """
**Talking points for the IA**

1. After rules are locked, randomness remains — upcard moves bust from ~11.5% (A) to ~42.3% (6).  
2. For hard 12–16, stand EV collapses to `2 × bust − 1`.  
3. Recursion and Monte Carlo should agree within sampling error (Batch tab).
"""
    )
