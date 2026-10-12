# -*- coding: utf-8 -*-
"""Macau-style Blackjack · Math IA Lab (Streamlit)."""
from __future__ import annotations

import sys
from pathlib import Path

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

st.set_page_config(
    page_title="21点 · 数学 IA 实验室",
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


def label(c: str) -> str:
    return c


def render_cards(cards: list[str]) -> str:
    return "  ".join(f"`{label(c)}`" for c in cards) if cards else "—"


def default_strategy_matrix() -> pd.DataFrame:
    rows = []
    for H in range(12, 17):
        row = {"硬点数": H}
        for u in UPCARDS:
            row[u] = optimal_action(H, u)
        rows.append(row)
    return pd.DataFrame(rows).set_index("硬点数")


def strategy_action(matrix: pd.DataFrame, H: int, u: str) -> str:
    if H < 12:
        return "H"
    if H >= 17:
        return "S"
    if H not in matrix.index or u not in matrix.columns:
        return optimal_action(min(max(H, 12), 16), u) if 12 <= H <= 16 else ("S" if H >= 17 else "H")
    val = str(matrix.loc[H, u]).upper().strip()
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
        return -bet, "玩家爆牌，庄家胜"
    if dealer_outcome == "bust":
        return bet, "庄家爆牌，玩家胜"
    d = int(dealer_outcome)
    if player_total > d:
        return bet, f"玩家 {player_total} > 庄家 {d}，玩家胜"
    if player_total < d:
        return -bet, f"玩家 {player_total} < 庄家 {d}，庄家胜"
    return 0, f"平局 {player_total}"


def apply_pnl(pnl: int) -> None:
    # bet already deducted when dealing
    st.session_state.balance += st.session_state.hand["bet"] + pnl
    st.session_state.stats["played"] += 1
    if pnl > 0:
        st.session_state.stats["wins"] += 1
    elif pnl < 0:
        st.session_state.stats["losses"] += 1
    else:
        st.session_state.stats["pushes"] += 1


def mc_custom_strategy(matrix: pd.DataFrame, n_hands: int, seed: int, progress_cb=None) -> dict:
    r = np.random.default_rng(seed)
    pay = 0.0
    wins = losses = pushes = busts_p = 0
    chunk = max(n_hands // 100, 1)
    for i in range(n_hands):
        cards = [draw(r), draw(r)]
        up = draw(r)
        total, soft = hand_total(cards)
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
    }


def fig_dealer_bust(exact_rows, mc_map=None):
    ups = [row["upcard"] for row in exact_rows]
    exact = [100 * row["bust"] for row in exact_rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ups, exact, "-o", color="#1f4e79", lw=2, label="精确 bust%")
    if mc_map:
        ax.plot(ups, [100 * mc_map[u] for u in ups], "--s", color="#c0392b", lw=1.6, label="蒙特卡洛 bust%")
    ax.set_xlabel("庄家明牌")
    ax.set_ylabel("爆牌概率 (%)")
    ax.set_title("庄家爆牌率（按明牌）— IA 核心图")
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
    ax.set_xlabel("庄家明牌")
    ax.set_ylabel("闲家硬点数")
    ax.set_title("ΔE = E(要牌)−E(停牌) 热力图 — IA 核心图")
    for i in range(len(Hs)):
        for j in range(len(UPCARDS)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_strategy_grid(custom: pd.DataFrame, base: pd.DataFrame):
    diff = (base.astype(str).values != custom.astype(str).values).astype(float)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.imshow(diff, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(UPCARDS)))
    ax.set_xticklabels(UPCARDS)
    ax.set_yticks(range(len(custom.index)))
    ax.set_yticklabels(list(custom.index))
    ax.set_title("自定义策略表（红底 = 与精确最优不同）")
    ax.set_xlabel("庄家明牌")
    ax.set_ylabel("硬点数")
    for i in range(custom.shape[0]):
        for j in range(custom.shape[1]):
            ax.text(j, i, str(custom.values[i, j]), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def fig_ev_bars(rows: list[dict]):
    labels = [r["name"] for r in rows]
    evs = [r["ev"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1f4e79" if e >= 0 else "#c0392b" for e in evs]
    ax.bar(labels, evs, color=colors)
    ax.axhline(0, color="gray", lw=1)
    ax.set_ylabel("单位注期望 EV")
    ax.set_title("策略期望对比（蒙特卡洛）")
    for i, v in enumerate(evs):
        ax.text(i, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    return fig


init_state()

st.title("🃏 虚拟二十一 · 数学 IA 实验室")
st.caption(
    "IB Math AA HL Exploration 配套工具：锁定澳门风格规则（CSM／无限牌靴近似、S17、硬牌要／停），"
    "提供练习对局、可编辑停牌策略、超大样本蒙特卡洛，以及 IA 文中的 bust 曲线与 ΔE 热力图。"
)

with st.sidebar:
    st.header("World Lock")
    st.markdown(
        """
**L1** 六副完整比例  
**L2** CSM＝固定有放回概率  
**L3** 庄家 S17  
**L4** 闲家先打完  
**L5** 正文：Hit / Stand  
**L6** 赢 +1 / 输 −1 / 和 0  
"""
    )
    st.divider()
    st.session_state.bet = st.select_slider(
        "练习桌单局下注", options=[5, 10, 25, 50, 100], value=st.session_state.bet
    )
    if st.button("重置积分与统计", use_container_width=True):
        st.session_state.balance = 10000
        st.session_state.stats = {"played": 0, "wins": 0, "losses": 0, "pushes": 0}
        st.session_state.hand = None
        st.rerun()

if "strategy_df" not in st.session_state:
    st.session_state.strategy_df = default_strategy_matrix()

tab_play, tab_strategy, tab_batch, tab_theory = st.tabs(
    ["🎮 练习对局", "🧩 定制停牌策略", "📈 大批量蒙特卡洛", "📐 精确模型与 IA 图"]
)

with tab_play:
    s = st.session_state.stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("积分余额", f"{st.session_state.balance:,}")
    m2.metric("已玩局数", s["played"])
    m3.metric("胜 / 负 / 和", f"{s['wins']} / {s['losses']} / {s['pushes']}")
    m4.metric("胜率", f"{(100 * s['wins'] / s['played']) if s['played'] else 0:.1f}%")
    st.info("练习桌帮助建立直觉；写进 IA 的数字请用右侧「大批量蒙特卡洛」和「精确模型」。")

    hand = st.session_state.hand
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.subheader("牌面")
        if hand is None:
            st.write("点击 **发牌** 开始。")
        else:
            if hand["done"]:
                st.markdown(f"**庄家**　{render_cards(hand['dealer_cards'])}　→ `{hand['dealer_final']}`")
            else:
                st.markdown(f"**庄家**　`{hand['dealer_up']}`　🂠")
            pt, soft = hand_total(hand["player"])
            st.markdown(f"**你**　{render_cards(hand['player'])}")
            st.caption(f"点数 {pt}{'（软）' if soft else '（硬）'} · 注码 {hand['bet']}")
            if hand["done"]:
                st.success(hand["message"])

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("发牌", type="primary", use_container_width=True, disabled=bool(hand and not hand["done"])):
            if st.session_state.balance < st.session_state.bet:
                st.error("积分不足")
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
            if b2.button("要牌", use_container_width=True):
                hand["player"].append(draw(rng()))
                pt, _ = hand_total(hand["player"])
                if pt > 21:
                    hand["done"] = True
                    hand["dealer_cards"] = [hand["dealer_up"]]
                    hand["dealer_final"] = "(未要牌)"
                    hand["message"] = "玩家爆牌，庄家胜"
                    apply_pnl(-hand["bet"])
                st.rerun()
            if b3.button("停牌", use_container_width=True):
                cards, final = finish_dealer(hand["dealer_up"], rng())
                hand["dealer_cards"] = cards
                hand["dealer_final"] = final
                pt, _ = hand_total(hand["player"])
                pnl, msg = settle(pt, final, hand["bet"])
                hand["done"] = True
                hand["message"] = msg
                apply_pnl(pnl)
                st.rerun()
            if b4.button("IA 建议", use_container_width=True):
                pt, soft = hand_total(hand["player"])
                if soft:
                    st.warning("软牌不在 IA 主表；经验：≥17 停，否则要。")
                elif 12 <= pt <= 16:
                    act = optimal_action(pt, hand["dealer_up"])
                    dE = E_hit(pt, hand["dealer_up"]) - E_stand(pt, hand["dealer_up"])
                    st.info(f"精确最优：**{'停牌' if act == 'S' else '要牌'}**（ΔE={dE:+.3f}）")
                else:
                    st.info("硬牌 <12 倾向要，≥17 倾向停。")
        if hand and hand["done"] and st.button("再来一局", use_container_width=True):
            st.session_state.hand = None
            st.rerun()
    with col_b:
        st.subheader("规则")
        st.markdown(
            """
| 项目 | 本实验室 |
|---|---|
| 抽牌 | CSM 固定比例 |
| 庄家 | S17 |
| 赔率 | 1:1（简化） |
| IA 焦点 | bust% · 硬12–16 |
"""
        )

with tab_strategy:
    st.subheader("硬 12–16 要／停矩阵（可编辑）")
    st.caption("单元格填 `H`（要牌）或 `S`（停牌）。默认填入精确最优；改完后用于大批量模拟。")
    base = default_strategy_matrix()
    edited = st.data_editor(
        st.session_state.strategy_df,
        use_container_width=True,
        key="strategy_editor",
    )
    # normalize
    cleaned = edited.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].astype(str).str.upper().str.strip().map(
            lambda x: "S" if x.startswith("S") else "H"
        )
    st.session_state.strategy_df = cleaned

    c1, c2, c3 = st.columns(3)
    if c1.button("恢复精确最优", use_container_width=True):
        st.session_state.strategy_df = default_strategy_matrix()
        st.rerun()
    if c2.button("全部停牌 S", use_container_width=True):
        df = default_strategy_matrix()
        for col in df.columns:
            df[col] = "S"
        st.session_state.strategy_df = df
        st.rerun()
    if c3.button("全部要牌 H", use_container_width=True):
        df = default_strategy_matrix()
        for col in df.columns:
            df[col] = "H"
        st.session_state.strategy_df = df
        st.rerun()

    st.pyplot(fig_strategy_grid(cleaned, base), clear_figure=True)
    diffs = int((base.astype(str).values != cleaned.astype(str).values).sum())
    st.write(f"与精确最优不同的格子数：**{diffs}** / {base.size}")

with tab_batch:
    st.subheader("超大样本蒙特卡洛")
    st.markdown("对照三组：①精确最优策略 ②你在上一页定制的策略 ③朴素阈值（硬牌 ≥ k 停）。")
    n_hands = st.select_slider(
        "模拟局数",
        options=[10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000],
        value=100_000,
    )
    n_bust = st.select_slider(
        "庄家 bust 校验：每明牌手数",
        options=[20_000, 50_000, 100_000, 200_000],
        value=50_000,
    )
    seed = st.number_input("随机种子", min_value=0, value=42, step=1)
    threshold = st.slider("朴素阈值 k（硬牌 ≥ k 停）", 12, 17, 17)

    if st.button("开始模拟并出图", type="primary"):
        prog = st.progress(0.0, text="运行中…")

        def cb(p):
            prog.progress(min(p, 1.0))

        # dealer bust validation
        prog.progress(0.0, text="校验庄家 bust…")
        mc_bust = mc_dealer_bust(int(n_bust), seed=int(seed))
        exact_rows = dealer_table()
        st.pyplot(fig_dealer_bust(exact_rows, mc_bust), clear_figure=True)

        bust_df = pd.DataFrame(
            [
                {
                    "明牌": u,
                    "精确bust": dealer_from_upcard(u).get("bust", 0.0),
                    "MC_bust": mc_bust[u],
                    "绝对误差": abs(mc_bust[u] - dealer_from_upcard(u).get("bust", 0.0)),
                }
                for u in UPCARDS
            ]
        )
        st.dataframe(
            bust_df.style.format({"精确bust": "{:.4%}", "MC_bust": "{:.4%}", "绝对误差": "{:.4%}"}),
            use_container_width=True,
        )

        # strategies
        opt = default_strategy_matrix()
        custom = st.session_state.strategy_df
        naive = default_strategy_matrix()
        for H in naive.index:
            for u in UPCARDS:
                naive.loc[H, u] = "S" if H >= threshold else "H"

        results = []
        for name, mat, frac0, frac1 in [
            ("精确最优", opt, 0.05, 0.35),
            ("自定义策略", custom, 0.35, 0.65),
            (f"阈值≥{threshold}停", naive, 0.65, 0.95),
        ]:
            prog.progress(frac0, text=f"模拟：{name}")

            def cb2(p, a=frac0, b=frac1):
                prog.progress(a + (b - a) * p, text=f"模拟：{name}")

            out = mc_custom_strategy(mat, int(n_hands), seed=int(seed) + hash(name) % 10000, progress_cb=cb2)
            out["name"] = name
            results.append(out)

        prog.progress(1.0, text="完成")
        res_df = pd.DataFrame(results)[["name", "n", "ev", "win_rate", "wins", "losses", "pushes", "player_busts"]]
        res_df.columns = ["策略", "局数", "EV", "胜率", "胜", "负", "和", "玩家爆牌"]
        st.dataframe(
            res_df.style.format({"EV": "{:+.5f}", "胜率": "{:.2%}"}),
            use_container_width=True,
        )
        st.pyplot(fig_ev_bars(results), clear_figure=True)
        st.pyplot(fig_delta_heatmap(), clear_figure=True)
        st.success("模拟完成。可将图表截图写入 IA 的 Validation / Results 节。")

with tab_theory:
    st.subheader("精确递归结果（与 `src/exact.py` 一致）")
    rows = dealer_table()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "明牌": r["upcard"],
                    "bust%": 100 * r["bust"],
                    "17%": 100 * r["17"],
                    "18%": 100 * r["18"],
                    "19%": 100 * r["19"],
                    "20%": 100 * r["20"],
                    "21%": 100 * r["21"],
                }
                for r in rows
            ]
        ).style.format({c: "{:.2f}" for c in ["bust%", "17%", "18%", "19%", "20%", "21%"]}),
        use_container_width=True,
    )
    st.pyplot(fig_dealer_bust(rows), clear_figure=True)
    st.pyplot(fig_delta_heatmap(), clear_figure=True)

    st.markdown("#### 硬 12–16 最优动作与 ΔE")
    grid = []
    for H in range(12, 17):
        row = {"H": H}
        for u in UPCARDS:
            dE = E_hit(H, u) - E_stand(H, u)
            row[u] = f"{optimal_action(H, u)}({dE:+.3f})"
        grid.append(row)
    st.dataframe(pd.DataFrame(grid).set_index("H"), use_container_width=True)

    st.markdown(
        """
**写进 IA 时可强调：**

1. 规则锁定后，随机性仍在——明牌把 bust 率从约 11.5%（A）推到约 42.3%（6）。  
2. 硬 12–16 停牌期望塌缩为 2×bust−1。  
3. 递归与蒙特卡洛在误差内一致（见大批量页）。
"""
    )
