"""Monte Carlo validation of dealer bust rates and hard hit/stand EV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from exact import P, UPCARDS, VAL, E_hit, E_stand, dealer_from_upcard, optimal_action

RNG = np.random.default_rng

CARDS = list(P.keys())
PROBS = np.array([P[c] for c in CARDS], dtype=float)


def draw(rng: np.random.Generator) -> str:
    return CARDS[int(rng.choice(len(CARDS), p=PROBS))]


def soft_total(low: int, has_ace: bool) -> int:
    return low + (10 if has_ace and low + 10 <= 21 else 0)


def play_dealer(upcard: str, rng: np.random.Generator) -> str | int:
    """Return final total 17..21 or 'bust'. Starts from upcard only (no hole card yet)."""
    low = 1 if upcard == "A" else VAL[upcard]
    has_ace = upcard == "A"
    # draw hole / subsequent cards until S17
    while True:
        T = soft_total(low, has_ace)
        if T > 21:
            return "bust"
        if T >= 17:
            return T
        c = draw(rng)
        low += VAL[c]
        has_ace = has_ace or (c == "A")


def mc_dealer_bust(n: int, seed: int = 0) -> dict:
    rng = RNG(seed)
    out = {}
    for u in UPCARDS:
        busts = 0
        for _ in range(n):
            if play_dealer(u, rng) == "bust":
                busts += 1
        out[u] = busts / n
    return out


def mc_ev_action(H: int, u: str, action: str, n: int, rng: np.random.Generator) -> float:
    """Estimate EV of forced hit-once-then-optimal or stand, vs dealer from upcard u."""
    payoff = 0.0
    for _ in range(n):
        if action == "S":
            player = H
        else:
            c = draw(rng)
            player = H + VAL[c]
            if player > 21:
                payoff += -1
                continue
            # continue optimally with hard totals only
            while player <= 21 and optimal_action(player, u) == "H":
                c = draw(rng)
                player = player + VAL[c]
                if player > 21:
                    payoff += -1
                    break
            else:
                if player > 21:
                    continue
                d = play_dealer(u, rng)
                if d == "bust" or (isinstance(d, int) and d < player):
                    payoff += 1
                elif isinstance(d, int) and d > player:
                    payoff += -1
            continue
        d = play_dealer(u, rng)
        if d == "bust" or (isinstance(d, int) and d < player):
            payoff += 1
        elif isinstance(d, int) and d > player:
            payoff += -1
    return payoff / n


def main() -> None:
    ap = argparse.ArgumentParser(description="Monte Carlo checks for Macau-style blackjack model")
    ap.add_argument("--n-dealer", type=int, default=200_000, help="hands per upcard for bust rate")
    ap.add_argument("--n-ev", type=int, default=80_000, help="hands per (H,u,action) EV check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    data.mkdir(exist_ok=True)

    print(f"MC dealer bust rates (n={args.n_dealer} per upcard)")
    mc_bust = mc_dealer_bust(args.n_dealer, seed=args.seed)
    lines = ["upcard,exact_bust,mc_bust,abs_err,se"]
    for u in UPCARDS:
        exact = dealer_from_upcard(u).get("bust", 0.0)
        mc = mc_bust[u]
        se = (mc * (1 - mc) / args.n_dealer) ** 0.5
        print(f"  {u:>3}: exact={100*exact:5.2f}%  mc={100*mc:5.2f}%  |err|={100*abs(mc-exact):.3f}%  SE={100*se:.3f}%")
        lines.append(f"{u},{exact:.6f},{mc:.6f},{abs(mc-exact):.6f},{se:.6f}")
    (data / "mc_dealer_bust.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Focus cells: boundary decisions from the exact table
    focus = [(12, "4"), (12, "5"), (16, "10"), (15, "A"), (13, "2")]
    print(f"\nMC EV checks (n={args.n_ev})")
    rng = RNG(args.seed + 1)
    ev_lines = ["hard,upcard,exact_E_stand,exact_E_hit,exact_action,mc_E_stand,mc_E_hit,mc_action"]
    for H, u in focus:
        es, eh = E_stand(H, u), E_hit(H, u)
        act = optimal_action(H, u)
        mcs = mc_ev_action(H, u, "S", args.n_ev, rng)
        mch = mc_ev_action(H, u, "H", args.n_ev, rng)
        mact = "S" if mcs >= mch else "H"
        print(
            f"  H={H} vs {u}: exact {act} (S={es:+.4f}, H={eh:+.4f}) | "
            f"mc {mact} (S={mcs:+.4f}, H={mch:+.4f})"
        )
        ev_lines.append(
            f"{H},{u},{es:.6f},{eh:.6f},{act},{mcs:.6f},{mch:.6f},{mact}"
        )
    (data / "mc_ev_focus.csv").write_text("\n".join(ev_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {data/'mc_dealer_bust.csv'} and {data/'mc_ev_focus.csv'}")


if __name__ == "__main__":
    main()
