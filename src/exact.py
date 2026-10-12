"""Exact dealer distributions and optimal hard hit/stand (infinite-deck / CSM, S17)."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Union

# Fixed draw probabilities = one full deck composition (same ratios as six complete decks under CSM)
P = {"A": 4 / 52}
for j in range(2, 10):
    P[str(j)] = 4 / 52
P["10"] = 16 / 52
VAL = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10}
UPCARDS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

Dist = Dict[Union[int, str], float]


@lru_cache(None)
def dealer_dist(total_low: int, has_ace: bool) -> Dist:
    if total_low > 21:
        return {"bust": 1.0}
    T = total_low + (10 if has_ace and total_low + 10 <= 21 else 0)
    if T >= 17:  # S17
        return {T: 1.0}
    res: Dist = {}
    for c, p in P.items():
        nl = total_low + VAL[c]
        ha = has_ace or (c == "A")
        for k, v in dealer_dist(nl, ha).items():
            res[k] = res.get(k, 0.0) + p * v
    return res


def dealer_from_upcard(u: str) -> Dist:
    if u == "A":
        return dealer_dist(1, True)
    return dealer_dist(VAL[u], False)


@lru_cache(None)
def E_stand(H: int, u: str) -> float:
    d = dealer_from_upcard(u)
    bust = d.get("bust", 0.0)
    if H <= 16:
        return 2 * bust - 1
    if H <= 20:
        win = bust + sum(d.get(k, 0.0) for k in range(17, H))
        lose = sum(d.get(k, 0.0) for k in range(H + 1, 22))
        return win - lose
    # hard 21 vs ENHC-style natural: dealer natural beats non-natural 21
    win = bust + sum(d.get(k, 0.0) for k in range(17, 21))
    nat = P["10"] if u == "A" else (P["A"] if u == "10" else 0.0)
    return win - nat


@lru_cache(None)
def E_hit(H: int, u: str) -> float:
    e = 0.0
    for c, p in P.items():
        new = H + VAL[c]  # Ace as 1 when continuing from a hard total
        if new > 21:
            e += p * (-1)
        else:
            e += p * max(E_stand(new, u), E_hit(new, u))
    return e


def optimal_action(H: int, u: str) -> str:
    return "S" if E_stand(H, u) >= E_hit(H, u) else "H"


def dealer_table() -> List[dict]:
    rows = []
    for u in UPCARDS:
        d = dealer_from_upcard(u)
        rows.append(
            {
                "upcard": u,
                "bust": d.get("bust", 0.0),
                "17": d.get(17, 0.0),
                "18": d.get(18, 0.0),
                "19": d.get(19, 0.0),
                "20": d.get(20, 0.0),
                "21": d.get(21, 0.0),
            }
        )
    return rows


def decision_table(H_min: int = 12, H_max: int = 16) -> List[dict]:
    rows = []
    for H in range(H_min, H_max + 1):
        for u in UPCARDS:
            es, eh = E_stand(H, u), E_hit(H, u)
            rows.append(
                {
                    "hard": H,
                    "upcard": u,
                    "E_stand": es,
                    "E_hit": eh,
                    "delta": eh - es,
                    "action": "S" if es >= eh else "H",
                }
            )
    return rows


if __name__ == "__main__":
    print("Dealer terminal distribution (exact)")
    for r in dealer_table():
        print(
            f"{r['upcard']:>3} bust={100*r['bust']:5.2f}% "
            f"17={100*r['17']:5.2f}% 18={100*r['18']:5.2f}% "
            f"19={100*r['19']:5.2f}% 20={100*r['20']:5.2f}% 21={100*r['21']:5.2f}%"
        )
    print("\nOptimal hard hit/stand")
    for H in range(12, 17):
        cells = []
        for u in UPCARDS:
            dE = E_hit(H, u) - E_stand(H, u)
            cells.append(f"{optimal_action(H,u)}({dE:+.3f})")
        print(f"{H}: " + " ".join(cells))
