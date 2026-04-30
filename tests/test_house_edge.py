"""Empirical UTH house-edge measurement under the hard-coded near-optimal
strategy. Run as a script, not a pytest unit test:

    python tests/test_house_edge.py

Pytest will collect the file but find no test_* functions, so a normal
`pytest tests/` run will skip it cleanly. The asserted band [1.5%, 3.5%]
is intentionally generous because the per-hand variance in UTH is high
(blind bonus pays up to 500x) and 100k hands gives a wide CI.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from game import UTHGame  # noqa: E402
from strategies.optimal import optimal_policy  # noqa: E402


def main(n_hands: int = 100_000, seed: int = 0) -> None:
    game = UTHGame(seed=seed)

    total = 0.0
    sum_sq = 0.0
    for _ in range(n_hands):
        net = game.play_hand(optimal_policy)["net_chips"]
        total += net
        sum_sq += net * net

    mean = total / n_hands
    variance = sum_sq / n_hands - mean * mean
    se = math.sqrt(max(variance, 0.0) / n_hands)
    half_ci = 1.96 * se

    # Ante = 1, so mean net chips per hand IS the per-ante outcome and
    # -mean is the per-ante house edge.
    house_edge_pct = -mean * 100
    half_ci_pct = half_ci * 100
    ci_low = house_edge_pct - half_ci_pct
    ci_high = house_edge_pct + half_ci_pct

    print(f"Hands played:    {n_hands:,}")
    print(f"Total net chips: {total:+.2f}")
    print(f"Mean per hand:   {mean:+.5f} chips (= -house edge per ante)")
    print(f"House edge:      {house_edge_pct:+.3f}% per ante")
    print(f"95% CI:          [{ci_low:+.3f}%, {ci_high:+.3f}%]")

    assert 1.5 <= house_edge_pct <= 3.5, (
        f"House edge {house_edge_pct:.3f}% outside expected band [1.5%, 3.5%]"
    )
    print("PASS: house edge within expected band [1.5%, 3.5%]")


if __name__ == "__main__":
    main()
