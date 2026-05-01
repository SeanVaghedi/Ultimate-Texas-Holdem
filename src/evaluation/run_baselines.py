"""Evaluate baselines + trained MC, Q-learning, SARSA, SARSA(lambda), and (if available) linear-FA agents.

Run from repo root:

    python -m src.evaluation.run_baselines

Or directly:

    python src/evaluation/run_baselines.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agents.mc_agent import MonteCarloAgent  # noqa: E402
from agents.mc_linear_fa import MonteCarloLinearFAAgent  # noqa: E402
from agents.q_learning_agent import QLearningAgent  # noqa: E402
from agents.sarsa_agent import SarsaAgent  # noqa: E402
from agents.sarsa_lambda_agent import SarsaLambdaAgent  # noqa: E402
from evaluation.evaluate import (  # noqa: E402
    evaluate_policy,
    print_evaluation_summary,
)
from linear_q import LinearQ  # noqa: E402
from qtable import QTable  # noqa: E402
from strategies.baselines import (  # noqa: E402
    always_fold_policy,
    naive_heuristic_policy,
    random_policy,
)
from strategies.optimal import optimal_policy  # noqa: E402


NUM_HANDS = 100_000
SEED = 0
MC_QTABLE_PATH = "results/mc_qtable_5M.npz"
Q_QTABLE_PATH = "results/q_qtable_5M.npz"
SARSA_QTABLE_PATH = "results/sarsa_qtable_5M.npz"
SARSA_LAMBDA_PATH = "results/sarsa_lambda_qtable_5M_lam09.npz"
LINEAR_FA_PATH = "results/mc_linear_fa_5M.npz"
OUTPUT_JSON_PATH = "results/baseline_evaluation.json"


def main() -> None:
    results: dict = {}

    rand_policy = random_policy(random.Random(SEED))
    results["random"] = evaluate_policy(rand_policy, NUM_HANDS, seed=SEED)
    print_evaluation_summary("Random", results["random"])

    results["always_fold"] = evaluate_policy(
        always_fold_policy, NUM_HANDS, seed=SEED
    )
    print_evaluation_summary("Always-Fold", results["always_fold"])

    results["naive"] = evaluate_policy(
        naive_heuristic_policy, NUM_HANDS, seed=SEED
    )
    print_evaluation_summary("Naive Heuristic", results["naive"])

    results["optimal"] = evaluate_policy(
        optimal_policy, NUM_HANDS, seed=SEED
    )
    print_evaluation_summary("EV-Optimal Baseline", results["optimal"])

    mc_q_table = QTable()
    mc_q_table.load(MC_QTABLE_PATH)
    mc_agent = MonteCarloAgent(mc_q_table, random.Random(SEED))
    results["mc_trained_5M"] = evaluate_policy(
        mc_agent.frozen_policy_fn, NUM_HANDS, seed=SEED
    )
    print_evaluation_summary(
        "Trained MC Agent (5M episodes)", results["mc_trained_5M"]
    )

    q_q_table = QTable()
    q_q_table.load(Q_QTABLE_PATH)
    q_agent = QLearningAgent(q_q_table, random.Random(SEED))
    results["q_trained_5M"] = evaluate_policy(
        q_agent.frozen_policy_fn, NUM_HANDS, seed=SEED
    )
    print_evaluation_summary(
        "Trained Q-Learning Agent (5M episodes)", results["q_trained_5M"]
    )

    sarsa_q_table = QTable()
    sarsa_q_table.load(SARSA_QTABLE_PATH)
    sarsa_agent = SarsaAgent(sarsa_q_table, random.Random(SEED))
    results["sarsa_trained_5M"] = evaluate_policy(
        sarsa_agent.frozen_policy_fn, NUM_HANDS, seed=SEED
    )
    print_evaluation_summary(
        "Trained SARSA Agent (5M episodes)", results["sarsa_trained_5M"]
    )

    sarsa_lambda_q_table = QTable()
    sarsa_lambda_q_table.load(SARSA_LAMBDA_PATH)
    sarsa_lambda_agent = SarsaLambdaAgent(
        sarsa_lambda_q_table, random.Random(SEED)
    )
    results["sarsa_lambda_5M_lam09"] = evaluate_policy(
        sarsa_lambda_agent.frozen_policy_fn, NUM_HANDS, seed=SEED
    )
    print_evaluation_summary(
        "Trained SARSA(lambda=0.9) Agent (5M episodes)",
        results["sarsa_lambda_5M_lam09"],
    )

    if Path(LINEAR_FA_PATH).exists():
        linear_q = LinearQ()
        linear_q.load(LINEAR_FA_PATH)
        linear_fa_agent = MonteCarloLinearFAAgent(
            linear_q, random.Random(SEED)
        )
        results["mc_linear_fa_5M"] = evaluate_policy(
            linear_fa_agent.frozen_policy_fn, NUM_HANDS, seed=SEED
        )
        print_evaluation_summary(
            "Trained MC + Linear FA Agent (5M episodes)",
            results["mc_linear_fa_5M"],
        )

    Path(OUTPUT_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
