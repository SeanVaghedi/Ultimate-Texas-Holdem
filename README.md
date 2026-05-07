# Ultimate Texas Hold'em — Tabular RL

Final project for BU.520.750.51 (AI-Driven Sequential Decision Making). We trained tabular RL agents to play Ultimate Texas Hold'em and tested whether bootstrap-based methods could match or beat Monte Carlo control on this kind of problem.

Spoiler: they couldn't. MC won, and the failure mode (cascade-bias from state aggregation) turned out to be the most interesting part of the project.

## Team

- Sean Vaghedi
- Adit Patel
- Peiyuan Song

## What's in here

A complete UTH simulator, five RL agents, baseline policies for comparison, and the analysis code for our final report.

```
src/
  game.py              # UTH game engine
  payouts.py           # Authentic UTH payouts (Blind bonus, ante, play)
  hand_eval.py         # Hand strength evaluation (uses treys)
  abstraction.py       # 75-cell state bucketing
  qtable.py            # Q-table data structure
  features.py          # Feature design for Linear FA
  linear_q.py          # Linear function approximation
  agents/              # All 5 RL agents
  training/            # Training scripts (CLI args)
  strategies/          # EV-Optimal + heuristic baselines
  evaluation/          # Evaluation against baselines
tests/                 # Unit tests
results/               # Saved Q-tables, logs, figures
```

## Setup

```bash
git clone https://github.com/SeanVaghedi/Ultimate-Texas-Holdem.git
cd Ultimate-Texas-Holdem
python -m venv venv
source venv/bin/activate    # on Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running things

Train the MC agent (this is the headline result):

```bash
python -m src.training.train_mc_agent --episodes 5000000 --seed 42
```

Train Q-learning at different learning rates (alpha-sweep):

```bash
python -m src.training.train_q_learning --episodes 5000000 --alpha 0.01
python -m src.training.train_q_learning --episodes 5000000 --alpha 0.1
python -m src.training.train_q_learning --episodes 5000000 --alpha 0.3
```

Train SARSA(λ) at different lambdas:

```bash
python -m src.training.train_sarsa_lambda --episodes 5000000 --lambda_val 0.9
```

Train Linear FA:

```bash
python -m src.training.train_mc_linear_fa --episodes 5000000
```

Run evaluation against all baselines:

```bash
python -m src.evaluation.evaluate --hands 100000
```

Run tests:

```bash
pytest tests/
```

## Results summary

After 5M training episodes and 100k evaluation hands per policy, our headline result is that tabular MC achieves 17.64% house edge versus EV-Optimal's 5.63%. SARSA and Q-learning, despite using the same MDP and training data, both performed worse than random play (52% and 61% house edge respectively). Our analysis traces this to bootstrap-aggregation cascade bias — bucket-averaged Q-values get used as bootstrap targets, and the bias compounds upstream through the trajectory.

The α-sweep on Q-learning showed the failure scales monotonically with learning rate (17% at α=0.01, 59% at α=0.30), and the SARSA(λ) spectrum showed a discontinuous improvement only at λ=1 (full MC). Both confirm bootstrap is the issue.

Full discussion is in the final report PDF.

## Reproducibility

All training is seeded (default `--seed 42`). Running the commands above should reproduce the headline numbers within statistical noise.

## Notes for the grader

- The 75-cell state abstraction is described in `src/abstraction.py`
- EV-Optimal uses exact 990-combo enumeration at the river (see `src/strategies/optimal.py`)
- The Linear FA feature design (marginal one-hots + cross-features = 592 weights) is in `src/features.py`
- Pre-computed results are in `results/` if you don't want to wait for 5M episodes to retrain

## License

Course project, not for commercial use.
