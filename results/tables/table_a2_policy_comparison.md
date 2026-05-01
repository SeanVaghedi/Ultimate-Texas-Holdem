# Table A2 — Policy comparison (100,000 evaluation hands per policy)

Source: `results/baseline_evaluation.json`. All trained agents were evaluated greedy (epsilon=0).

| Policy | Mean / hand | SE | House edge | 95% CI on house edge | Fold rate | Pre-flop dist | Flop dist | River dist |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| Random | -0.5417 | 0.0158 | +54.17% | [+51.07%, +57.26%] | 8.4% | check=33.5% / bet_3x=33.5% / bet_4x=33.0% | check=50.4% / bet_2x=49.6% | fold=49.8% / bet_1x=50.2% |
| Always-Fold | -2.0000 | 0.0000 | +200.00% | [+200.00%, +200.00%] | 100.0% | check=100.0% / bet_3x=0.0% / bet_4x=0.0% | check=100.0% / bet_2x=0.0% | fold=100.0% / bet_1x=0.0% |
| Naive Heuristic | -0.1493 | 0.0137 | +14.93% | [+12.25%, +17.61%] | 14.6% | check=79.7% / bet_3x=0.0% / bet_4x=20.3% | check=53.1% / bet_2x=46.9% | fold=34.4% / bet_1x=65.6% |
| EV-Optimal | -0.0563 | 0.0144 | +5.63% | [+2.81%, +8.45%] | 19.3% | check=62.4% / bet_3x=0.0% / bet_4x=37.6% | check=87.6% / bet_2x=12.4% | fold=35.3% / bet_1x=64.7% |
| MC tabular (5M) | -0.1764 | 0.0156 | +17.64% | [+14.58%, +20.70%] | 6.8% | check=47.3% / bet_3x=16.4% / bet_4x=36.2% | check=37.6% / bet_2x=62.4% | fold=38.1% / bet_1x=61.9% |
| Q-learning (5M) | -0.6088 | 0.0111 | +60.88% | [+58.71%, +63.05%] | 47.0% | check=99.1% / bet_3x=0.0% / bet_4x=0.9% | check=79.5% / bet_2x=20.5% | fold=59.6% / bet_1x=40.4% |
| SARSA (5M) | -0.5192 | 0.0110 | +51.92% | [+49.77%, +54.07%] | 54.1% | check=97.7% / bet_3x=0.9% / bet_4x=1.4% | check=83.9% / bet_2x=16.1% | fold=66.0% / bet_1x=34.0% |
| SARSA(λ=0.9) (5M) | -0.4362 | 0.0119 | +43.62% | [+41.28%, +45.96%] | 47.3% | check=83.3% / bet_3x=10.8% / bet_4x=5.9% | check=89.7% / bet_2x=10.3% | fold=63.3% / bet_1x=36.7% |
| MC + Linear FA (5M) | -0.2840 | 0.0127 | +28.40% | [+25.91%, +30.89%] | 34.9% | check=85.0% / bet_3x=1.8% / bet_4x=13.2% | check=68.5% / bet_2x=31.5% | fold=60.0% / bet_1x=40.0% |
