# Table A1 — Per-bucket visit counts

Source: `results/mc_qtable_5M.npz` (Monte Carlo, 5M training episodes).

## Pre-flop visit counts

| Bucket | Description | check | bet_3x | bet_4x | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Premium pairs (AA/KK/QQ) | 9,774 | 9,626 | 48,152 | 67,552 |
| 1 | Mid pairs (JJ/TT/99/88) | 12,972 | 12,923 | 64,902 | 90,797 |
| 2 | Low pairs (77-22) | 19,450 | 19,533 | 96,965 | 135,948 |
| 3 | Suited ace-broadway (AKs-ATs) | 8,744 | 9,024 | 42,672 | 60,440 |
| 4 | Suited ace-low (A9s-A2s) | 17,349 | 17,228 | 86,586 | 121,163 |
| 5 | Offsuit ace-broadway (AKo-ATo) | 26,082 | 25,949 | 128,959 | 180,990 |
| 6 | Offsuit ace-low (A9o-A2o) | 51,922 | 52,059 | 259,048 | 363,029 |
| 7 | Suited Kx high (KQs/KJs/KTs) | 6,621 | 6,495 | 32,352 | 45,468 |
| 8 | Suited K low / offsuit K high | 36,851 | 36,817 | 183,249 | 256,917 |
| 9 | Offsuit K low (K9o-K2o) | 51,738 | 56,015 | 253,174 | 360,927 |
| 10 | Suited QJ/QT/JT | 6,408 | 30,563 | 8,152 | 45,123 |
| 11 | Offsuit QJ/QT/JT | 19,820 | 24,294 | 92,094 | 136,208 |
| 12 | Suited connectors mid (T9s-53s) | 25,802 | 128,612 | 26,221 | 180,635 |
| 13 | Other suited | 86,061 | 430,050 | 86,768 | 602,879 |
| 14 | Other offsuit | 1,527,236 | 487,328 | 337,360 | 2,351,924 |

_Min visits: 6,408. Max visits: 1,527,236. Cells with <100 visits: 0._

## Flop visit counts

| Bucket | Description | check | bet_2x | Total |
| --- | --- | ---: | ---: | ---: |
| 0 | Straight or better | 2,448 | 9,613 | 12,061 |
| 1 | Three of a kind | 6,813 | 26,747 | 33,560 |
| 2 | Two pair | 15,951 | 66,279 | 82,230 |
| 3 | Top pair (uses hole, >= board) | 28,580 | 102,645 | 131,225 |
| 4 | Other one pair | 118,573 | 534,511 | 653,084 |
| 5 | 4-flush or OESD (no pair) | 17,888 | 75,332 | 93,220 |
| 6 | Gutshot or high 3-flush | 43,285 | 178,203 | 221,488 |
| 7 | High-card air | 559,351 | 120,611 | 679,962 |

_Min visits: 2,448. Max visits: 559,351. Cells with <100 visits: 0._

## River visit counts

| Bucket | Description | fold | bet_1x | Total |
| --- | --- | ---: | ---: | ---: |
| 0 | Royal / straight flush | 37 | 68 | 105 |
| 1 | Quads or full house | 2,897 | 6,453 | 9,350 |
| 2 | Flush or straight | 10,102 | 33,646 | 43,748 |
| 3 | Trips or two pair | 34,786 | 106,359 | 141,145 |
| 4 | Top pair (uses hole, >= board) | 5,042 | 13,049 | 18,091 |
| 5 | Other one pair | 66,077 | 284,293 | 350,370 |
| 6 | High card | 192,357 | 37,723 | 230,080 |

_Min visits: 37. Max visits: 284,293. Cells with <100 visits: 2._
