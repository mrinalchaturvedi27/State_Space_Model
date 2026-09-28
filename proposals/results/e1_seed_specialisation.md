# E1 -- seed specialisation (R1: MoE-as-signers)

### mamba, test (6068 sentences)

| system | BLEU-4 | chrF2 |
|---|---|---|
| seed 13 | 4.09 | 22.27 |
| seed 42 | 3.54 | 20.7 |
| seed 1337 | 3.57 | 20.68 |
| seed mean | 3.73 | 21.22 |
| ensemble (logit avg) | 4.91 | 23.18 |
| oracle best-of-3 (peeks at ref) | 5.61 | 25.77 |
| random pick per sentence | 3.77 | 21.22 |

Pearson r of sentence chrF2 between seeds: 13-42 0.67, 13-1337 0.67, 42-1337 0.66
Decided sentences: 6028 of 6068; winner share s13 41.7%, s42 29.1%, s1337 29.3%
Winner concentration within video (196 videos, 5967 sentences): observed 0.3660, permutation null 0.3652 ± 0.0015, p = 0.299

### transformer, test (6068 sentences)

| system | BLEU-4 | chrF2 |
|---|---|---|
| seed 13 | 2.89 | 20.18 |
| seed 42 | 2.97 | 18.94 |
| seed 1337 | 2.74 | 18.95 |
| seed mean | 2.87 | 19.36 |
| ensemble (logit avg) | 3.98 | 20.92 |
| oracle best-of-3 (peeks at ref) | 4.31 | 23.6 |
| random pick per sentence | 2.91 | 19.36 |

Pearson r of sentence chrF2 between seeds: 13-42 0.63, 13-1337 0.62, 42-1337 0.64
Decided sentences: 6029 of 6068; winner share s13 41.4%, s42 28.7%, s1337 29.9%
Winner concentration within video (196 videos, 5968 sentences): observed 0.3669, permutation null 0.3643 ± 0.0015, p = 0.0465

### mamba, val (5653 sentences)

| system | BLEU-4 | chrF2 |
|---|---|---|
| seed 13 | 4.27 | 21.33 |
| seed 42 | 4.18 | 21.15 |
| seed 1337 | 4.03 | 21.22 |
| seed mean | 4.16 | 21.23 |
| oracle best-of-3 (peeks at ref) | 6.44 | 25.96 |
| random pick per sentence | 4.11 | 21.16 |

Pearson r of sentence chrF2 between seeds: 13-42 0.67, 13-1337 0.67, 42-1337 0.65
Decided sentences: 5613 of 5653; winner share s13 33.9%, s42 33.5%, s1337 32.6%
Winner concentration within video (181 videos, 5497 sentences): observed 0.3539, permutation null 0.3552 ± 0.0015, p = 0.797

### transformer, val (5653 sentences)

| system | BLEU-4 | chrF2 |
|---|---|---|
| seed 13 | 3.51 | 19.72 |
| seed 42 | 3.27 | 19.34 |
| seed 1337 | 3.72 | 19.52 |
| seed mean | 3.50 | 19.53 |
| oracle best-of-3 (peeks at ref) | 5.66 | 24.11 |
| random pick per sentence | 3.48 | 19.48 |

Pearson r of sentence chrF2 between seeds: 13-42 0.60, 13-1337 0.63, 42-1337 0.56
Decided sentences: 5617 of 5653; winner share s13 35.3%, s42 32.2%, s1337 32.4%
Winner concentration within video (181 videos, 5500 sentences): observed 0.3584, permutation null 0.3557 ± 0.0016, p = 0.0495
