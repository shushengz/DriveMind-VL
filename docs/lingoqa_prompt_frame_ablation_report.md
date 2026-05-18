# LingoQA Prompt / Frame Ablation Report

## Setup

This report compares prompt variants and frame-selection strategies on the 100-sample LingoQA control subset.
It is intended as a lightweight server experiment before any additional SFT/RFT training.

## Overall

| prompt | frame | images | answer_f1 | pass_rate | avg_reward |
|---|---|---:|---:|---:|---:|
| spatial | first_middle_last | 3 | 0.2912 | 0.5500 | 0.4664 |
| spatial | uniform | 5 | 0.2737 | 0.4400 | 0.4587 |
| current | first_n | 5 | 0.2508 | 0.4100 | 0.4526 |
| temporal | uniform | 5 | 0.2487 | 0.4200 | 0.4544 |
| evidence | uniform | 5 | 0.2440 | 0.3900 | 0.4522 |

## Capability Focus

| prompt | frame | spatial_f1 | object_f1 | counting_f1 | reasoning_f1 |
|---|---|---:|---:|---:|---:|
| spatial | first_middle_last | 0.3134 | 0.3791 | 0.1738 | 0.2034 |
| spatial | uniform | 0.2753 | 0.3561 | 0.1759 | 0.1589 |
| current | first_n | 0.2653 | 0.3824 | 0.1336 | 0.1690 |
| temporal | uniform | 0.2345 | 0.3661 | 0.1079 | 0.1633 |
| evidence | uniform | 0.2522 | 0.3410 | 0.1158 | 0.1378 |

## How To Read

- A useful prompt/frame strategy should improve spatial localization without collapsing object recognition.
- Overall F1 alone is not enough because LingoQA is skewed toward spatial localization.
- If a strategy improves text-only-like answers but not visual-control gaps, it should not be treated as grounding progress.
