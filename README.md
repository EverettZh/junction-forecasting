# Junction 2025 – Fortum Forecasting Challenge  
### Electricity Consumption Forecasting (48-Hour & 12-Month Horizons)

This repository contains my full solution for the Fortum challenge at Junction 2025.  
The goal is to forecast electricity consumption for 112 customer groups at two horizons:

- **48-hour short-term forecast** (hourly resolution)
- **12-month long-term forecast** (monthly resolution)

The forecasts are generated using **Prophet** models trained independently on each customer group.

All code, data processing, forecasting, evaluation, and exports required by the challenge
are included here.

**Here is the overall FVA averages in comparison:**

For consumption prediction:
-1.2715501121040884 is barely worse for the entire dataset and the other two did worse.

So we must include this as this is basically as good if not worse than the baseline. It is not too bad though. The majority
59% did significantly well, while 34% did significantly worse with the rest 7% did around the same. However, the ones did significantly worse
was really bad and it kind of push our accuracy the other way.

Better (model clearly beats baseline)

Count: 66 / 112 (~59%)

Examples: 28, 29, 30, 38, 41, 43, 73, 74, 76, 116, …

Top performers (largest FVA%):

30 → +26.74%

698 → +26.41%

76 → +25.59%

151 → +25.39%

577 → +25.27%

Worse (model worse than baseline)

Count: 38 / 112 (~34%)

Examples: 36, 37, 39, 40, 42, 157, 196, 199, 201, 231, …

Worst cases (most negative FVA%):

396 → −94.16%

401 → −88.69%

404 → −79.24%

398 → −75.44%

459 → −69.19%

Similar to baseline

Count: 8 / 112 (~7%)

Series with FVA ≈ 0 (model ~ same as baseline):
295, 298, 307, 450, 466, 573, 623, 708

The improved univariate version actually did better! (finally) with FVA score of 3.795034619214879

1️⃣ Better than baseline (FVA > 5%)

Model clearly improves on the naive baseline.

Count: 68 / 112 (~61%)

Examples:
28, 29, 30, 38, 41, 43, 74, 76, 116, 150, 151, 152, 197, 198, 200, 213, 222, 225, 233, 234, 235, 237, 238, 271, 298, 303, 304, 305, 307, 308, 348, 378, 380, 390, 395, 400, 403, 450, 466, 468, 469, 538, 541, 542, 561, 570, 573, 577, 580, 581, 582, 583, 585, 586, 625, 657, 658, 659, 682, 693, 694, 695, 697, 698, 708, 738, 740, 741

Top performers (highest FVA):

30 → FVA +26.38%

698 → +26.23%

76 → +26.00%

235 → +25.31%

741 → +24.64%

also very strong: 116, 152, 577, 380, 659, 682, 577, 583, 586, 577 (all ≈ +23–24%)

These are your best “showcase” series.

2️⃣ Around the same as baseline (−5% ≤ FVA ≤ 5%)

Model and baseline are essentially similar.

Count: 15 / 112 (~13%)

Series:
73, 149, 157, 196, 201, 231, 295, 302, 397, 402, 405, 460, 623, 692, 706

These are fine, but the model doesn’t add much over the naive forecast here.

3️⃣ Worse than baseline (FVA < −5%)

Model is clearly worse than the naive method — good candidates for special treatment or a different approach.

Count: 29 / 112 (~26%)

Examples:
36, 37, 39, 40, 42, 199, 270, 301, 346, 347, 385, 387, 391, 393, 394, 396, 398, 399, 401, 404, 447, 459, 622, 624, 626, 691, 705, 707, 709

Worst cases (most negative FVA):

391 → FVA −69.31%

385 → −69.10%

396 → −56.83%

398 → −53.39%

393 → −51.12%

also very bad: 401, 404, 459, 707, 447 (large negative FVA).

We can see many of the worst cases has improved to baseline level and the worst case has improved from -94% to -69%, this means our improvement actually worked.
