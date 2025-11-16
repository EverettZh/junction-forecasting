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


