# Experimental Analysis & Project Overview

**Repository**: `CDC_FM_AD` (`BenjiClothier/sde_ddpm`)  
**Scope**: Comprehensive documentation of experiments, lessons learnt, and practical problems encountered while developing time-series anomaly detection models.  
**Total Documented Experiments**: 30 standalone reports across 10 phases of development.

---

## 1. How We Analyse Pilot Tests (Diagnostic Checks)

When evaluating a candidate model on our 10 pilot datasets, we do not rely solely on a single summary metric (such as VUS-PR or ROC). We use five diagnostic checks to understand precisely *why* an algorithm works or fails:

1. **False Alarms on Normal Peaks (Precision vs Ranking)**:
   - *High ROC but Low PR*: The model successfully ranks the anomaly higher than most background noise, but it repeatedly triggers false alarms on sharp normal peaks (such as normal heartbeats in an ECG or daily traffic surges).
   - *Low ROC and Low PR*: The model is completely blind to that type of anomaly (for example, a small window completely missing a 200-step sensor flatline).

2. **Signal-to-Noise Contrast (Peak-to-Noise Ratio)**:
   - We measure the ratio between the highest anomaly score during the anomaly and the standard deviation of scores during normal baseline operation.
   - A ratio below 2.0 indicates that the anomaly is buried in background noise. A ratio above 10.0 indicates clean, reliable separation.

3. **When False Alarms Occur (Warm-Up vs Cyclical)**:
   - *Warm-Up Transients*: False alarms that occur exclusively in the first few timestamps because digital filters or delay buffers need time to fill.
   - *Cyclical False Alarms*: False alarms that recur rhythmically on every cycle of normal data, indicating a failure to account for normal periodic variation.

4. **Effective Coordinate Dimensions ($d_{\text{eff}}$)**:
   - How many independent coordinates the model is genuinely using to represent the signal.
   - If $d_{\text{eff}} \le 2$, the representation has collapsed onto a line or plane, making it unable to distinguish complex patterns.
   - If $d_{\text{eff}}$ approaches the total window size $L$, the model is not compressing anything and is simply picking up high-frequency sensor noise.

5. **Matching the Natural Physical Cycle**:
   - Does the model's observation window match the genuine repetition period of the physical system?
   - For example, if an industrial motor has a physical cycle of 130 steps, an effective model must track that 130-step cycle rather than locking onto a 10-step surface ripple.

---

## 2. Pre-Benchmark Checklist (Before Running the Full 870 Datasets)

Evaluating a model across all 870 benchmark datasets requires significant GPU cluster time (typically 2 to 6 hours). Before launching a full benchmark sweep, any candidate model must pass three strict checks on our 10 pilot datasets:

| Check | Target Requirement | Plain English Meaning |
| :--- | :---: | :--- |
| **1. Average VUS-PR** | $\ge \mathbf{0.650}$ (Target: $\ge \mathbf{0.700}$) | Across the 10 pilot datasets, the model must achieve an average VUS-PR of at least 0.650 without per-dataset tuning. |
| **2. Worst-Case VUS-PR Floor** | $\ge \mathbf{0.200}$ | The model must not completely fail (near 0.000) on any individual dataset, regardless of anomaly type. |
| **3. Anomaly Coverage** | **All 4 Anomaly Types** | The model must reliably detect: (1) sharp spikes, (2) abnormal cycles (e.g. cardiac arrhythmia), (3) gradual sensor drifts, and (4) sensor flatlines. |

### Current Status of Evaluated Architectures:

- **Discrete Coarse-Fine Grid Search (Phase 8)**:
  - *Pilot VUS-PR*: **0.7816** (VUS-ROC: 0.9380).
  - *Limitation*: Requires training 38 separate models per dataset (~7 minutes per time series). Useful only as a brute-force reference ceiling.
- **Single-Shot Polynomial Embedding (Phase 8, FDE)**:
  - *Pilot VUS-PR*: **0.7321** (full 795 benchmark average: **0.3732** VUS-PR, **0.8097** VUS-ROC).
  - *Strength*: Extremely fast (11 seconds per dataset).
  - *Limitation*: Relied on a fixed heuristic for window size; struggled on datasets with extreme multi-scale characteristics.
- **Harmonic State-Space Models (Phase 9, Versions 1–3)**:
  - *Pilot VUS-PR*: Rose from **0.5292** (v1) to **0.5965** (v2) and **0.6070** (v3, with worst-case floor improving to 0.2424).
  - *Strengths*: Vaulted spike detection from 0.009 to 0.7587 VUS-PR; detected ECG arrhythmia cleanly (0.8235 VUS-PR).
  - *Limitation*: Suffered from filter ringing and sensitive hyperparameter weighting between velocity and position components.
- **Calibrated Multiscale Diffusion Geometry (Phase 9, Experiment 28)**:
  - *Pilot VUS-PR*: **0.4176** in an initial un-tuned nearest-neighbour test (Oracle ceiling: 0.8864).
  - *Breakthrough*: Solved the multi-scale noise mismatch problem without manual tuning, boosting flatline detection from 0.2850 to 0.9848 VUS-PR, and sharp spike detection from 0.0268 to 0.2166 VUS-PR.
- **Multi-Scale Phase-Space Pipeline (Phase 10, Experiment 32)**:
  - *Pilot VUS-PR*: **0.5364** (VUS-ROC: **0.9258**, worst-case floor: **0.1781**).
  - *Breakthrough*: New continuous streaming state-of-the-art across 10 pilot datasets, outperforming Harmonic SSM v3 (**0.5151** VUS-PR) and Exp 28 (**0.4176** VUS-PR).
  - *Efficiency*: Deterministic closed-form Diffusion Geometry executing in **5.13 seconds** across all 10 datasets on standard CPU, eliminating neural network training entirely.
  - *Key Strengths*: Fully resolved baseline level shifts (`596_YAHOO`: **0.9301** vs 0.0501 in CFS) and high-velocity spikes (`149_Stock`: **0.8900**, `716_YAHOO`: **0.8891**) without test-time velocity division.

---

## 3. Project Timeline: What We Tried and What We Learnt

Below is an overview of the nine research phases, summarising the core concept tested, the real-world outcome, and the key practical lesson:

| Phase | Core Concept | Practical Outcome | Key Lesson Learnt |
| :--- | :--- | :--- | :--- |
| **Phase 1: Baselines** | Use model prediction error (loss) to choose the best window size $L$. | The algorithm picked the smallest possible window ($L=15$) on 72% of datasets. | Raw prediction error is inherently smaller for shorter windows; it cannot be used to choose window size. |
| **Phase 2: Trajectory Smoothness** | Use trajectory smoothness (flow roughness) to choose window size without labels. | Eliminated the short-window bias on periodic data; accurately identified the natural cycle on 59% of datasets. | Measuring how smoothly reconstructed trajectories flow through space reveals the natural physical period. |
| **Phase 3: Joint Optimisation** | Attempt to optimise window size $L$ and manifold dimension $d$ simultaneously. | Distances broke down for large windows ($L > 50$); all points appeared equidistant. | High-dimensional Euclidean distance comparisons fail; representations must remain compact. |
| **Phase 4: Fourier Bases** | Project time-series history onto sine and cosine frequency modes. | Improved overall detection (+5.4%), but adding too many high-frequency modes washed out sharp spikes. | Using too many frequency coordinates dilutes sudden, localised anomalies into background noise. |
| **Phase 5: Autoencoders** | Compress the time series through a non-linear neural bottleneck autoencoder. | The autoencoder squashed anomalies straight into the normal data cluster; detection fell to coin-toss levels (50% ROC). | Non-linear neural compression distorts geometric space; linear orthogonal projections are far more reliable. |
| **Phase 6: The Centering Bug** | Discovered that the scoring function was subtracting the global dataset mean rather than the local healthy neighbour. | Correcting this single bug increased our benchmark performance ceiling from 0.48 to 0.79 VUS-PR. | Local neighbourhood comparisons are crucial; subtracting a global average ruins circular patterns. |
| **Phase 7: Trajectory Subspaces** | Compared linear trajectory decomposition (Hankel SVD) against non-negative simplex coordinates. | SVD achieved 0.88 VUS-PR on clean data. Simplex coordinates forced numbers to be positive, crushing waves. | Symmetrical oscillations require unconstrained coordinate systems; forcing positivity destroys wave geometry. |
| **Phase 8: Smooth Polynomials** | Project past history onto smooth continuous Legendre polynomials ($K=8$). | Replaced the 38-model brute-force search with a single 11-second model (0.7321 pilot VUS-PR, 0.3732 full benchmark). | Smooth polynomials decouple the observation window length from the number of model coordinates. |
| **Phase 9: State Space & Multiscale** | Tested physical spring-mass oscillator filters and multiscale combinations. | Oscillators reached 0.6070 VUS-PR on pilots. Experiment 28 resolved multi-scale noise mismatch using training-set standardisation. | A single window cannot detect both spikes and flatlines; calibrated multi-scale monitoring is essential. |
| **Phase 10: Phase-Space Pipeline** | Multi-Scale Phase-Space Pipeline with Carré du champ resolvent and null calibration. | Achieved 0.5364 VUS-PR in 5.1s on CPU. Outperformed Harmonic SSM v3 and Exp 28 without neural training. | Physical phase space $(z, \dot{z})$ avoids the velocity trap; resolvent projection eliminates artificial dimensional cliffs. |

---

## 4. Master Experiment Table (All 30 Experiments)

Every experiment conducted in this repository is recorded below with its dedicated report link and practical conclusion:

| ID | Title | Date | What We Tested | Outcome & Practical Takeaway |
| :---: | :--- | :---: | :--- | :--- |
| **01** | [Baseline NLL Pilot](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/01_baseline_and_nll_memo_sweep.md) | Sep 3 | Raw flow prediction loss for window selection | **Failed**: Raw loss is not comparable across different window lengths. |
| **02** | [Benchmark NLL Sweep](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/02_benchmark_nll_memo_deep_dive.md) | Sep 3 | 798-dataset sweep using flow loss | **Failed**: 72.3% of datasets defaulted to the smallest window ($L=15$). |
| **03** | [Trajectory Smoothness](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/03_diffusion_geometry_continuous_benchmark.md) | Sep 4 | Flow roughness on 870 datasets | **Success**: Eliminated the $L=15$ bias; reliably found natural cycles without labels. |
| **04** | [Automatic Dimension $d$](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/04_d_val_intrinsic_dimension_benchmark.md) | Sep 4 | Estimating manifold dimension dynamically | **Success**: Demonstrated that assuming a fixed dimension of 2 fails on complex data. |
| **05** | [Joint Window & Dim Search](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/05_joint_qm_coarse_fine_grid.md) | Sep 7 | Joint grid search over window $L$ and dimension $d$ | **Mixed**: Distance measures broke down in larger windows due to high dimensionality. |
| **06** | [Tangent Plane Rotation](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/06_joint_qm_grassmann_subspace.md) | Sep 7 | Measuring tangent plane rotation velocity | **Failed**: Penalised normal sharp biological transitions, such as ECG heartbeats. |
| **07** | [Early Stopping on Loss](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/07_joint_qm_genias_and_early_stopping.md) | Sep 7 | Early stopping based on flow loss | **Failed**: Stopping early prevented sharp decision boundaries from forming (0.022 VUS-PR). |
| **08** | [Compensating Window Noise](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/08_joint_qm_fix2_fix3_pilot.md) | Sep 7 | Adjusting scores for window noise | **Mixed**: Stabilised dimension estimation, but tended to overshoot to very large windows ($L=512$). |
| **09** | [Fourier Projection](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/09_joint_qm_harmonic_and_anti_dilution.md) | Sep 8 | Projecting history onto sine/cosine modes | **Success**: +5.4% improvement; established smooth window boundary weighting. |
| **10** | [Frequency Router](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/10_dynamic_harmonic_router_dharm.md) | Sep 8 | Using dominant frequencies to prune search space | **Success**: Cut the search space by 73.5% with no reduction in detection accuracy. |
| **11** | [Targeted Suite Test](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/11_focused_suite_dharm.md) | Sep 9 | 48-dataset validation run | **Success**: Surpassed 0.68 VUS-PR on targeted datasets; 0.81 VUS-PR on stochastic series. |
| **12** | [Autoencoder Compression](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/12_latent_cdc_fm_benchmark.md) | Sep 9 | Compressing data via neural bottleneck | **Failed**: The neural network squashed anomalies directly into the normal data cluster (0.50 ROC). |
| **13** | [Masked Autoencoder](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/13_himae_multiscale_benchmark.md) | Sep 9 | Multi-scale masked representation | **Partial**: Multi-scale masking helped, but was computationally much slower than geometry. |
| **14** | [Correcting Mean Subtraction](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/14_oracle_benchmark_and_bad_scorer_crisis.md) | Sep 10 | Fixing global mean subtraction bug | **Milestone**: The true benchmark performance ceiling rose from 0.4842 to 0.7922 VUS-PR. |
| **15** | [Contrast Survey (860 ds)](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/15_ground_truth_contrast_and_morphology.md) | Sep 10 | Measuring anomaly contrast across the benchmark | **Discovery**: Proved that noise accumulates in large windows across 88.6% of time series. |
| **16** | [Full 870 Benchmark Run](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/16_benchmark_870_multistream.md) | Sep 10 | Comprehensive 870-dataset comparison | **Result**: Linear trajectory projection reached 0.58 VUS-PR, outperforming neural flows. |
| **17** | [Hankel Matrix SVD Pilot](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/17_hankel_trajectory_embedding_pilot.md) | Sep 10 | Matrix decomposition on 16 datasets | **Success**: 0.88 VUS-PR with zero manual parameter tuning in 9.1 seconds. |
| **18** | [Simplex Coordinates](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/18_geosimplex_simplicial_geometry.md) | Sep 14 | Forcing coordinates to be non-negative | **Failed**: Forcing non-negative coordinates reduced anomaly VUS-PR by over 34% compared to standard linear projections. |
| **19** | [Fixed Geometric Scoring](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/19_fixed_cdc_scoring_and_benchmark_870.md) | Sep 14 | Removing artificial distance clamping | **Milestone**: 0.7816 pilot VUS-PR; 0.7052 across 150 benchmark datasets. |
| **20** | [Smooth Polynomials (FDE)](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/20_functional_continuous_delay_embedding.md) | Sep 14 | Continuous Legendre basis ($K=8$) | **Major Leap**: A single 11-second model scored 0.7321 pilot VUS-PR (0.3732 across 795 benchmark datasets). |
| **21** | [Dyadic Boxcar Windows](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/21_dyadic_multiscale_cdc.md) | Sep 14 | Block-averaging across scales | **Failed**: Discontinuous box filters fractured smooth trajectory curves. |
| **22** | [Training Duration vs Overfitting](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/22_epoch_sensitivity_and_overfitting.md) | Sep 14 | Studying epoch sensitivity | **Discovery**: Noisy traffic datasets overfit after 25 epochs; periodic signals benefit from 100+. |
| **23** | [Synthetic Anomaly Tuning](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/23_synthetic_anomaly_injection_sai.md) | Sep 14 | Tuning parameters with synthetic noise | **Success**: 0.805 VUS-PR without requiring any ground-truth labels. |
| **24** | [Harmonic Resonators v1](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/24_state_space_models_ssm_pilot.md) | Sep 14 | Physical spring-mass oscillator filters | **Progress**: Spike detection jumped from 0.009 to 0.758 VUS-PR; flatlines remained weak (0.09 VUS-PR). |
| **25** | [Harmonic Resonators v2](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/25_harmonic_ssm_v2_pilot.md) | Sep 14 | Adding bandpass filtering and dual scoring | **Progress**: Reached 0.5965 VUS-PR on pilots; flatline detection rose to 0.221 VUS-PR. |
| **26** | [Harmonic Resonators v3](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/26_harmonic_ssm_v3_pilot.md) | Sep 14 | Standardised component scoring | **Diagnostic**: Flatline detection rose to 0.3091 VUS-PR; revealed why dividing by velocity fails. |
| **27** | [Diffusion Geometry Theory](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/27_diffusion_geometry_framework.md) | Sep 15 | Reference mathematical formulation | **Theory**: Defined local covariance metrics, normal bundles, and projection geometry. |
| **28** | [Unsupervised Scale Selection](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/28_unsupervised_scale_selection.md) | Sep 15 | Testing multi-scale combinations without cheating | **Key Finding**: Standardising normal training noise fixes multi-scale false alarms (boosting pilot VUS-PR from 0.3080 to 0.4176). |
| **29** | [Visual Guide to Failure Modes](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/29_visual_guide_to_failure_modes.md) | Sep 15 | Visual plots of failure modes (ground truth vs prediction) | **Visual Guide**: Side-by-side plots of multi-scale calibration, heartbeat false alarms, velocity flaws, and short-window bias. |
| **30** | [Compressed Function Space Analysis](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/30_compressed_function_space_analysis.md) | Sep 15 | Empirical analysis of Jones & Lanners (2026) | **Evaluation & Limits**: Strong on structured data (0.9130 on NAB), but failed on Yahoo (0.0501); grid sweeps prove severe $W$ and $n_0$ sensitivity. |
| **31** | [Ideal Latent Space Design](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/31_ideal_latent_space_design.md) | Sep 15 | Conceptual design principles & trade-offs | **Theory & Trade-offs**: Plain UK English guide outlining desired latent properties, velocity/drift traps, and proposed hypotheses to test. |
| **32** | [Multi-Scale Phase-Space Pipeline](file:///home/clothibenj/Documents/git/myGit/CDC_FM_AD/docs/analysis/32_multiscale_phase_space_pipeline.md) | Sep 15 | Continuous multi-scale phase-space pipeline with resolvent projection | **New SOTA**: 0.5364 pilot VUS-PR in 5.1s CPU runtime; resolved step level shifts (0.9301) and high-velocity spikes (0.8891) without neural training or velocity division. |

---

## 5. Systematic Guide to Discovered Failure Modes (What Went Wrong & How We Fixed It)

Across our 29 experimental iterations, we identified 12 distinct failure modes. Below is a detailed, plain English explanation of each problem: what was attempted, what broke in practice, why it broke, and how we solved or worked around it.

---

### Failure Mode 1: The Short-Window Bias (Why Algorithms Naturally Pick Tiny Windows)

- **What Was Attempted**: When building an autonomous system that selects its own window length $L$, the most intuitive approach is to pick the window size that produces the lowest prediction error (loss) on normal data.
- **What Happened in Practice**: In Experiments 01 and 02 across 798 datasets, 72.3% of all time series defaulted straight to the smallest window allowed ($L=15$). The algorithm ignored the true cycles of the data (such as 200-step machine cycles) and collapsed to the lower boundary.
- **Why It Broke**: Predicting 15 time steps into the future is inherently much easier than predicting 200 time steps. A 15-step model only has to track immediate local momentum, so its total error is naturally tiny. The algorithm was not picking $L=15$ because it understood the system's dynamics; it picked $L=15$ simply because smaller windows contain less data to predict.
- **How We Fixed It**: In Experiment 03, we stopped using prediction error to choose window size. Instead, we measured **trajectory smoothness** (flow roughness). A periodic physical process traces smooth, continuous loops in state space only when the window is correctly tuned to its natural physical period.

[![Failure Mode 1: Short-Window Loss Bias vs Trajectory Smoothness](./figures/failure_mode_01_short_window_bias.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_01_short_window_bias.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_01_short_window_bias.png) | [Relative link](./figures/failure_mode_01_short_window_bias.png))*

---

### Failure Mode 2: Noise Accumulation in Large Windows (Diluting Brief Spikes)

- **What Was Attempted**: Large windows (e.g. $L \ge 256$) are necessary to capture long-term context, slow trends, and extended sensor flatlines.
- **What Happened in Practice**: In Experiment 15 across 860 datasets, we found that 88.6% of series suffered from severe signal dilution when evaluated in large windows. A sharp voltage drop (plunging from ~1684 down to 98 at $t=2518$) in dataset `054_WSD` was easily detected in a 16-point window, but suffered severe contrast dilution when evaluated in a 128-point or 256-point window.
- **Why It Broke**: If a transient defect lasts only a few time steps, placing it inside a large window means that a few points represent the anomaly while hundreds of points represent ambient noise. When an algorithm computes the total distance or error across all 128 or 256 dimensions, the transient signal gets diluted by the accumulated background noise of the surrounding dimensions.
- **How We Fixed It**: Instead of treating all raw points as independent coordinates, we project the window onto a compact set of smooth mathematical curves (Legendre polynomials or harmonic modes, $K=8$). This captures both the broad trend and sharp changes without summing up dozens of dimensions of high-frequency sensor noise (Experiments 09 and 20).
- **How Panel (d) Was Calculated**: The dilution curve is derived from the statistical signal model for a transient of length $\ell=5$ and amplitude $A=2.5$ embedded in white sensor noise ($\sigma=0.25$):
  - **Signal Energy**: $E_{\text{signal}} = \ell \cdot A^2 = 5 \times 2.5^2 = 31.25$.
  - **Uncompressed Euclidean Noise**: Ambient variance accumulates across all $(L - \ell)$ non-anomalous dimensions, giving effective noise magnitude $\sqrt{(L - 5)\sigma^2}$. Thus, $\text{SNR}_{\text{uncompressed}} = \frac{E_{\text{signal}}}{\sqrt{(L-5)\sigma^2}} \propto \frac{1}{\sqrt{L}}$ (decaying from $37.7$ at $L=16$ to $5.5$ at $L=512$).
  - **Orthogonal Subspace Noise**: Projecting onto an orthonormal $K=8$ basis leaves a noise variance of only $K\sigma^2$ (as $(L - K)$ ambient noise dimensions lie in the null space). Thus, $\text{SNR}_{\text{legendre}} = \frac{E_{\text{signal}}}{\sqrt{K\sigma^2}} \approx 44.2$ (constant for all $L$).

[![Failure Mode 2: Noise Accumulation in Large Windows](./figures/failure_mode_02_noise_accumulation_large_windows.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_02_noise_accumulation_large_windows.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_02_noise_accumulation_large_windows.png) | [Relative link](./figures/failure_mode_02_noise_accumulation_large_windows.png))*

---

### Failure Mode 3: High-Dimensional Distance Breakdown (The "All Points Look Equidistant" Problem)

- **What Was Attempted**: Detecting anomalies by measuring raw Euclidean distances between a test window and its nearest normal training window in window space $\mathbb{R}^L$ (where the embedding dimension $D$ equals the window size $L$).
- **What Happened in Practice**: In Experiment 05, when window sizes grew beyond $L > 50$ timestamps, uncompressed nearest-neighbour anomaly detection broke down. The algorithm could no longer distinguish between normal points and genuine anomalies.
- **What Relative Distance Contrast Means**: Defined as $\frac{d_{\max} - d_{\min}}{d_{\min}}$, it measures how much further away the most distant point is compared to the nearest neighbour. In compact spaces ($L=4$), contrast is $4.8$ (distant points are 480% further away, giving clear clusters). In uncompressed large windows ($L=256$), contrast collapses to $0.08$ (the nearest neighbour is barely 8% closer than the most distant point in the entire dataset!).
- **Why It Broke**: Under the mathematical concentration of measure (*Beyer et al., 1999*), adding independent ambient noise across $L$ dimensions causes pairwise Euclidean distances to tightly concentrate around $\sqrt{2 L \sigma^2}$. As a result, every point appears virtually equidistant from every other point, rendering uncompressed nearest-neighbour retrieval meaningless.
- **Why Some Algorithms Still Use $L > 50$**: Modern neural networks (CNNs, Autoencoders, Transformers) or linear models (N-BEATS, DLinear) do not evaluate raw pairwise Euclidean distances across $L$ points. They either (1) compress the $L$ inputs into a compact bottleneck ($d \ll L$), or (2) evaluate a 1-dimensional forecasting residual ($|x_{t+1} - \hat{x}_{t+1}|$). Failure Mode 3 specifically affects methods that compute raw Euclidean distances in uncompressed $\mathbb{R}^L$.
- **How We Fixed It**: Never calculate raw Euclidean distances or Gaussian kernel affinities across uncompressed time steps. Always project the window into a compact coordinate space (such as $K \le 8$ smooth polynomial coefficients, phase-space jets, or SVD modes) before evaluating distances or nearest neighbours.

[![Failure Mode 3: High-Dimensional Distance Breakdown](./figures/failure_mode_03_high_dim_distance_breakdown.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_03_high_dim_distance_breakdown.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_03_high_dim_distance_breakdown.png) | [Relative link](./figures/failure_mode_03_high_dim_distance_breakdown.png))*

---

### Failure Mode 4: Stopping Neural Training Too Early

- **What Was Attempted**: In standard deep learning, training is stopped as soon as the validation loss flattens out (early stopping) to prevent overfitting and save time.
- **What Happened in Practice**: In Experiment 07, stopping neural training when flow matching loss flattened caused anomaly detection to collapse to just 0.022 VUS-PR (2.2% AUC).
- **Why It Broke**: A flow matching neural network learns the general shape of the normal data distribution within the first 20 to 25 epochs, causing the loss curve to flatten early. However, anomaly detection relies on having steep, sharp decision boundaries right at the outermost perimeter of the normal data. Those sharp outer boundaries require 100 or more epochs to form. Stopping early leaves the boundary soft and blurry, so abnormal points get treated as normal.
- **How We Fixed It**: Do not stop training based on flat loss curves. Enforce a fixed training budget of at least 100 epochs to ensure sharp boundaries develop.

[![Failure Mode 4: Stopping Neural Training Too Early](./figures/failure_mode_04_early_stopping_boundary_blur.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_04_early_stopping_boundary_blur.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_04_early_stopping_boundary_blur.png) | [Relative link](./figures/failure_mode_04_early_stopping_boundary_blur.png))*

---

### Failure Mode 5: Autoencoders Squash Anomalies into Normal Clusters

- **What Was Attempted**: Training a neural autoencoder (with an encoder and a decoder) to compress the time series into a low-dimensional bottleneck representation.
- **What Happened in Practice**: In Experiment 12 across all 870 datasets, testing neural bottleneck autoencoders resulted in an average ROC score of 0.50 (equivalent to flipping a coin).
- **Why It Broke**: Deep neural networks are non-linear and excel at generalisation. When an autoencoder encounters an unusual anomaly at test time, its non-linear layers bend and squash the unfamiliar pattern directly into the cluster of normal points, treating it as just another variation of normal data.
- **How We Fixed It**: Discard non-linear black-box neural autoencoders for dimensionality reduction. Use linear, orthogonal projections (such as SVD or Legendre polynomials) which preserve true geometric distances and cannot squash away deviations.

[![Failure Mode 5: Autoencoders Squash Anomalies into Normal Clusters](./figures/failure_mode_05_autoencoder_squashing_anomalies.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_05_autoencoder_squashing_anomalies.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_05_autoencoder_squashing_anomalies.png) | [Relative link](./figures/failure_mode_05_autoencoder_squashing_anomalies.png))*

---

### Failure Mode 6: The Baseline Centering Bug (Subtracting the Dataset Mean)

- **What Was Attempted**: Pre-processing the time series by subtracting the overall average of the entire dataset before measuring deviations.
- **What Happened in Practice**: In Experiment 14, our model's performance ceiling appeared stuck at 0.4842 VUS-PR. Fixing this single data-centering bug caused our benchmark ceiling to immediately jump from 0.4842 to 0.7922 VUS-PR.
- **Why It Broke**: In any cyclical signal (such as an ECG heartbeat or daily temperature swing), the trajectory naturally travels far away from the overall dataset average during each cycle. If an algorithm subtracts the global mean, every normal peak looks like a massive deviation. Genuine anomalies get completely obscured by normal periodic swings.
- **How We Fixed It**: Never center cyclical data by subtracting the global dataset mean. Instead, compare each test point locally to its nearest healthy neighbouring point on the normal trajectory cycle.

[![Failure Mode 6: Global Mean Centering Bug vs Local Manifold Distance](./figures/failure_mode_06_global_mean_centering_bug.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_06_global_mean_centering_bug.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_06_global_mean_centering_bug.png) | [Relative link](./figures/failure_mode_06_global_mean_centering_bug.png))*

---

### Failure Mode 7: Forcing Non-Negative Numbers Squashes Waves

- **What Was Attempted**: Constraining model coordinates to be strictly positive (such as mapping data onto a probability simplex where numbers must be non-negative and sum to 1), hoping this would improve stability.
- **What Happened in Practice**: In Experiment 18, forcing non-negative coordinates reduced anomaly VUS-PR by over 34% compared to standard linear projections.
- **Why It Broke**: Time-series waves naturally oscillate symmetrically above and below a central baseline (positive and negative values). Forcing coordinates to be positive folds the negative half of the wave upward, crushing circular trajectory loops against the zero boundary and destroying the geometric contrast needed to spot abnormal deviations.
- **How We Fixed It**: Always use coordinate representations that naturally allow both positive and negative values (unconstrained Hilbert spaces).

[![Failure Mode 7: Forcing Non-Negative Numbers Squashes Waves](./figures/failure_mode_07_nonnegative_simplex_crushing_waves.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_07_nonnegative_simplex_crushing_waves.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_07_nonnegative_simplex_crushing_waves.png) | [Relative link](./figures/failure_mode_07_nonnegative_simplex_crushing_waves.png))*

---

### Failure Mode 8: Heavy Smoothing Destroys Sudden Spikes (Filter Ringing & Overdamping)

- **What Was Attempted**: Using simple first-order exponential moving-average filters to track time-series history smoothly.
- **What Happened in Practice**: In Experiment 24, standard first-order filters caused spike detection to collapse to between 0.002 and 0.010 VUS-PR on datasets like `Yahoo_A1` and `054_WSD`.
- **Why It Broke**: First-order filters act as strong low-pass smoothing filters. When a sharp, single-step spike occurs, the filter smooths it out across multiple future time steps, reducing its peak height by over 60% and flattening the trajectory onto a 1-dimensional line. The model literally cannot see the spike.
- **How We Fixed It**: Use second-order harmonic resonators (spring-mass systems) rather than simple exponential smoothers. These resonators track both the position and the speed (velocity) of the signal, allowing them to respond sharply to sudden impacts.

[![Failure Mode 8: Heavy Smoothing Destroys Sudden Spikes](./figures/failure_mode_08_filter_overdamping_spikes.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_08_filter_overdamping_spikes.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_08_filter_overdamping_spikes.png) | [Relative link](./figures/failure_mode_08_filter_overdamping_spikes.png))*

---

### Failure Mode 9: False Alarms on Normal Peaks (The Heartbeat Problem)

- **What Was Attempted**: Using raw geometric distance from the normal manifold to detect abnormal cardiac waveforms in medical data.
- **What Happened in Practice**: In Experiment 25 on medical dataset `234_SED`, the model achieved a high ROC score (0.94) but an abysmal precision score (0.22). Normal heartbeat peaks regularly generated anomaly scores of 3.0 to 3.3, whereas the actual flatline anomaly (where the heartbeat stopped) only scored 2.2 to 2.4. Every normal heartbeat triggered a false alarm.
- **Why It Broke**: Normal heartbeat spikes move very rapidly and deviate far from the baseline. If an algorithm measures raw geometric displacement without accounting for normal variability, every fast normal peak looks like an anomaly.
- **How We Fixed It**: Standardise anomaly scores using the normal variation observed at that point in the cycle during training. Because a heartbeat peak normally exhibits high variation, its raw deviation is divided by that expected variation, eliminating false alarms.

[![Failure Mode 9: Heartbeat False Alarms vs Baseline Standardisation](./figures/failure_mode_09_heartbeat_false_alarms.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_09_heartbeat_false_alarms.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_09_heartbeat_false_alarms.png) | [Relative link](./figures/failure_mode_09_heartbeat_false_alarms.png))*

---

### Failure Mode 10: Locking onto Tiny Vibrations Instead of the Main Cycle

- **What Was Attempted**: Using autocorrelation to identify the dominant cycle length of a machine automatically.
- **What Happened in Practice**: In Experiment 25 on engine dataset `050_WSD`, the dominant physical cycle was approximately 130 timestamps long. However, autocorrelation locked onto a tiny 10-step surface vibration ($T=10$), causing the model's filters to miss the main 130-step engine cycle entirely.
- **Why It Broke**: High-frequency ripples can produce very sharp, localized autocorrelation peaks. A naive algorithm that simply looks for the highest autocorrelation peak picks up the surface ripple rather than the broader structural cycle.
- **How We Fixed It**: Smooth the signal before cycle estimation, or look at cumulative spectral energy across frequency bands to identify the fundamental structural frequency.

[![Failure Mode 10: Locking onto Tiny Vibrations Instead of the Main Cycle](./figures/failure_mode_10_autocorrelation_surface_ripples.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_10_autocorrelation_surface_ripples.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_10_autocorrelation_surface_ripples.png) | [Relative link](./figures/failure_mode_10_autocorrelation_surface_ripples.png))*

---

### Failure Mode 11: Dividing by Signal Speed Hurts Dynamic Anomalies

- **What Was Attempted**: Dividing anomaly scores by the signal's instantaneous speed to make flatlines (which have zero speed) stand out dramatically.
- **What Happened in Practice**: In Experiment 26, dividing scores by signal speed improved flatline detection on `234_SED` (from 0.3091 to 0.4571 VUS-PR). However, it destroyed performance across dynamic anomalies: spike detection on `054_WSD` crashed from 0.7587 to 0.4132 VUS-PR, ECG anomaly detection on `303_UCR` crashed from 0.8235 to 0.3721 VUS-PR, and trend changes on `004_NAB` crashed from 0.6025 to 0.2371 VUS-PR.
- **Why It Broke**: A flatline has zero speed, so dividing by speed makes the score shoot towards infinity (highlighting the flatline). But genuine spikes and abnormal heartbeats move extremely fast! Dividing their large displacement by their high velocity shrinks their anomaly score back down to near zero, making dangerous anomalies look completely normal.
- **How We Fixed It**: Never divide anomaly scores by instantaneous test-time velocity. Instead, use **Training Dispersion Standardisation**: normalise each scoring component by its standard deviation measured across the normal training set ($s / \sigma_{\text{train}}$). This balances different score components without penalising fast signals.

[![Failure Mode 11: Dividing by Signal Speed vs Training Dispersion Standardisation](./figures/failure_mode_11_velocity_division_flaw.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_11_velocity_division_flaw.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_11_velocity_division_flaw.png) | [Relative link](./figures/failure_mode_11_velocity_division_flaw.png))*

---

### Failure Mode 12: Scale Noise Mismatch in Multiscale Models

- **What Was Attempted**: Combining multiple observation windows ($W \in \{16, 64, 256\}$) simultaneously to catch both short spikes and extended flatlines.
- **What Happened in Practice**: In Experiment 28, taking the raw maximum score across windows resulted in a dismal 0.3080 average VUS-PR. The 256-point window generated continuous false alarms on datasets with short spikes (`050_WSD` collapsed from 0.2104 to 0.0068 VUS-PR, and `054_WSD` collapsed from 0.2166 to 0.0268 VUS-PR).
- **Why It Broke**: A 256-point window spans 16 times as much data as a 16-point window. Natural background drift and trajectory variance in the 256-point window produce raw anomaly scores that are orders of magnitude larger than those of the 16-point window. If you simply take $\max(s_{16}, s_{64}, s_{256})$, the 256-point window wins on almost every time step, drowning out spikes.
- **How We Fixed It**: **Training Null Standardisation**. Before comparing or combining scores across different window sizes, standardise each window's score using its mean and standard deviation on normal training data:
  $$z_t^{(W)} = \frac{s_t^{(W)} - \mu_0^{(W)}}{\sigma_0^{(W)}}$$
  On normal data, every window now has an average score of 0 and a spread of 1. When a spike hits, the $W=16$ window shoots up to $z=15$, while the $W=256$ window stays quiet at $z=1$. Taking the maximum of the standardised scores immediately boosted pilot VUS-PR from 0.3080 to 0.4176 without cheating or tuning (and on flatlines reached 0.9848 VUS-PR).

[![Failure Mode 12: Multiscale Scale Mismatch vs Null-Standardisation](./figures/failure_mode_12_multiscale_calibration.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_12_multiscale_calibration.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/failure_mode_12_multiscale_calibration.png) | [Relative link](./figures/failure_mode_12_multiscale_calibration.png))*

---

## 6. Empirical Analysis of the Compressed Function Space (Jones & Lanners 2026)

To rigorously evaluate whether the **Compressed Function Space** $\mathcal{A} = \text{Span}\{\phi_1, \dots, \phi_{n_0}\}$ (Section 3.2.1 of Jones & Lanners 2026) provides a viable foundation for a learned representation (**Pathway B**), we evaluated it across 6 diverse benchmark datasets spanning stochastic noise, volatility shifts, impulse spikes, and semi-periodic dynamics.

The full mathematical report and grid ablation are documented in [`docs/analysis/30_compressed_function_space_analysis.md`](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/30_compressed_function_space_analysis.md).

### Objective Performance Summary & Real-World Limitations

| Dataset ID | Dynamic Category & Setting | Window $W$ | Subspace $n_0$ | Normal Error | Anomaly Error | Separation Ratio | Raw VUS-PR | Raw AUC-ROC | Empirical Verdict / Limitation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`001_NAB`** | **Stochastic Telemetry + Diurnal** (AWS Server) | 70 | 16 | 43.95 | 315.14 | **7.17×** | **0.9130** | 0.9754 | **Strong at oracle $W=70$**, but collapses to $0.3731$ if $W=16$. |
| **`303_UCR`** | **Complex Medical Waveform** (ECG Arrhythmia) | 80 | 16 | 2.69 | 18.61 | **6.92×** | **0.5614** | 1.0000 | Good point separation, but moderate range overlap. |
| **`050_WSD`** | **Harmonic Cycles + Noise** (Engine Vibration) | 64 | 16 | 1.28 | 28.53 | **22.35×** | **0.3017** | 0.9999 | **AUC-ROC illusion**: 0.9999 AUC masks poor 0.3017 VUS-PR. |
| **`054_WSD`** | **Transient Impulse** (Voltage Spike) | 16 | 8 | 0.57 | 10.58 | **18.61×** | **0.2369** | 0.9891 | **Weak**: Far below physical resonator baseline ($0.7580$). |
| **`149_Stock`**| **Financial Volatility** (Stock Return Jump) | 50 | 12 | 42.93 | 59.34 | **1.38×** | **0.7614** | 0.6683 | **Poor separation (1.38×)**; heavy overlap with normal noise. |
| **`596_YAHOO`**| **Stochastic Web Traffic** (Server Request Shifts) | 24 | 8 | 209.87 | 345.78 | **1.65×** | **0.0501** | 0.6897 | **Critical Failure (0.0501 VUS-PR)** due to level shifts. |

---

### Empirical Proof of Sensitivity: $W$ and $n_0$ Grid Sweeps

A systematic grid search reveals that the compressed function space **does not eliminate hyperparameter sensitivity**:

1. **Catastrophic $W$ Sensitivity on `001_NAB`**:
   - At oracle $W=70$, VUS-PR is **0.9269**.
   - Reducing window size to $W=16$ causes VUS-PR to collapse to **0.3731** (a $2.5\times$ drop). The method is crippled if the observation horizon is mismatched.
2. **Dual $W$ and $n_0$ Fragility on `054_WSD`**:
   - Setting $n_0 = 4$ causes VUS-PR to collapse to **0.0834** (underfitting normal dynamics).
   - Increasing window size to $W=128$ causes VUS-PR to drop to **0.0519** (ambient noise drowning the brief 5-step spike).
   - Even at the optimal grid point ($W=32, n_0=32$), VUS-PR reaches only **0.3397**, less than half the performance of our 2nd-order resonator ($0.7580$).
3. **The Oracle Confound**: The initial headline numbers relied on hand-picked oracle window lengths. In an autonomous setting, $W$ is unknown.

---

### Diagnostic Visual Gallery: Compressed Function Space

#### 1. Stochastic Sensor Telemetry (`001_NAB`)
[![001_NAB Compressed Space](./figures/compressed_space_001_NAB.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_001_NAB.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_001_NAB.png) | [Relative link](./figures/compressed_space_001_NAB.png))*

- **Analysis**: At $W=70$, the compressed function space filters white noise while tracking the smooth diurnal temperature cycle (separation ratio $7.17\times$). However, selecting $W=16$ collapses VUS-PR from $0.9130$ to $0.3731$.

---

#### 2. Complex Medical Waveform (`303_UCR`)
[![303_UCR Compressed Space](./figures/compressed_space_303_UCR.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_303_UCR.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_303_UCR.png) | [Relative link](./figures/compressed_space_303_UCR.png))*

- **Analysis**: Normal heartbeats form a clean limit cycle in $(\phi_2, \phi_3)$ with near-zero baseline error ($2.69$). Ectopic beats deviate clearly from the ring ($1.0000$ AUC-ROC). However, range-based VUS-PR is $0.5614$, showing that sharp QRS boundaries still create score transitions.

---

#### 3. Engine Vibration with Harmonic Noise (`050_WSD`)
[![050_WSD Compressed Space](./figures/compressed_space_050_WSD.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_050_WSD.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_050_WSD.png) | [Relative link](./figures/compressed_space_050_WSD.png))*

- **Analysis**: The separation ratio is large ($22.35\times$) and AUC-ROC is $0.9999$. However, VUS-PR is only **$0.3017$**. The residual score fluctuates along the multi-harmonic engine cycle, producing intermittent dips inside the anomaly range that penalise VUS-PR.

---

#### 4. Transient Power Grid Voltage Spike (`054_WSD`)
[![054_WSD Compressed Space](./figures/compressed_space_054_WSD.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_054_WSD.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_054_WSD.png) | [Relative link](./figures/compressed_space_054_WSD.png))*

- **Analysis**: Low-frequency eigenfunctions cannot reconstruct sharp 5-step impulse spikes. While this yields a $18.61\times$ separation ratio, the resulting VUS-PR ($0.2369$) is poor compared to physical dynamic models ($0.7580$).

---

#### 5. Financial Stock Price Volatility (`149_Stock`)
[![149_Stock Compressed Space](./figures/compressed_space_149_Stock.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_149_Stock.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_149_Stock.png) | [Relative link](./figures/compressed_space_149_Stock.png))*

- **Analysis**: Stock return volatility exhibits heavy non-Gaussian tails. The separation ratio is only **$1.38\times$** (normal error $42.93$ vs anomaly error $59.34$), demonstrating that financial stochastic noise is poorly separated by static graph Laplacians.

---

#### 6. Stochastic Web Server Traffic (`596_YAHOO`)
[![596_YAHOO Compressed Space](./figures/compressed_space_596_YAHOO.png)](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_596_YAHOO.png)
*([Open full-resolution image](file:///home/clothibenj/Documents/git/PhD-Research-Log/Weekly_Logs/Week_11/figures/compressed_space_596_YAHOO.png) | [Relative link](./figures/compressed_space_596_YAHOO.png))*

- **Analysis**: **Complete failure (0.0501 VUS-PR)**. The Yahoo traffic contains non-stationary baseline shifts. Because the training kernel has no support in the shifted state space, test-time Nyström projection degenerate completely.

---

## 7. Where We Stand Now & Implications for Pathway B

Our objective analysis shows that while Diffusion Geometry provides a principled functional framework, **a naive fixed-window compressed function space inherits the exact same scale ($W$) and capacity ($n_0$) sensitivities**:
1. It fails on non-stationary shifts (`596_YAHOO`, $0.0501$ VUS-PR).
2. It collapses if $W$ is mismatched ($0.9269 \to 0.3731$ on `001_NAB`).
3. It underperforms dynamic physical resonators on transient spikes ($0.2369$ vs $0.7580$ on `054_WSD`).

### Required Architecture for Pathway B:
To prevent a learned parametric encoder $f_\theta$ from merely memorising these fixed-window vulnerabilities:
- **Multi-Scale Invariance**: The encoder must ingest multi-scale dilated receptive fields rather than a single fixed $W$.
- **Null-Standardised Calibration**: Outputs must be calibrated using normal training dispersion ($z = (s - \mu_0)/\sigma_0$) as proven in Experiment 28.
- **Dynamic State Integration**: For transient shocks, the latent space must track phase-space velocity (e.g. 2nd-order resonator states) rather than relying exclusively on static window distances.
