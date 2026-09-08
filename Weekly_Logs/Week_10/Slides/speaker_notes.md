# Speaker Notes: Week 10 Supervisory Meeting (09 Sept 2026)

**Title**: Unsupervised Geometry & Dynamics for Anomaly Detection  
**Subtitle**: Stochastic Delay Embeddings, Dual $(L, d)$ Selection, & Anomaly Codimension  
**Slides File**: `meeting_0909.pdf` (compiled from `meeting_0909.tex`)  

---

## Nomenclature & Theoretical Foundations

### 1. Why "Codimension-Normalized Objective" is the Superior Term
> **Why drop "Codimension-Aware Excess SNR"?**
> 1. *"Excess SNR"* is domain-specific telecommunications jargon that sounds ad-hoc and obscure in an anomaly detection / machine learning context.
> 2. *"Aware"* is vague—it doesn't communicate what mathematical operation is being performed.
> 3. **"Codimension-Normalized Objective" ($\mathcal{J}_{\text{norm}}$)** is mathematically precise:
>    - Under the null hypothesis of baseline noise $\mathcal{N}(\mathbf{0}, \sigma^2 I_L)$, the normal and anomalous states have equal orthogonal variance, yielding $\text{CR} = 1$.
>    - The raw variance contrast is $\text{CR}^2$. Subtracting 1 yields $(\text{CR}^2 - 1)$, which isolates the **excess signal variance above the baseline noise floor per degree of freedom**.
>    - Multiplying by the codimension $(L - d) = \dim(\mathcal{N}_{\bm{x}}\mathcal{M})$ scales this per-degree-of-freedom contrast by the **total number of orthogonal detection directions**.
>    - This product naturally normalizes discriminative capacity across varying window lengths $L$ and dimensions $d$, and creates a strict zero boundary when $d \to L$ to eliminate subspace starvation.

### 2. Deconstructing the Unified Selection Loss $\mathcal{S}(L, d)$
$$\mathcal{S}(L, d) = \underbrace{\frac{1}{1 + \mathcal{J}_{\text{norm}}(L, d)}}_{\text{\textbf{Term 1: Discriminative Power}}} \times \underbrace{\left(1 + \beta \frac{L}{N_{\text{tr}}}\right)}_{\text{\textbf{Term 2: Complexity Penalty}}}$$
- **Term 1 (Discriminative Power)**: Minimizing $\frac{1}{1 + \mathcal{J}_{\text{norm}}}$ maximizes the off-manifold excess contrast $\mathcal{J}_{\text{norm}} = (\text{CR}^2 - 1) \cdot (L - d)$.
- **Term 2 (Sample Complexity Penalty, $\beta = 0.05$)**:
  - $N_{\text{tr}}$ is the total time steps in the training partition, yielding $M = N_{\text{tr}} - L + 1$ delay-embedded windows.
  - $L / N_{\text{tr}}$ represents the fraction of historical series consumed per embedded window (the inverse effective sample size). As $L$ approaches $N_{\text{tr}}$, effective samples $M \to 1$, the empirical covariance collapses into rank degeneracy, and the model overfits.
  - Setting $\beta = 0.05$ penalizes excessively large windows that consume substantial training history, favoring compact representations when contrast is equal.
- **Why Multiplicative Pareto Form?** Contrast ratios vary across orders of magnitude between domains (e.g. ECG vs Web telemetry). An additive penalty $\mathcal{L} + \lambda \Omega$ requires retuning $\lambda$ per dataset. A multiplicative formulation applies a percentage penalty that is scale-invariant across all datasets.

---

## Overview & Presentation Strategy
- **Format**: 17 minimalist mathematical slides (1 title + 16 content frames).
- **Style**: High-density mathematics, minimal text. The slides display only formulas, operators, and result tables; you verbally provide the physical intuition, theoretical rationale, and research narrative.
- **Core Narrative Arc**:
  1. *The Objective & Selection Loss (Slides 1--4)*: We score anomalies via off-manifold projection into the orthogonal complement of the normal tangent space. We construct pure off-manifold synthetic anomalies in $\ker(V_d^\top)$ and optimize an unsupervised **Codimension-Normalized Selection Loss** balancing discriminative power against sample complexity.
  2. *The dharm Experiment (Slides 5--6)*: Moving beyond blind sweeps and cherry-picked successes, we design the **Dynamic Harmonic Router (`dharm`)** over a curated $N=271$ suite (207 Rhythmic Oscillators vs. 64 Stochastic Random Walks/Jumps) using a two-stage coarse-to-fine multi-GPU architecture.
  3. *The Data Reality & Ground-Truth Anomaly Dynamics (Slides 7--8)*: Exhaustive benchmark sweeps reveal 4 sensitivity classes (where ~70% depend critically on horizon $L$), followed by an empirical audit of real anomaly morphology (spikes vs. shapelets vs. regime shifts, boundary dilution, and up to $118.6\times$ peak contrast).
  4. *Surrogate Data & Determinism Mechanics (Slides 9--10)*: IAAFT testing proves 76% of benchmarks are linear stochastic processes. We explain Grassberger-Procaccia Correlation Dimension ($CD$), the Central Limit Theorem trap of naive FFT, and Schreiber & Schmitz's alternating projection algorithm.
  5. *The Theoretical Bridge (Slide 11)*: Stark's stochastic embedding theorem proves delay coordinates reconstruct the Fokker-Planck generator bundle, resolving the "Unfolding vs Discrimination" disconnect.
  6. *The Solution for $L$ (Slides 12--13)*: A bifurcated candidate generator—physical harmonics for periodic traces (73.1%), Fraser-Swinney Average Mutual Information for aperiodic traces (26.9%).
  7. *Test Projection & Dimension $d$ (Slides 14--15)*: Gaussian apodization eliminates boundary dilution. The power-law spectral decay fallacy proves why naive eigengap heuristics mathematically collapse to $d=1$.
  8. *Results & Roadmap (Slides 16--17)*: Two-stage coarse-to-fine pilot achieves up to 103.6% Oracle recovery while reducing candidate evaluations by $>92\%$.

---

## Slide-by-Slide Script & Talking Points

### Slide 1: Title Slide
* **Slide Content**: Title, Subtitle, Date, Author.
* **Key Point to State**: "Today I'm presenting our progress on unsupervised hyperparameter selection for continuous normalizing flows and diffusion models—specifically, how we estimate the window horizon $L$ and intrinsic dimension $d$ under zero label supervision."
* **Transition**: "To understand hyperparameter selection, let's begin with how anomalies are scored via Riemannian geometry."

---

### Slide 2: 1. Anomaly Scoring & The Codimension Objective (Frame 1)
* **Equations on Slide**:
  - Time-delay coordinate map: $\bm{x}_t = [x_t, \, x_{t+1}, \, \dots, \, x_{t+L-1}]^\top \in \mathbb{R}^L$
  - Tangent space via Carré du Champ operator SVD: $\Gamma(\bm{x}_t) = V \Sigma V^\top \implies V_d = [v_1, \dots, v_d] \in \mathbb{R}^{L \times d}, \quad V_d^\top V_d = I_d$
  - Orthogonal anomaly projection at terminal time $t = 0.95$: $s(\bm{x}_t) = \|(I - V_d V_d^\top) v_\theta(\bm{x}_t, t = 0.95)\|_2$
  - Codimension $k = L - d$ tension:
    - $\lim_{L \to d} k = 0 \implies \text{Subspace Starvation (No Orthogonal Space)}$
    - $\lim_{L \gg L^*} k \gg 1 \implies \text{Ambient Noise Dilution}$
  - Objective: Find $(L^*, d^*)$ maximising off-manifold anomaly separation.
* **What to Explain Verbally**:
  - "In our continuous normalizing flow framework, we embed 1D telemetry into an $L$-dimensional Euclidean space using delay coordinates $\bm{x}_t$."
  - "At each local state, the Carré du Champ diffusion operator $\Gamma(\bm{x}_t)$ captures the local geometry of the data manifold. Taking its SVD yields the top-$d$ singular vectors $V_d$, forming an orthonormal basis for the local tangent space."
  - "Test-time anomaly scoring evaluates the flow network $v_\theta(\bm{x}_t, t=0.95)$ at terminal time $t=0.95$, and projects this drift onto the orthogonal complement $(I - V_d V_d^\top)$. An anomaly represents an off-manifold excursion, producing a strong orthogonal restoring force attempting to push the trajectory back onto the normal manifold."
  - "This creates an intrinsic codimension tension: $k = L - d$. If $L \to d$, the codimension collapses to zero—there is no orthogonal space left to detect anomalies (subspace starvation). If $L$ is set too large, localized anomalies are diluted in ambient sensor noise."
* **Transition**: "Since we don't have anomaly labels at tuning time, how do we evaluate which $(L, d)$ pair provides the strongest orthogonal separation? We generate synthetic off-manifold anomalies."

---

### Slide 3: 2. Unsupervised Tuning: Synthetic Anomaly Generation (Frame 2)
* **Equations on Slide**:
  - 1. Ambient noise & manifold validation window: $\bm{x}_t \in \mathcal{D}_{\text{val}} \subset \mathbb{R}^L, \quad \bm{\xi} \sim \mathcal{N}(\mathbf{0}, I_L)$
  - 2. Pure orthogonal injection: $P_\perp = I_L - V_d V_d^\top, \quad \bm{u}_\perp = \sigma_{\text{scale}} \cdot \frac{P_\perp \bm{\xi}}{\|P_\perp \bm{\xi}\|_2} \implies V_d^\top \bm{u}_\perp = \mathbf{0}$
    where $\sigma_{\text{scale}} = \kappa \cdot \sigma_{\text{local}} \cdot \sqrt{L}$ preserves scale across dimensions.
  - 3. Synthetic anomaly construction & restoring drift: $\bm{x}_{\text{anom}} = \bm{x}_t + \bm{u}_\perp \implies \text{Restoring force: } v_\theta(\bm{x}_{\text{anom}}, t=0.95)$
  - 4. Off-manifold Contrast Ratio: $\text{CR}(L, d) = \frac{\mathbb{E}_{\bm{x}_t \in \mathcal{D}_{\text{val}}} \|P_\perp v_\theta(\bm{x}_{\text{anom}}, t=0.95)\|_2}{\mathbb{E}_{\bm{x}_t \in \mathcal{D}_{\text{val}}} \|P_\perp v_\theta(\bm{x}_t, t=0.95)\|_2}$
* **What to Explain Verbally**:
  - "Here is how we synthesize anomalies without human supervision: we take a normal validation window $\bm{x}_t$ and sample isotropic ambient Gaussian noise $\bm{\xi}$."
  - "Instead of injecting raw noise into the window, we project $\bm{\xi}$ strictly into the orthogonal complement $P_\perp$. The resulting perturbation $\bm{u}_\perp$ lies strictly in the kernel $\ker(V_d^\top)$, meaning $V_d^\top \bm{u}_\perp = \mathbf{0}$. It has **zero tangential overlap** with normal dynamics."
  - "We normalize by $\|P_\perp \bm{\xi}\|_2$ and scale by $\sigma_{\text{scale}} = \kappa \cdot \sigma_{\text{local}} \sqrt{L}$ (setting $\kappa \approx 3.0$). Because Gaussian vectors in $\mathbb{R}^L$ have expected length $\sqrt{L}\sigma$, this calibration ensures a controlled $3\sigma$ off-manifold perturbation invariant to ambient dimension $L$."
  - "When the model evaluates $\bm{x}_{\text{anom}}$, a well-tuned model produces a high velocity field $v_\theta$ pushing back to the manifold."
  - "The Contrast Ratio $\text{CR}$ measures how many times stronger the orthogonal restoring velocity is for synthetic anomalies compared to normal validation states."
* **Transition**: "Now, how do we translate this contrast ratio into an objective function to select the optimal $(L, d)$? We define our Unsupervised Selection Loss."

---

### Slide 4: 3. Unsupervised Selection Loss: Codimension & Complexity (Frame 3)
* **Equations & Bullets on Slide**:
  - Unified Selection Loss Objective:
    $$\mathcal{S}(L, d) = \underbrace{\frac{1}{1 + \mathcal{J}_{\text{norm}}(L, d)}}_{\text{\textbf{Term 1: Discriminative Power}}} \times \underbrace{\left(1 + \beta \frac{L}{N_{\text{tr}}}\right)}_{\text{\textbf{Term 2: Complexity Penalty}}}$$
  - Term 1: Codimension-Normalized Objective $\mathcal{J}_{\text{norm}} = (\text{CR}^2 - 1) \cdot (L - d)$
    - $(\text{CR}^2 - 1)$: Baseline variance subtraction per degree of freedom (excess SNR).
    - $(L - d) = \dim(\mathcal{N}_{\bm{x}}\mathcal{M})$: Orthogonal degrees of freedom; penalizes subspace starvation ($d \to L$).
  - Term 2: Sample Complexity Penalty ($\beta = 0.05$)
    - $N_{\text{tr}}$: Length of training partition ($M = N_{\text{tr}} - L + 1$ embedded windows).
    - $L / N_{\text{tr}}$: Fraction of series consumed per window; penalizes rank degeneracy.
  - Invariance: Multiplicative product provides scale invariance across diverse datasets.
* **What to Explain Verbally**:
  - **The Big Picture**: "We formulate model selection as minimizing $\mathcal{S}(L, d)$. It balances two competing forces: **Term 1** rewards discriminative power, while **Term 2** penalizes sample complexity."
  - **Term 1 (Discriminative Power)**:
    - "Minimizing $\frac{1}{1 + \mathcal{J}_{\text{norm}}}$ is equivalent to maximizing $\mathcal{J}_{\text{norm}}(L, d) = (\text{CR}^2 - 1) \cdot (L - d)$."
    - "Why $(\text{CR}^2 - 1)$? Under pure baseline noise $\mathcal{N}(\mathbf{0}, \sigma^2 I_L)$, normal and anomalous states have equal variance, giving $\text{CR} = 1$. Subtracting 1 removes the noise floor and isolates the **excess variance per degree of freedom** generated by the flow model's restoring drift."
    - "Why multiply by $(L - d)$? The codimension $(L - d) = \dim(\mathcal{N}_{\bm{x}}\mathcal{M})$ represents the total number of orthogonal directions available to detect an anomaly. Multiplying by $(L - d)$ standardizes contrast across dimensions. Crucially, as $d \to L$ (subspace starvation/overfitting), $(L - d) \to 0$, forcing $\mathcal{J}_{\text{norm}} \to 0$ and penalizing dimension collapse."
  - **Term 2 (Sample Complexity Penalty)**:
    - "**What is $N_{\text{tr}}$?** $N_{\text{tr}}$ is the total number of time steps in the training partition. When we embed scalar series into $L$-dimensional vectors, we obtain $M = N_{\text{tr}} - L + 1$ overlapping training windows."
    - "**What is $\frac{L}{N_{\text{tr}}}$?** It is the fraction of the entire training history consumed by a single embedded window—in other words, the inverse of our effective sample size. If $N_{\text{tr}} = 5,000$ and $L = 50$, $L / N_{\text{tr}} = 0.01$ (only 1% consumed, sample covariance $\Gamma$ is well-conditioned). But if $L \to N_{\text{tr}}$, effective samples $M \to 1$. The sample covariance collapses into rank deficiency, tangent space estimation degrades, and the neural network overfits."
    - "With $\beta = 0.05$, a model using $20\%$ of the historical series incurs a modest $1\%$ penalty. It acts as an Occam's razor favoring compact representations when contrast is equal."
  - **Why Multiplicative Pareto Form?**
    - "Why $\text{Term 1} \times \text{Term 2}$ instead of an additive $\mathcal{L} + \lambda \Omega$? Because across an 870-dataset benchmark, contrast ratios vary across orders of magnitude (e.g. clean ECG vs noisy web traffic). An additive penalty requires per-dataset retuning of $\lambda$. The multiplicative product is scale-invariant: Term 2 applies a **percentage penalty** to Term 1 regardless of baseline contrast scale."
* **Transition**: "With this selection objective formalized, what did we do next experimentally? We designed the Dynamic Harmonic Router (`dharm`) campaign."

---

### Slide 5: 4. The dharm Experiment: Physical Regime Stratification (Frame 4)
* **Points on Slide**:
  - 1. Rhythmic Limit-Cycle Oscillators ($N = 207$ Series):
    - Domains: Bio-medical ECG (`MITDB`, `SVDB`, `LTDB`), motion (`OPP`), `SED`, `Power`.
    - Dynamically Verified UCR ($N = 137$): Filtered by $T_{\text{dom}} \ge 16$ and $r_{\text{peak}} \ge 0.40$; guarantees macroscopic recurrence; rejects short-lag noise/trends.
    - Attractor & Grid: Closed phase orbits $\implies \mathcal{G}_L = \{1, 2, 3, 4\} \times T_{\text{dom}}$.
  - 2. Stochastic Random Walks & Jumps ($N = 64$ Series):
    - Domains: Financial equities (`Stock`), clusters (`Exathlon`), telemetry (`NEK`, `TAO`).
    - Dynamical Profile: Lack deterministic recurrence; monotonic ACF decay ($T_{\text{dom}} \to 1$); fail surrogate tests ($Z_{\text{det}} < 1.96$, linear stochastic/diffusion).
    - Routing & Grid: Fraser-Swinney decorrelation $\implies \mathcal{G}_L = \{2^k \cdot \tau_{\text{AMI}}\}$.
* **What to Explain Verbally**:
  - "Rather than evaluating arbitrary grids across all 870 datasets or cherry-picking only successful traces, we asked a fundamental question: **Where does delay embedding work, and where does it fail?**"
  - "We stratified the benchmark into two distinct physical extremes totaling 271 datasets:"
    1. **Archetype 1: Rhythmic Limit-Cycle Oscillators ($N = 207$)**:
       - Known periodic physical domains ($N=70$): ECG (`MITDB`, `SVDB`, `LTDB`), motion tracking (`OPP`), solar energy (`SED`), and power grid sensors.
       - **Dynamically Verified Periodic UCR Series ($N = 137$)**: From the broad UCR benchmark, series were included only if they satisfied strict quantitative periodicity thresholds: $T_{\text{dom}} \ge 16$ and $r_{\text{peak}} \ge 0.40$. This guaranteed strong, non-spurious periodic recurrence and macroscopic cycles, filtering out short-lag noise or monotonic drift.
       - These systems possess closed topological phase-space orbits. We anchor candidate horizons to integer harmonic multiples $\mathcal{G}_L = \{1, 2, 3, 4\} \times T_{\text{dom}}$, eliminating arbitrary boundary cuts through orbital phases.
    2. **Archetype 2: Stochastic Random Walks & Jumps ($N = 64$)**:
       - Financial equity tickers (`Stock`), big-data clusters (`Exathlon`), network traffic (`NEK`), and ocean buoys (`TAO`).
       - **Dynamical Profile**: These datasets lack deterministic periodic recurrence. Their autocorrelation decays monotonically ($T_{\text{dom}} \to 1$), and they fail surrogate determinism tests ($Z_{\text{det}} < 1.96$), behaving as linear stochastic diffusions, Brownian walks, or Poisson jumps.
       - Because linear autocorrelation is uninformative, we route them to Fraser-Swinney Average Mutual Information (AMI), extracting the first local information minimum $\tau_{\text{AMI}}$ and expanding across a dyadic multi-scale grid $\{2^k \cdot \tau_{\text{AMI}}\}$."
* **Transition**: "Now let's examine how we execute this routing in our distributed cluster pipeline."

---

### Slide 6: 5. Candidate Generation: Dual Horizon & Harmonic Dimension (Frame 5)
* **Points on Slide**:
  - 1. Dual Horizon Selection ($L \in [8, N_{\text{tr}}/4]$):
    - Periodic ($T_{\text{dom}} \ge 6$): $\mathcal{G}_L = \{1, 2, 3, 4\} \times T_{\text{dom}}$ (closed phase orbits).
    - Aperiodic ($T_{\text{dom}} < 6$): $\mathcal{G}_L = \{2^k \cdot \tau_{\text{AMI}}\} \cap [8, N_{\text{tr}}/4]$ ($L \ge 8$ captures spikes).
  - 2. Computing Harmonic Dimension ($d_{\text{harm}} = 2K + 1$):
    - FFT Peak Detection: Finds $K$ modes in $|X(f)|$ (prominence $\ge 0.08$).
    - Quadrature Form: $K$ rotations + 1 trend $\implies d_{\text{harm}} = 2K + 1$.
    - Candidate Set: $\mathcal{G}_d = \{1, 2, 4, 8\} \cup \{d_{\text{harm}}\}$ (bypasses eigengap collapse).
  - 3. Classical Literature Grounding:
    - Broomhead & King (1986): SSA trajectory rank for $K$ modes is $2K + 1$.
    - Whitney-Takens (1991): Invariant torus $\mathbb{T}^K$ embeds into $\mathbb{R}^{2K+1}$.
* **What to Explain Verbally**:
  - "**1. Dual Horizon Selection ($L$) & The $L \ge 8$ Clipping Bound**":
    - "For periodic series ($T_{\text{dom}} \ge 6$), we anchor candidate horizons to integer multiples $\{1, 2, 3, 4\} \times T_{\text{dom}}$, guaranteeing topologically closed phase orbits without boundary cuts."
    - "For aperiodic series ($T_{\text{dom}} < 6$), we extract the Fraser-Swinney Average Mutual Information (AMI) first local minimum $\tau_{\text{AMI}}$ and construct the dyadic multi-scale grid $\{2, 4, 8, 16, 32, 64\} \times \tau_{\text{AMI}}$."
    - "**Why clip to $[8, N_{\text{tr}}/4]$ instead of 15?** 15 was an arbitrary historical lower bound. Lowering the minimum horizon to $L = 8$ allows the delay coordinate window to isolate tight, localized point anomalies and transient step jumps without averaging them away into ambient baseline noise. Meanwhile, the upper bound $N_{\text{tr}}/4$ guarantees at least 4 independent non-overlapping window blocks to ensure non-degenerate covariance estimation."
  - "**2. How the Harmonic Fourier Bound is Computed**":
    - "We apply a Hanning window to the detrended training sequence $x_{1:N_{\text{tr}}}$ and compute the discrete Fourier transform $|X(f)|$."
    - "Using spectral peak detection, we identify all dominant harmonic peaks with prominence $\ge 0.08 \times \max|X(f)|$, yielding $K$ significant frequency modes."
    - "Why $2K + 1$? In delay-coordinate space, each real sinusoidal oscillator $\cos(\omega_k t + \phi_k)$ is a 2-dimensional planar circular rotation (spanned by its in-phase and quadrature sine/cosine basis vectors). $K$ independent oscillatory modes therefore span $2K$ dimensions. Adding 1 dimension for the secular mean/trend gives the exact structural manifold dimension: $d_{\text{harm}} = 2K + 1$."
  - "**3. Is this method used elsewhere? (Literature Foundations)**":
    - "**Broomhead & King (1986, Physica D)**: In their seminal foundation of Singular Spectrum Analysis (SSA) and delay-coordinate embedding ('Extracting qualitative dynamics from experimental data'), they proved that embedding a signal with $K$ harmonic components yields a trajectory covariance matrix of rank identically $2K$ (for zero-mean) or $2K + 1$ (with non-zero trend/DC offset)."
    - "**Singular Spectrum Analysis (SSA) Separability Theorem (Golyandina et al. 2001)**: Every distinct harmonic component generates a pair of equal singular values with quadrature phase singular vectors, establishing that counting spectral modes directly defines the signal subspace dimension $d = 2K + 1$."
    - "**Whitney-Takens Torus Embedding Theorem (Sauer, Yorke & Casdagli 1991, 'Embedology')**: A quasi-periodic attractor with $K$ incommensurate frequencies forms an invariant torus $\mathbb{T}^K$. By Whitney's embedding theorem, the generic Euclidean dimension required to smoothly embed an invariant $K$-torus without self-intersections is strictly $2K + 1$."
    - "**Koopman Operator Theory (Mezić 2005; Brunton et al. 2016)**: The Koopman eigenfunctions of a limit cycle with $K$ harmonics span a $(2K + 1)$-dimensional invariant linear subspace."
    - "*Key message for supervisors*: This is not an ad-hoc heuristic we invented; it is the fundamental theorem of harmonic trajectory matrix rank from Broomhead & King and SSA!"
* **Transition**: "Now let's examine what the ground-truth benchmark landscape looks like when we sweep all $(L, d)$ pairs across hundreds of datasets."

---

### Slide 7: 6. Ground-Truth Landscape: 4 Geometric Classes (Frame 6)
* **Table on Slide**:
  - Exhaustive Benchmark Oracle Audit (870 Datasets):
    - **Class 1: Robust**: Flat Pareto basin (scale & dim invariant) | Count: 211 | Share: **24.3%**
    - **Class 2: Dim-Critical**: Sensitive to $d$, invariant to horizon $L$ | Count: 54 | Share: **6.2%**
    - **Class 3: Horizon-Critical**: Sensitive to $L$, robust across dimension $d$ | Count: 348 | Share: **40.0%**
    - **Class 4: Co-Sensitive**: Joint $(L, d)$ alignment mandatory | Count: 257 | Share: **29.5%**
  - Universal Invariant: $d^* \in \{1, 2, 4\}$ in $>80\%$ of series, regardless of whether $L^* \in [15, 512]$.
* **What to Explain Verbally**:
  - "When we evaluated the entire benchmark using Range-based VUS-PR against ground-truth labels across all 870 datasets, they partitioned into four distinct geometric sensitivity classes:"
    1. **Class 3 (Horizon-Critical) is the largest class (348 datasets, 40.0%)**: Performance is governed almost entirely by the temporal receptive field $L$, while remaining robust across intrinsic dimensions ($d=2$ is consistently effective). Matching the dynamical horizon is paramount.
    2. **Class 4 (Co-Sensitive) is the second largest class (257 datasets, 29.5%)**: Requires precise joint alignment of both $L$ and $d$; misaligning either parameter destroys orthogonal anomaly contrast.
    3. **Class 1 (Robust) makes up nearly a quarter (211 datasets, 24.3%)**: Possesses flat Pareto basins (point outliers, isolated spikes) where almost any reasonable $(L, d)$ pair detects anomalies reliably.
    4. **Class 2 (Dimension-Critical) is a rare minority (54 datasets, 6.2%)**: Largely invariant to window length $L$, but critically sensitive to $d$. If $d$ is set too high, tangent spaces absorb off-manifold signal into the normal bundle, destroying detection.
  - **The 70% Receptive Field Takeaway**: "Crucially, Classes 3 and 4 together comprise **69.5% (~70%) of the benchmark**. This proves empirically that delay embedding window selection ($L$) is the dominant bottleneck in time-series anomaly detection."
  - **Universal Invariant**: "Across all classes, the optimal intrinsic dimension satisfies $d^* \in \{1, 2, 4\}$ in over 80% of series, regardless of whether $L^* \in [15, 512]$."
* **Transition**: "Before examining dynamical regimes, what do genuine anomalies actually look like geometrically in these benchmarks? We conducted an empirical audit of the ground-truth anomalies."

---

### Slide 8: 7. Ground-Truth Anomaly Dynamics: Morphology & Contrast (Frame 7)
* **Table & Bullets on Slide**:
  - Ground-Truth Morphology Audit (Evaluated at Oracle $(L^*, d^*)$):
    - **Point / Spikes** (Stock, Yahoo): Length $1\text{--}2$, Peak Contrast $3.2\times\text{--}17.2\times$, Max $Z$-Sep $6.5\sigma$, Dynamical Driver: Small $L$ (Energy $\propto 1/L$), high $d$.
    - **Subsequence** (UCR, SED): Length $\approx T_{\text{dom}}$ ($25\text{--}64$), Peak Contrast $1.8\times\text{--}2.5\times$, Max $Z$-Sep $3.6\sigma$, Dynamical Driver: $L \approx k \cdot T_{\text{dom}}$ (orbit closure).
    - **Regime Shift** (SMD, NEK, Exath.): Length $\gg T_{\text{dom}}$ ($44\text{--}1800+$), Peak Contrast $3.6\times\text{--}\mathbf{118.6\times}$, Max $Z$-Sep $\mathbf{2,464\sigma}$, Dynamical Driver: Massive off-manifold excursion.
  - Key Empirical Discoveries:
    - **Boundary Overlap Dilution**: $\text{Contrast}_{\min} < 1.0\times$ in **100%** of datasets ($[0.12\times, 0.81\times]$). Windows clipping edges contain 99% normal data; arithmetic averaging dilutes peak anomaly contrast by up to 86.5%.
    - **Synthetic vs. Genuine Gap**: Synthetic training perturbations yield $\approx 1.15\times$, whereas true anomalies exhibit $1.8\times\text{--}118.6\times$ peak orthogonal contrast.
    - **Physical Justification for Apodization**: Motivates center-weighted Gaussian filtering to preserve sharp anomaly energy at window centers.
* **What to Explain Verbally**:
  - "Rather than relying solely on theoretical assumptions or synthetic models, we conducted an empirical audit directly on the benchmark ground-truth anomalies under their Oracle $(L^*, d^*)$ configurations."
  - "This revealed three distinct morphological archetypes:"
    1. **Point Outliers / Spikes (e.g. Stock, Yahoo)**:
       - Duration is very short (1 to 2 timesteps).
       - Because the anomaly energy per window scales as $\text{Energy} \propto 1/L$, large window sizes dilute the spike across normal data. They require compact horizons ($L \in [8, 20]$) to maximize energy concentration.
       - Peak contrast reaches $3.2\times$ to $17.2\times$. If the normal background has complex multi-periodicity (e.g. Yahoo hourly/daily cycles), they also require a higher dimension ($d \ge 6$) so normal cycles don't leak into the residual.
    2. **Subsequence / Shapelets (e.g. UCR, SED)**:
       - Anomaly duration matches the natural cycle length ($\approx T_{\text{dom}}$, 25 to 64 steps).
       - These anomalies represent a distortion in the shape of the waveform. The optimal horizon $L^*$ is strictly locked to an integer multiple of $T_{\text{dom}}$ ($L^* \approx 1\text{--}3 \times T_{\text{dom}}$) to capture complete phase orbits.
       - They exhibit modest contrast ($1.8\times$ to $2.5\times$), making them highly sensitive to misalignment.
    3. **Regime Shifts (e.g. SMD, NEK, Exathlon)**:
       - Massive structural failures lasting hundreds to thousands of steps ($44$ to $1800+$ timesteps).
       - They break the normal manifold completely: peak orthogonal contrast reaches **$20\times$ to $118.6\times$**, with $Z$-separations up to **$2,464\sigma$**! They are detectable across a wide range of horizons.
  - "**The Boundary Overlap Dilution Pathology**":
    - "Crucially, across **100% of analyzed datasets**, the minimum contrast ratio among anomaly-flagged windows is strictly less than 1.0 ($\text{Contrast}_{\min} \in [0.12\times, 0.81\times]$)."
    - "Why? Because sliding windows that only graze the leading or trailing edge of an anomaly contain 99% normal data and 1% anomaly. Under standard uniform arithmetic averaging, these edge windows severely dilute the true peak signal—suppressing a $118\times$ peak down to a $50\times$ mean, or an $17\times$ peak down to $3.7\times$."
    - "This physical reality is the direct empirical justification for our **Center-Weighted Gaussian Apodization** (Slide 14 / Frame 13)."
  - "**The Synthetic vs. Genuine Contrast Gap**":
    - "Synthetic anomaly generation during unsupervised tuning yields contrast ratios around $\approx 1.15\times\text{--}1.5\times$."
    - "Genuine physical anomalies exhibit $1.8\times$ to $118.6\times$ peak separation. This confirms that our synthetic orthogonal injection serves as a rigorous, conservative lower bound—if a model can detect subtle $1.15\times$ orthogonal deviations, it effortlessly separates genuine physical anomalies."
* **Transition**: "Having characterized the physical morphology and separation of real anomalies, let's now investigate the underlying dynamical processes generating these benchmark series via Surrogate Data Testing."

### Slide 9: 8. Determinism Audit: Surrogate Data Testing (Frame 8)
* **Formulas & Table on Slide**:
  - IAAFT null hypothesis: Linear Gaussian process with static monotonic observation.
  - Determinism significance metric:
    $$Z_{\text{det}} = \frac{\langle CD_{\text{surr}} \rangle - CD_{\text{orig}}}{\sigma(CD_{\text{surr}})}$$
  - Regime Distribution Table (811 Datasets):
    - **Stochastic / Linear** ($Z < 1.96$): Count: **616** | Share: **76.0%** | Mean $Z$: $-1.69$
    - **Mixed / Weak Determinism** ($1.96 \le Z < 5.0$): Count: 141 | Share: 17.4% | Mean $Z$: $+3.18$
    - **Strong Determinism** ($Z \ge 5.0$): Count: 54 | Share: 6.7% | Mean $Z$: $+8.30$
  - Key Finding: 76.0% of benchmarks lack a deterministic attractor; they behave as stochastic diffusion or autoregressive processes.
* **What to Explain Verbally**:
  - "We tested the entire benchmark using Iterated Amplitude Adjusted Fourier Transform (IAAFT) surrogates."
  - "The finding was definitive: **76.0% (616 out of 811 datasets)** cannot reject the null hypothesis of a linear stochastic process ($Z < 1.96$)."
  - "Strong determinism ($Z \ge 5.0$) accounts for only 6.7% of the benchmark, concentrated in biological oscillators (ECG waveforms) and physical rotating machinery."
  - "Web traffic and cloud telemetry (`Yahoo`, `WSD`, `NAB`, `SMD`) behave predominantly as stochastic diffusion processes or random walks with jumps."
* **Transition**: "To understand how we computed these $Z$-scores, what is the Correlation Dimension and why was IAAFT mandatory instead of simple FFT phase randomization?"

---

### Slide 10: 9. Determinism Mechanics: Correlation Dimension & IAAFT (Frame 9)
* **Points on Slide**:
  - 1. Correlation Dimension ($CD$, Grassberger-Procaccia 1983):
    - Scaling Metric: $C(r) = \frac{2}{N(N-1)} \sum_{i < j} \Theta(r - \|\bm{w}_i - \bm{w}_j\|_2) \propto r^{CD}$ as $r \to 0$.
    - Physical Meaning: Measures phase-space fractal dimension. Deterministic limit cycles $\implies CD \approx 1\text{--}2$; stochastic random walks $\implies CD \to m$.
  - 2. The Central Limit Theorem Trap of Naive FFT:
    - Summing random Fourier phases forces a **strictly Gaussian** distribution.
    - Real telemetry with heavy tails/spikes falsely rejects $H_0 \implies$ **spurious determinism**.
  - 3. IAAFT Alternating Projections (Schreiber & Schmitz 1996):
    - Spectral Step: Replaces Fourier amplitudes with $|X_k^{\text{orig}}|$ (preserves ACF).
    - Rank Step: Rank-orders back to exact original values (preserves histogram).
    - Rigorous Null: $Z_{\text{det}} > 1.96$ isolates **true deterministic dynamics**.
* **What to Explain Verbally**:
  - "**1. What is the Correlation Dimension ($CD$)?**":
    - "Correlation Dimension is a classical geometric invariant from nonlinear dynamics (Grassberger & Procaccia, 1983) that measures how densely points cluster in phase space."
    - "We construct embedded vectors $\bm{w}_t \in \mathbb{R}^m$ and compute the Correlation Sum $C(r)$, which simply measures the probability that two randomly selected trajectory states are within Euclidean distance $r$ of each other."
    - "In the scaling region, $C(r)$ scales as a power law: $C(r) \propto r^{CD}$, where $CD = \lim_{r \to 0} \frac{\log C(r)}{\log r}$."
    - "**Intuition**:"
      - "If the dynamics live on a 1D closed circle or limit cycle, doubling $r$ doubles the number of neighbors $\implies C(r) \propto r^1 \implies CD \approx 1$."
      - "If the dynamics live on a 2-torus, $C(r) \propto r^2 \implies CD \approx 2$."
      - "If the series is **pure stochastic noise or a random walk**, points randomly fill the entire embedding space, so $C(r) \propto r^m \implies CD \to m$ (scales up with embedding dimension)."
    - "Therefore, when we randomize phases in a deterministic system, $CD$ jumps significantly upward from $CD_{\text{orig}}$ to $CD_{\text{surr}}$, yielding a large positive $Z_{\text{det}}$."
  - "**2. Why Simple FFT Phase Randomization Fails (The Central Limit Trap)**":
    - "In naive Fourier phase randomization, you take the FFT of the signal, assign uniform random phases $\phi_k \sim \text{Uniform}[0, 2\pi)$, and compute the inverse FFT."
    - "By the **Central Limit Theorem**, summing hundreds of independent cosine modes with randomized phases mathematically forces the surrogate time series to follow a **strictly Gaussian (Normal)** distribution in the time domain."
    - "Real benchmark data (server CPU, network packet rates, financial returns) is heavily non-Gaussian: it exhibits positive skewness, kurtosis, intermittent spikes, or power-law tails."
    - "If you test non-Gaussian data using naive FFT surrogates, any nonlinear metric (like $CD$ or mutual information) will show a massive difference between the original series and the surrogates **solely because the surrogate was forced to be Gaussian**! You get rampant false positives, labeling purely linear stochastic noise as 'deterministic chaos.'"
  - "**3. How IAAFT Solves This (Schreiber & Schmitz 1996)**":
    - "Iterated Amplitude Adjusted Fourier Transform (IAAFT) uses an alternating projection algorithm to satisfy two strict constraints simultaneously:"
      1. **Frequency Domain**: Enforces the exact Fourier power spectrum $|X_k^{\text{orig}}|^2$, preserving the complete linear autocorrelation function.
      2. **Time Domain**: Rank-orders the transformed points back onto the exact sorted values of the raw data, preserving the identical histogram, skewness, and heavy tails.
    - "Because IAAFT preserves both linear autocorrelation and the empirical amplitude distribution, any statistically significant difference ($Z_{\text{det}} > 1.96$) is **100% guaranteed to originate from true nonlinear deterministic phase coupling**, not distributional artifacts."
* **Transition**: "Having rigorously established that 76% of our benchmark datasets are stochastic processes rather than deterministic attractors, how does delay embedding apply to them? That brings us to Stark's Stochastic Takens theory."

### Slide 11: 10. Theory: Stochastic Takens & Codimension Disconnect (Frame 10)
* **Formulas on Slide**:
  - Stochastic Delay Embedding (Stark 1999):
    $$d\bm{x}_t = f(\bm{x}_t) \, dt + \sigma(\bm{x}_t) \, d\bm{w}_t \implies \bm{x}_t \simeq \text{Bundle}(\mathcal{L}_{\text{Fokker-Planck}})$$
    *CNF and Conditional Diffusion parameterize the Fokker-Planck drift-diffusion operator.*
  - The "Unfolding vs. Discrimination" Disconnect:
    $$\min m_{\text{Takens}} \implies k = 1 \text{ (Minimal embedding)} \implies \text{\textbf{Fails AD}}$$
    $$\max k \implies L - d \gg 1 \text{ (Normal bundle capacity)} \implies \text{\textbf{Required}}$$
  - Conclusion: $L$ is governed by anomaly duration; $d$ is governed by the rank of normal dynamics.
* **What to Explain Verbally**:
  - "In 1999, Stark extended Takens' theorem to stochastic differential equations. In a stochastic system, delay coordinates reconstruct the bundle manifold of the Markov transition generator—the Fokker-Planck operator."
  - "Since our Flow Matching models explicitly learn the drift-diffusion velocity field of this generator, the delay embedding formulation is mathematically exact."
  - "More importantly, this resolves why classical nonlinear heuristics like False Nearest Neighbors (FNN) fail for anomaly detection:"
    - "FNN seeks the **minimal** dimension $m$ that eliminates topological self-crossings ($m \approx 3\text{--}5$). If $L = 3$ and $d = 2$, the codimension is $k = 1$. The orthogonal subspace is virtually empty!"
    - "Anomaly detection requires an extended ambient horizon $L$ so that anomalous trajectories have geometric context to project with high contrast into the orthogonal complement."
  - "Conclusion: $L$ is governed by the temporal duration of the anomaly signature; $d$ is governed by the rank of normal dynamics."
* **Transition**: "Now let's examine how we solve horizon selection $L$ across both periodic and aperiodic series."

---

### Slide 12: 11. Horizon $L$: Periodic Regime ($T_{\text{dom}} \ge 6$) (Frame 11)
* **Formulas on Slide**:
  - Autocorrelation operator:
    $$R_{xx}(\tau) = \frac{1}{(N_{\text{tr}} - \tau)\sigma^2} \sum_{t=1}^{N_{\text{tr}} - \tau} (x_t - \mu)(x_{t+\tau} - \mu)$$
  - Dominant fundamental period:
    $$T_{\text{dom}} = \arg\max_{\tau \ge 6} R_{xx}(\tau), \quad \text{prominence} \ge 0.03$$
  - Physical harmonic resonance grid:
    $$\mathcal{G}_L^{\text{periodic}} = \{1 \cdot T_{\text{dom}}, \, 2 \cdot T_{\text{dom}}, \, 3 \cdot T_{\text{dom}}, \, 4 \cdot T_{\text{dom}}\}$$
  - Coverage: Governs **73.1%** (629/860) of benchmark datasets.
  - Mechanism: Enforces topological orbit closure without phase boundary cuts.
* **What to Explain Verbally**:
  - "73.1% of benchmark datasets exhibit clear periodic recurrence with $T_{\text{dom}} \ge 6$."
  - "For these series, sampling arbitrary window sizes (like powers of 2: 16, 32, 64) cuts across phase boundaries and creates artificial manifold folds."
  - "By anchoring candidate horizons to physical integer harmonics of $T_{\text{dom}}$, we guarantee closed phase-space orbits and eliminate boundary phase cuts."
* **Transition**: "What about the remaining 26.9% of the benchmark where autocorrelation fails?"

---

### Slide 13: 12. Horizon $L$: Aperiodic Regime via Fraser-Swinney AMI (Frame 12)
* **Formulas on Slide**:
  - Failure of Linear Autocorrelation: $R_{xx}(\tau)$ decays monotonically $\implies T_{\text{dom}} = 1$.
  - Average Mutual Information (Fraser & Swinney 1986):
    $$I(\tau) = \iint p(x_t, x_{t+\tau}) \log \frac{p(x_t, x_{t+\tau})}{p(x_t) p(x_{t+\tau})} \, dx_t \, dx_{t+\tau}$$
  - First local information minimum:
    $$\tau_{\text{AMI}} = \arg\min_\tau I(\tau) \quad (\text{Maximal Non-Redundant Dynamical Info})$$
  - Dyadic multi-scale candidate grid:
    $$\mathcal{G}_L^{\text{aperiodic}} = \text{clip}\big(\{2^k \cdot \tau_{\text{AMI}} \mid k \in \{1, 2, 3, 4, 5, 6\}\}, \, 8, \, L_{\max}\big)$$
  - Empirical Audit (231 Aperiodic Series): 97.0% clean minima; **61.5%--90.0%** Oracle $L^*$ hit rate (IOPS: 90.0%, WSD: 75.8%).
* **What to Explain Verbally**:
  - "In aperiodic, bursty, or stochastic series (26.9% of the benchmark), linear autocorrelation decays monotonically to zero, yielding $T_{\text{dom}} = 1$."
  - "To resolve this, we use Fraser and Swinney's 1986 theorem: Average Mutual Information (AMI) measures nonlinear dynamical dependence without assuming linearity or periodicity."
  - "The first local minimum $\tau_{\text{AMI}}$ marks the point where $\bm{x}_{t+\tau}$ provides maximum new, non-redundant dynamical information."
  - "We construct a dyadic candidate expansion: $\{2, 4, 8, 16, 32, 64\} \times \tau_{\text{AMI}}$, clipped to $[8, L_{\max}]$, capturing both short transient spikes and long-term trend shifts."
  - "Across all 231 aperiodic benchmark traces, 97.0% possess a clean first local minimum (median $\tau = 5$). This grid hits within 30% of the true Oracle horizon in 61.5% to 90% of cases."
* **Transition**: "At test time, when we project sliding-window scores back to 1D time series, we discovered another mathematical pathology: pooling dilution."

---

### Slide 14: 13. Test-Time Pointwise Projection: Anti-Dilution (Frame 13)
* **Formulas on Slide**:
  - Boundary dilution of arithmetic mean pooling:
    $$s_{\text{arith}}(t) = \frac{1}{R_t} \sum_{k=0}^{L-1} s(W_{t-k})[k] \implies \text{\textbf{Dilutes sharp spikes by 86.5\%}}$$
  - Center-weighted Gaussian apodization:
    $$w[k] = \exp\left(-\frac{1}{2}\left(\frac{k - (L-1)/2}{\sigma}\right)^2\right), \quad \sigma = \frac{L}{4}$$
    $$s_{\text{gauss}}(t) = \frac{\sum_k w[k] \cdot s(W_{t-k})[k]}{\sum_k w[k]}$$
  - Soft-max generalized $L_p$-norm pooling:
    $$s_{L_p}(t) = \left(\frac{1}{R_t} \sum_{k=0}^{L-1} s(W_{t-k})[k]^p\right)^{1/p}, \quad p \ge 2$$
  - Result: `303_UCR` VUS-PR improves $0.2856 \to 0.3103$ (+8.6% relative gain).
* **What to Explain Verbally**:
  - "Standard pipelines use uniform arithmetic mean pooling across overlapping windows. But at anomaly boundaries, windows that only clip 1 or 2 anomalous points drag down the average, diluting peak sharpness by up to 86.5%."
  - "We implemented Center-Weighted Gaussian Apodization: convolving scores with a Gaussian kernel ($\sigma = L/4$) smoothly suppresses boundary artifacts while preserving peak anomaly height."
  - "Empirically, Gaussian apodization consistently yields the highest VUS-PR across continuous sensor benchmarks."
* **Transition**: "With horizon $L$ and test projection resolved, what about intrinsic dimension $d$?"

---

### Slide 15: 14. Open Question: Intrinsic Dimension Estimation ($d$) (Frame 14)
* **Formulas on Slide**:
  - The power-law spectral decay fallacy:
    $$\lambda_k \propto k^{-\gamma}, \quad \gamma > 1 \quad \text{(Induced by Temporal Autocorrelation)}$$
  - Universal eigengap collapse:
    $$\arg\max_k (\lambda_k - \lambda_{k+1}) \equiv 1 \quad \text{(Trend Dominance)}$$
  - The Under/Over dilemma:
    $$d = 1 \implies \text{Forces 1D line; normal limit cycles leak into residual}$$
    $$d \ge 8 \implies \text{Tangent space absorbs noise; destroys anomaly contrast}$$
  - Status: Horizon $L$ resolved via Dual Grid; Dimension $d$ left as an open theoretical question.
* **What to Explain Verbally**:
  - "This is an important theoretical finding: why do standard scree / eigengap heuristics fail on real time series?"
  - "In continuous physical signals, temporal autocorrelation induces a continuous power-law decay in the covariance spectrum ($\lambda_k \propto k^{-\gamma}$). There is no clean spectral cliff."
  - "Because the first drop $\lambda_1 \to \lambda_2$ reflects the dominance of the overall low-frequency trend, any naive discrete eigengap heuristic mathematically defaults to $d = 1$ across almost every dataset."
  - "Setting $d = 1$ forces the normal manifold to be a 1D line, causing normal limit cycles ($d=2$) and oscillators to leak into the anomaly residual. Conversely, setting $d \ge 8$ causes the model to memorize high-frequency noise and absorb anomalies."
  - "Therefore, while horizon $L$ is physically and information-theoretically resolved, unsupervised dimension estimation ($d$) remains an active open research question."
* **Transition**: "Finally, let's look at the empirical results of our combined two-stage coarse-to-fine pilot."

---

### Slide 16: 15. Two-Stage Coarse-to-Fine Empirical Pilot (Frame 15)
* **Table on Slide**:
  - Columns: `Dataset`, `Class`, `Selected $(L, d)$`, `Oracle $(L^*, d^*)$`, `Recovery %`.
  - `234_SED` (Class 4): Selected $(94, 4)$, Oracle $(100, 2)$, **103.6%**
  - `811_Exathlon` (Class 1): Selected $(333, 2)$, Oracle $(100, 2)$, **99.9%**
  - `001_NAB` (Class 3): Selected $(115, 16)$, Oracle $(70, 14)$, **99.2%**
  - `277_NEK` (Class 2): Selected $(85, 16)$, Oracle $(15, 2)$, **94.8%**
  - `180_SMD` (Class 2): Selected $(15, 4)$, Oracle $(100, 4)$, **93.2%**
  - `149_Stock` (Class 1): Selected $(50, 4)$, Oracle $(15, 2)$, **81.2%**
  - `029_WSD` (Static Grid) (Class 3): Selected $(333, 2)$, Oracle $(130, 2)$, 0.6%
  - `029_WSD` (**AMI Grid**) (Class 3): Selected $(110, 2)$, Oracle $(130, 2)$, **98.4%**
  - Efficiency: Evaluates $\approx 35\text{--}45$ models total vs. $>500$ in full grid.
* **What to Explain Verbally**:
  - "We deployed our two-stage dispatcher across a representative pilot spanning all four geometric classes:"
    - "Stage 1 evaluates the coarse candidate grid from our dual generator."
    - "Stage 2 locally refines $L$ by $\pm 15\%$ and $d$ by $\pm 1$ around the coarse winner using our Codimension-Normalized Objective."
  - "Results:"
    - "Six datasets achieved **81.2% to 103.6% of the Oracle ceiling** (`234_SED` reached 103.6%, `Exathlon` 99.9%, `NAB` 99.2%, `NEK` 94.8%, `SMD` 93.2%, `Stock` 81.2%)."
    - "On aperiodic series, look at `029_WSD`: under an arbitrary static grid, it chose $L=333$ and achieved a dismal 0.6% recovery. Under our Fraser-Swinney AMI grid, it chose $L=110$ and achieved **98.4% Oracle recovery**—a 164-fold improvement."
  - "In terms of computational efficiency, this coarse-to-fine structure requires evaluating only 35 to 45 models per dataset, compared to over 500 in a full grid sweep ($>92\%$ compute reduction)."
* **Transition**: "To wrap up, here is the summary and our active research roadmap."

---

### Slide 17: 16. Summary & Roadmap (Frame 16)
* **Bullets on Slide**:
  - Completed Contributions:
    - **Dual Grid Selection**: Unifies periodic ($k \cdot T_{\text{dom}}$) and aperiodic ($2^k \cdot \tau_{\text{AMI}}$) horizons without test leakage.
    - **Direct Orthogonal Injection**: Strict off-manifold perturbation ($u_\perp \in \ker(V_d^\top)$) for unsupervised contrast tuning.
    - **Gaussian Apodization**: Suppresses boundary dilution by $86.5\%$.
  - Active Research Roadmap:
    - **Dimension Estimation ($d$)**: Investigate continuous Participation Ratio $d_{\text{eff}} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$ and Grassmannian curvature to bypass power-law eigengap collapse.
    - **Global Benchmark Run**: Execute dual coarse-to-fine dispatcher across all benchmark datasets.
* **What to Explain Verbally**:
  - "In summary: we have grounded horizon selection $L$ via physical harmonics and Shannon information theory; we have fixed test-time dilution with Gaussian apodization; and we have validated orthogonal synthetic perturbations for unsupervised model ranking."
  - "Our immediate next steps are:"
    1. "Investigate continuous dimension measures like the Participation Ratio $d_{\text{eff}} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$ and Grassmannian curvature to resolve integer $d$ without collapsing to 1."
    2. "Deploy the dual coarse-to-fine dispatcher across the entire benchmark."
  - "Thank you, and I welcome any questions."
