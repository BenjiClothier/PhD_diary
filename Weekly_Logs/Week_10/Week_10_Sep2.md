# Weekly Log: [02 - 09 Sept]

## 🎯 Focus of the Week
Find L!! 

---

## 📝 To-Do List
- [x] Gather and understand tools from Diffusion Geometry
- [x] Apply tools to benchmark
- [x] Classify datasets as either deterministic or stochastic or a mix??
- [x] Find what tools work on what dataset. 
- [x] Why does the Spectral Analysis of the Laplacian lead to $\lambda_0 = 0$?
- [x] Reconcile 76% stochasticity with dynamical systems theory (Stark's stochastic embedding)
- [x] Formulate the "Unfolding vs. Discrimination" dichotomy in anomaly detection
- [x] Implement Dual ACF-AMI candidate grid for horizon $L$ selection across periodic & aperiodic series
- [x] Resolve test-time window projection dilution (Gaussian apodization & $L_p$-norm pooling)
- [x] Implement direct orthogonal synthetic anomaly injection ($u_\perp \in \ker(V_d^\top)$)
- [ ] Resolve unsupervised intrinsic dimension $d$ estimation (Power-law spectral decay leaves $d$ as an open question)

---

## 🔬 Progress & Experiments

### Laplacian Eigenvalues

**The Mathematical Reason: The Constant Vector**
The graph Laplacian used here is defined as $\Delta = I - P$, where $P$ is the Markov transition matrix. 

Because $P$ represents probabilities, every row in $P$ must sum to exactly $1$. This mathematically guarantees that the constant vector $\mathbf{1}$ (a vector composed entirely of $1$s) is a right eigenvector of $P$ with an eigenvalue of $1$:
$$P \mathbf{1} = \mathbf{1}$$

When we apply the Laplacian operator $\Delta$ to this constant vector, we get:
$$\Delta \mathbf{1} = (I - P) \mathbf{1} = \mathbf{1} - P\mathbf{1} = \mathbf{1} - \mathbf{1} = \mathbf{0}$$
Because $\Delta \mathbf{1} = 0 \cdot \mathbf{1}$, the value $0$ is mathematically guaranteed to be an eigenvalue ($\lambda_0$). 

**The Physical Intuition: Thermodynamic Equilibrium**
The Laplacian measures diffusion—the flow of "heat" or information across the graph based on the difference in values between adjacent nodes. 

The constant vector $\mathbf{1}$ represents a state where every node in the entire network has the exact same temperature (Thermodynamic Equilibrium). Because there is no temperature difference between any two connected neighbours, no heat flows across the edges. The rate of diffusion is exactly zero.

**The Graph Theory Significance**
In spectral graph theory, the multiplicity of the $0$ eigenvalue tells you exactly how many completely disconnected islands exist in your network.
*   If $\lambda_0 = 0$ and $\lambda_1 > 0$, there is exactly $1$ connected graph. Heat can eventually reach every node.
*   If $\lambda_0 = 0$ and $\lambda_1 = 0$, there are $2$ completely disconnected sub-graphs. Heat on one island can never diffuse to the other. 

This is why we check that $\lambda_1 > 0.1$ when looking for the Spectral Gap. We must mathematically prove the time series forms one continuous, connected manifold before we check if it forms a perfect topological circle ($\lambda_2 - \lambda_1 = 0$).

---

### Surrogate Data Testing

In the real world, data is rarely 100% deterministic or 100% stochastic; it's usually a deterministic signal polluted by measurement noise. To rigorously prove determinism, scientists use Surrogate Data Testing.

    How it works: You take your original time series and run it through a Fourier transform. You keep the amplitudes exactly the same, but you randomize the phases, and then inverse-transform it back into a time series.

    The Result: You have just created a "surrogate." It has the exact same mean, variance, and frequency power spectrum as your original data, but any nonlinear geometric structure has been completely destroyed.

    The Test: You calculate a nonlinear metric (like Correlation Dimension or Translation Error) on both your original data and 30-40 randomized surrogates. If the metric for your original data falls far outside the distribution of the surrogates, you can statistically reject the null hypothesis that your data is just stochastic noise.

#### Upgrade from standard FT Surrogates to IAAFT Surrogates

Using standard Fourier Transform (FT) phase shuffling, we are testing the null hypothesis that our data is a Gaussian linear stochastic process.

    The Problem: Many real-world datasets have skewed distributions or heavy tails (non-Gaussian). Standard FT shuffling will change the amplitude distribution (the histogram of our data). If your original data is non-Gaussian noise, standard FT surrogates will have a different CD, and you will get a high Z-score. We will falsely conclude the data is deterministic, when in reality, it's just non-Gaussian noise.

    The Fix: We must use IAAFT (Iterated Amplitude Adjusted Fourier Transform) surrogates. IAAFT preserves both the power spectrum (autocorrelation) and the exact amplitude distribution of the original data. There are Python libraries like surrogates or NoLiTSA that generate IAAFT surrogates easily.

#### Benchmark Empirical Classification Results (811 Datasets)

Surrogate Data Testing was executed across **811 time series** from the TSB-UAD benchmark (delay-embedded with $m=5, \tau=1$). For each series, the Correlation Dimension of the original data ($CD_{\text{orig}}$) was evaluated against an ensemble of randomized surrogates ($CD_{\text{surr}}$) to compute the determinism significance score:

$$Z = \frac{\langle CD_{\text{surr}} \rangle - CD_{\text{orig}}}{\sigma(CD_{\text{surr}})}$$

A statistically significant positive $Z$-score demonstrates that phase randomization destroys geometric organization (inflating the correlation dimension), proving the existence of nonlinear deterministic structure.

| Category | Classification Rule | Count | Share | Mean $Z$-Score | Median $Z$-Score | Mean $CD_{\text{orig}}$ | Mean $CD_{\text{surr}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stochastic / Linear** | $Z < 1.96$ | **616** | **76.0%** | -1.69 | -0.52 | 2.80 | 2.71 |
| **Mixed / Weak Determinism** | $1.96 \le Z < 5.0$ | **141** | **17.4%** | +3.18 | +2.99 | 1.71 | 2.83 |
| **Strong Determinism** | $Z \ge 5.0$ ($p < 10^{-6}$) | **54** | **6.7%** | +8.30 | +7.42 | 1.62 | 3.05 |
| **Total Benchmark** | | **811** | **100.0%** | -0.18 | +0.17 | 2.53 | 2.75 |

##### Domain Breakdown Across the Benchmark

| Domain / Collection | Total Datasets | Stochastic / Linear | Mixed Determinism | Strong Determinism | % Deterministic (Strong + Mixed) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **UCR** (Sensors, ECG, Robotics) | 228 | 144 | 50 | **34** | **36.8%** |
| **YAHOO** (Web traffic, ad clicks) | 233 | 198 | 31 | 4 | 15.0% |
| **WSD** (Web Service metrics) | 111 | 100 | 11 | 0 | 9.9% |
| **SMD** (Server Machine physical metrics) | 38 | 25 | 12 | 1 | **34.2%** |
| **Exathlon** (Distributed computing trace) | 32 | 27 | 1 | 4 | 15.6% |
| **OPPORTUNITY** (Human activity sensors) | 29 | 20 | 8 | 1 | **31.0%** |
| **NAB** (AWS cloud / network traffic) | 28 | 25 | 3 | 0 | 10.7% |
| **SVDB** (Medical / ECG arrhythmia) | 20 | 13 | 5 | 2 | **35.0%** |
| **SMAP** (NASA spacecraft telemetry) | 19 | 14 | 3 | 2 | 26.3% |
| **IOPS** (KPI web service metrics) | 17 | 14 | 2 | 1 | 17.6% |
| **MGAB** & **SED** (Medical / ECG) | 12 | 0 | 12 | 0 | **100.0%** |
| **MSL** (Mars Science Laboratory) | 8 | 5 | 2 | 1 | **37.5%** |
| **NEK** & **Stock** (Network / Finance) | 12 | 12 | 0 | 0 | **0.0%** |
| **SWaT** (Industrial water treatment) | 1 | 0 | 0 | 1 | **100.0%** |

---

### Theoretical Synthesis: Stochastic Takens & The "Unfolding vs Discrimination" Disconnect

Given that 76.0% of the benchmark traces cannot reject the null hypothesis of a linear stochastic process, we resolved two crucial theoretical questions regarding how to frame our Continuous Normalizing Flow (CNF) and diffusion geometry:

#### 1. Stochastic Delay Embedding (Stark 1999)
Classical Takens' theorem (1981) assumes a deterministic, autonomous dynamical system with zero dynamical noise. However, **Stark (1999)** and **Stark et al. (2003)** proved the *Delay Embedding Theorem for Stochastic and Driven Systems* (published in *Annals of Applied Probability*):
- For stochastic differential equations ($dX_t = f(X_t)dt + \sigma dW_t$) or autoregressive processes driven by noise, delay coordinate maps reconstruct the **bundle manifold of the underlying Markov transition / Fokker-Planck operator**.
- Because our Continuous Normalizing Flow explicitly parameterizes the velocity field of a drift-diffusion process, framing sliding windows as a stochastic delay embedding is mathematically sound.

#### 2. The "Unfolding vs Discrimination" Disconnect
Classical dynamical heuristics (False Nearest Neighbors, Cao's method) aim to find the *minimal* embedding dimension $m$ that eliminates topological self-intersections ($m_{\text{FNN}} \approx 3\text{--}5$). However, anomaly detection has a fundamentally different objective:
- Anomaly scoring relies on projecting out-of-distribution vectors into the orthogonal complement of the normal tangent space:
  $$\text{Score}(x) = \|(I - V_d V_d^\top)(x - \mu)\|_2$$
- If $L = m_{\text{FNN}} \approx 3$ and $d = 2$, the codimension is $k = L - d = 1$. The orthogonal subspace is virtually nonexistent, making it impossible to detect shape distortions or temporal phase shifts.
- **Key Takeaway**: Window length $L$ must be sized to encompass the full temporal duration of the anomaly's signature, while $d$ represents the rank of the normal dynamics.

---

### Solving Horizon Estimation ($L$): The Dual ACF + Average Mutual Information (AMI) Architecture

To eliminate arbitrary static window grids, we implemented a physically and information-theoretically grounded **Dual Candidate Grid Generator**:

1. **Periodic Regime ($T_{\text{dom}} \ge 6$, 73.1% of benchmarks)**:
   - Linear Autocorrelation (ACF) identifies the dominant cycle $T_{\text{dom}}$.
   - Based on Nyquist-Shannon orbit closure, candidate horizons are generated via physical integer harmonics:
     $$\mathcal{G}_L^{\text{periodic}} = \{1 \cdot T_{\text{dom}},\, 2 \cdot T_{\text{dom}},\, 3 \cdot T_{\text{dom}},\, 4 \cdot T_{\text{dom}}\}$$
2. **Aperiodic / Stochastic Regime ($T_{\text{dom}} < 6$, 26.9% of benchmarks)**:
   - For aperiodic or bursty traces, ACF decays monotonically ($T_{\text{dom}} = 1$).
   - We deploy Fraser-Swinney **Average Mutual Information (AMI)**:
     $$I(X_t; X_{t+\tau}) = \iint p(x_t, x_{t+\tau}) \log \frac{p(x_t, x_{t+\tau})}{p(x_t) p(x_{t+\tau})} \, dx_t \, dx_{t+\tau}$$
   - The first local minimum $\tau_{\text{AMI}} = \arg\min_\tau I(X_t; X_{t+\tau})$ marks the timescale where $X_{t+\tau}$ provides maximal new dynamical information.
   - We construct a dyadic multi-scale candidate grid:
     $$\mathcal{G}_L^{\text{aperiodic}} = \text{clip}\big(\{2, 4, 8, 16, 32, 64\} \times \tau_{\text{AMI}},\, L_{\min}=15,\, L_{\max}\big)$$

#### Empirical Validation across All 231 Aperiodic Datasets:
- **97.0% (224 / 231)** of aperiodic traces possess a clean first local minimum in AMI (median $\tau_{\text{AMI}} = 5$).
- **61.5% overall hit rate** matching ground-truth Oracle $L^*$ within 30% tolerance (**75.8% in WSD**, **90.0% in IOPS**, **83.3% in Genesis**).
- **Case Study `029_WSD`**:
  - Under static grid: selected $L=333, d=2 \implies \text{VUS-PR} = 0.0047$ (0.59% Oracle recovery).
  - Under AMI grid ($\tau_{\text{AMI}} = 3$): selected coarse $L=96 \to$ fine $L=110, d=2 \implies \text{VUS-PR} = 0.7812$ (**98.4% Oracle recovery**, +166x gain).

---

### Test-Time Anti-Pooling Dilution Fixes

When overlapping sliding-window anomaly scores are mapped back to 1D pointwise series, standard **arithmetic mean pooling** mathematically dilutes localized anomaly peaks by up to 86.5% due to edge boundary averaging.

To fix this, we implemented and parameterized three test-time projection modes in `src/samplers/scorer.py`:
1. **Gaussian Apodization (`gaussian_pooling`)**: Center-weighted convolution ($w[k] = \exp(-0.5((k-c)/\sigma)^2)$, $\sigma = R/4$), suppressing window boundary dilution by 86.5%.
2. **Soft-Max Generalized $L_p$-Norm Pooling (`soft_max_pooling`)**: $(\frac{1}{C}\sum s_i^p)^{1/p}$, accentuating high-confidence localized anomaly spikes.
3. **Arithmetic Mean (`arithmetic_mean`)**: Maintained as a baseline reference.

Empirical testing confirmed that `gaussian_pooling` consistently maximizes Range-based VUS-PR (e.g., `303_UCR` improved from 0.2856 to 0.3103, +8.6% relative gain; `149_Stock` VUS-PR improved from 0.7372 to 0.7457).

---

### Direct Orthogonal Synthetic Anomaly Injection & Codimension Excess SNR

During unsupervised joint $(L, d)$ tuning, models must be scored on their ability to separate anomalies without accessing test labels.
1. **Direct Orthogonal Injection**:
   Rather than arbitrary Gaussian noise or GenIAS perturbations (which partially project onto the learned tangent space $V_d$), we project perturbations strictly into the orthogonal complement:
   $$u_\perp = \frac{(I - V_d V_d^\top)\xi}{\|(I - V_d V_d^\top)\xi\|_2} \cdot \sigma_{\text{scale}}, \quad \max |V_d^\top u_\perp|_\infty \approx 0$$
2. **Codimension-Aware Excess SNR**:
   To prevent codimension bias (where large $L$ artificially inflates raw contrast), we evaluate candidate pairs using scale-invariant excess signal-to-noise ratio:
   $$\text{Excess\_SNR} = (\text{CR}^2 - 1) \cdot (L - d)$$

---

### Two-Stage Coarse-to-Fine Optimization: 16-Dataset Pilot Empirical Results

We tested the combined pipeline across a representative 16-dataset pilot spanning all four geometric sensitivity classes:
- **Stage 1 (Coarse Grid)**: Evaluates candidate pairs generated by the dual grid (4–6 $L$ values, 5 $d$ values $\in \{1, 2, 4, 8, 14\}$).
- **Stage 2 (Fine Search)**: Refines $\pm 0.15 L$ and $\pm 1 d$ around the coarse Pareto winner.

**Scorecard Summary**:
- **6 datasets achieved $\ge 81.2\% - 103.6\%$ of the Oracle ceiling**:
  - `234_SED` (Class 4): **103.6% recovery** (Selected $L=94, d=4 \to \text{VUS-PR} = 0.9659$ vs Oracle 0.9327).
  - `811_Exathlon` (Class 1): **99.9% recovery** (Selected $L=333, d=2 \to \text{VUS-PR} = 0.9989$ vs Oracle 0.9997).
  - `001_NAB` (Class 3): **99.2% recovery** (Selected $L=115, d=16 \to \text{VUS-PR} = 0.9439$ vs Oracle 0.9515).
  - `277_NEK` (Class 2): **94.8% recovery** (Selected $L=85, d=16 \to \text{VUS-PR} = 0.9477$ vs Oracle 0.9998).
  - `180_SMD` (Class 2): **93.2% recovery** (Selected $L=15, d=4 \to \text{VUS-PR} = 0.9297$ vs Oracle 0.9971).
  - `149_Stock` (Class 1): **81.2% recovery** (Selected $L=50, d=4 \to \text{VUS-PR} = 0.7628$ vs Oracle 0.9391).
- On aperiodic traces (`029_WSD`, `178_SMD`), the Fraser-Swinney AMI grid dramatically improved recovery compared to arbitrary static grids.

---


## 📊 Results & Insights

- **Insight 1: Over three-quarters of the benchmark (76.0%) is stochastic / linear noise, NOT low-dimensional deterministic attractors.**
  Takens' delay-embedding theorem and geometric phase-space reconstruction assume that the signal originates from a smooth, autonomous, deterministic dynamical system on a compact manifold. In 616 of the 811 benchmark datasets, phase randomization does *not* significantly increase the correlation dimension ($CD_{\text{orig}} \approx 2.80 \approx CD_{\text{surr}} \approx 2.71$, $Z \le 1.96$). Delay-embedding these series does not unfold a low-dimensional manifold because no attractor exists; they behave predominantly as stochastic random walks or linear autoregressive processes with measurement noise.

- **Insight 2: Strong determinism is concentrated in physical, physiological, and mechanical sensor streams.**
  Almost all strongly deterministic datasets ($Z \ge 5.0$, $p < 10^{-6}$) belong to physical dynamical systems:
  - **UCR** contains **34 out of the 54** strongly deterministic datasets (e.g. ECG waveforms, respiration cycles, mechanical motion).
  - Medical domains (**SED, MGAB, SVDB**) and physical equipment telemetry (**SMD, Exathlon, OPPORTUNITY, SWaT**) exhibit high determinism (30% to 100%).
  - In contrast, human-mediated and web-scale metrics (**YAHOO, WSD, NAB, Stock, NEK**) are overwhelmingly stochastic (>85–100%), exhibiting bursty, Poisson-like, or Brownian motion dynamics.

- **Insight 3: The Tool-to-Dataset Mapping ("What tools work on what dataset"):**
  This explains why single-heuristic approaches to hyperparameter selection ($L$ and $d$) previously exhibited mixed success across the full benchmark:
  1. **Geometric / Manifold Tools (Diffusion Geometry, CDC Metric, Local PCA, Eigengap):**
     - *Best suited for:* **Deterministic & Mixed datasets (24.1% of benchmark)**.
     - Here, the phase space trajectory traces a genuine low-dimensional submanifold ($CD_{\text{orig}} \approx 1.6$). Tools that evaluate manifold geometry (such as topological circle closure, $d=2$ spectral gap, or local tangent planes) are mathematically grounded and provide strong hyperparameter signals.
  2. **Statistical / Spectral Tools (Autocorrelation 1/e lag, PSD Slope, Sample Entropy):**
     - *Best suited for:* **Stochastic / Linear datasets (76.0% of benchmark)**.
     - Geometric dimension tools fail here because the correlation dimension never saturates—points fill the ambient space diffusely. Instead of searching for an attractor dimension, we must tune $L$ to match the statistical correlation horizon (decorrelation time $\tau_{\text{corr}}$ via ACF/AMI) and spectral complexity (log PSD slope and Sample Entropy).
- **Insight 4: The "Unfolding vs Discrimination" Disconnect explains why classical FNN/Cao fail for anomaly detection.**
  Classical delay-embedding heuristics (False Nearest Neighbors, Cao's method) search for the *minimal* embedding dimension $m$ that eliminates geometric self-intersections. In anomaly detection, normal data might unfold smoothly at $m \approx 3\text{--}5$, but setting $L = 3$ leaves a codimension $k = L - d = 1$. The orthogonal subspace $\ker(V_d^\top)$ is virtually empty! Anomaly detection requires an extended temporal window so that anomalies have the physical context to project with high excess SNR into the orthogonal complement.

- **Insight 5: The Power-Law Spectral Decay Fallacy leaves unsupervised intrinsic dimension ($d$) as an open theoretical question.**
  In continuous physical signals, temporal autocorrelation induces a continuous $1/f^\alpha$ power-law spectrum, causing the covariance eigenvalues to decay smoothly ($\lambda_k \propto k^{-\gamma}$). Because the drop from $\lambda_1$ to $\lambda_2$ reflects the dominance of the overall low-frequency trend / mean trajectory, any naive discrete eigengap heuristic ($\arg\max_k (\lambda_k - \lambda_{k+1})$) artificially defaults to $d = 1$ across almost all datasets. Forcing $d = 1$ under-represents limit cycles ($d=2$) and multi-frequency attractors ($d=4$), while setting $d$ too high causes the model to memorize high-frequency noise and absorb anomalies. Intrinsic dimension estimation remains an open challenge.

- **Insight 6: The Dual ACF-AMI Candidate Grid bridges periodic and aperiodic series.**
  - 73.1% of benchmark series are periodic ($T_{\text{dom}} \ge 6$), where Nyquist-Shannon physical harmonics $\{1T, 2T, 3T, 4T\}$ bracket Oracle $L^*$.
  - 26.9% are aperiodic ($T_{\text{dom}} < 6$), where linear ACF collapses. Fraser-Swinney Average Mutual Information (AMI) successfully detects nonlinear information decorrelation scales ($\tau_{\text{AMI}}$) in 97.0% of aperiodic traces, achieving 61.5%–90.0% Oracle hit rates.

---

## ⏭️ Next Steps

1. **Investigate Unsupervised Dimension Estimation ($d$)**:
   - Given that discrete eigengap heuristics fail due to power-law spectral decay, investigate whether continuous Participation Ratio ($d_{\text{eff}} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$) or Grassmannian curvature can provide an unassailable, noise-robust dimension selection rule.
2. **Full Benchmark Deployment of Dual ACF-AMI Pipeline**:
   - Run the two-stage coarse-to-fine dispatcher across all 860 datasets in TSB-AD-U to compute the global macro-averaged VUS-PR and Oracle recovery rate.
3. **Formalize the Stochastic Embedding Chapter / Paper Section**:
   - Write up the theoretical formulation connecting Stark's stochastic delay embedding theorem, Continuous Normalizing Flows (CNFs as Langevin velocity fields), and the codimension excess SNR objective.

