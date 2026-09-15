# Speaker Notes: Week 11 Supervisory Meeting (16 Sept 2026)

**Title**: Hankel CDC-FM for Anomaly Detection  
**Subtitle**: Subspace Decoupling, Riemannian Flow Matching, \& Benchmark SOTA  
**Slides File**: `meeting_1609.pdf` (compiled from `meeting_1609.tex`)  
**Author**: Benji Clothier  

---

## Executive Summary & Narrative Arc

Today's presentation presents the complete architecture, differential geometry, and benchmark results for **Hankel CDC-FM-AD** (Hankel Carré du Champ Conditional Flow Matching for Anomaly Detection).

### Core Narrative Arc
1. **The Fundamental Failure of Neural Autoencoders (Slide 1)**:  
   Why standard neural autoencoders score only `0.1165` on TSB-AD. Deep non-linear encoders have unconstrained null spaces that absorb anomalies into latent space, destroying orthogonal contrast.
2. **The Two-Tier Geometric Solution (Slides 2--3)**:  
   We replace the black-box encoder with an exact linear Karhunen--Loève coordinate partition. Sliding windows in $\mathbb{R}^L$ are split into a global dynamical subspace $\mathbb{R}^d$ and an orthogonal null space $\mathbb{R}^{L-d}$. By the algebraic codimension invariant, off-subspace anomaly energy cannot be squashed.
3. **Physics-Driven Parameter Estimation (Slide 4)**:  
   Why grid searching reconstruction error fails. We derive three physical invariants: multi-cycle recurrence ($L \in [3, 5] T_{\text{dom}}$), torus rank ceilings ($d \le 6$), and non-arbitrary first-trough regime classification.
4. **Local Manifold Geometry & Generative Flows (Slides 5--6)**:  
   Within $\mathbb{R}^d$, nominal data lives on a curved Riemannian manifold. We construct the local Carré du Champ operator $\Gamma_z$ via density-normalised diffusion weights to extract local tangent spaces, and train a continuous normalizing flow with metric drift.
5. **The LatentFlowResNet Architecture & Training (Slides 7--8)**:  
   Why plain MLPs fail (temporal signal washout and premature convergence). We detail Fourier positional embeddings, AdaLN scale/shift conditioning, zero-initialisation, and the 100-epoch requirement to escape initialization.
6. **Multi-Stream Scoring & The Arrow of Time (Slides 9--10)**:  
   We evaluate three orthogonal streams: global Hankel residual, latent CDC tangent force, and phase-space kinetic velocity residue. We explain why $z_{\text{implied}} = z_{\text{test}} - v_\theta$ is mathematically mandatory, and how projecting $v_\theta$ alone causes fatal algebraic cancellation.
7. **Cascaded Confidence Routing & Apodisation (Slides 11--12)**:  
   Why naive standardisation / max-pooling injects false alarms (+3$\sigma$ spikes). We use dynamic excursion ratios relative to the training envelope, coupled with center-weighted Gaussian apodisation and continuous Wiener soft projection.
8. **Benchmark Breakthrough & Verification (Slides 13--15)**:  
   On the full TSB-AD benchmark (857 evaluated datasets), Hankel CDC-FM achieves **`0.6157` Mean Range-Aware VUS-PR (median `0.6762`)**, officially beating the published benchmark SOTA of `0.59` unsupervised with a 90.4% win rate against deep autoencoders.

---

## Slide-by-Slide Script & Talking Points

### Slide 1: Why Deep Autoencoders Fail at Anomaly Detection
* **Objective**: Establish the problem immediately and explain why classical deep learning fails on this task.
* **Spoken Narrative**:
  > "Good morning everyone. Today I am presenting our work on Hankel Carré du Champ Conditional Flow Matching, or Hankel CDC-FM. To motivate why we built this method, we must first look at why standard deep neural autoencoders perform so poorly on time-series anomaly detection.
  > 
  > On the TSB-AD benchmark, standard autoencoders achieve an average VUS-PR of only 0.1165—essentially close to random chance. Why does this happen? In a standard neural autoencoder, the encoder is a deep, non-linear network. These networks have unconstrained null spaces. When an anomalous event arrives—such as a spike or an irregular pulse—the encoder simply absorbs the perturbation into the latent coordinates. As a result, the distance between normal and anomalous latent vectors is virtually indistinguishable. The decoder then reconstructs the anomaly almost as well as normal data, destroying our detection signal.
  > 
  > Our goal is therefore to build a mathematically guaranteed firewall that prevents anomalous energy from leaking into the latent representation."
* **Anticipated Question**: "Why don't regularisation methods like VAEs or contractive autoencoders fix this?"
* **Answer**: "VAEs impose a Gaussian prior, but the non-linear encoder still maps out-of-distribution points into the high-probability volume of the prior. Regularisation softens the space, but it does not provide an algebraic guarantee that out-of-subspace perturbations are rejected."

---

### Slide 2: The Two-Tier Geometric Solution
* **Objective**: Introduce the core architectural philosophy of two-tier geometric decoupling.
* **Spoken Narrative**:
  > "Instead of relying on a deep black-box encoder, we introduce an exact two-tier geometric decoupling.
  > 
  > In Tier 1, we perform a global linear subspace separation using Hankel SVD. We project sliding delay windows from ambient dimension $L$ onto an optimal $d$-dimensional dynamical subspace $U_d$. Crucially, any large off-manifold deviation is immediately routed into the orthogonal complement $U_\perp$.
  > 
  > In Tier 2, we zoom into that clean, low-dimensional coordinate space $\mathbb{R}^d$, where $d$ is typically 4 or 6. Here, the nominal dynamics form a smooth, curved manifold. We use the differential-geometric Carré du Champ operator to capture local curvature, and a continuous normalizing flow to model nominal transitions.
  > 
  > The mathematical foundation is the Algebraic Codimension Invariant shown at the bottom: $\|w\|_2^2 = \|z\|_2^2 + \|r_\perp\|_2^2$. Because the basis is strictly orthonormal, any orthogonal anomaly energy $a_\perp$ cannot be squashed or absorbed; it is algebraically preserved in the residual norm."
* **Transition**: "Let us look at how this subspace is constructed in closed form."

---

### Slide 3: Hankel Trajectory & Subspace Partitioning
* **Objective**: Walk through the closed-form Hankel SVD formulation.
* **Spoken Narrative**:
  > "We form delay-coordinate vectors $w_t$ of length $L$ from the univariate time series. Stacking these vectors across the training partition gives the block Hankel matrix $H_{\text{train}}$, from which we compute the lag-covariance Gramian matrix $C$.
  > 
  > By computing the eigendecomposition of $C$, we obtain the Karhunen--Loève orthonormal basis. We partition this basis into the top $d$ eigenvectors $U_d$, representing the active dynamical attractor, and the remaining $L-d$ eigenvectors $U_\perp$, representing the orthogonal null space.
  > 
  > For any incoming sliding window $w$, we can immediately compute its latent state $z = U_d^\top w$ and its macro-anomaly residual $r_\perp = (I - U_d U_d^\top)w$ in closed form. This requires zero iterative gradient descent and runs instantly."

---

### Slide 4: Setting $(L^*, d^*)$ via Physical Invariants
* **Objective**: Explain how hyperparameters are chosen unsupervised without label cheating or naive reconstruction sweeps.
* **Spoken Narrative**:
  > "A central question in delay embedding is: how do we choose the window length $L$ and the latent rank $d$?
  > 
  > In traditional machine learning, people often minimise in-sample reconstruction error. That fails completely here: if you let $d \to L$, reconstruction error trivially goes to zero, but you have eliminated the orthogonal complement, completely destroying anomaly detection.
  > 
  > Instead, we determine $(L^*, d^*)$ using physical dynamical invariants:
  > First, the Multi-Cycle Recurrence Principle: for periodic systems, $L$ must cover at least 3 to 5 fundamental periods $T_{\text{dom}}$. A single-cycle window contains no consecutive cycles to compare; phase slips and ectopic beats look normal within a single window. Expanding $L$ to 3 cycles on the UCR ECG benchmark lifted VUS-PR from 0.0034 to 0.8738—a 250-fold improvement.
  > Second, the Low-Dimensional Torus Ceiling: quasi-periodic attractors reside on low-dimensional tori $\mathbb{T}^1$ or $\mathbb{T}^2$. Each harmonic frequency needs exactly 2 dimensions (sine and cosine). Capping $d$ at 4 or 6 prevents the subspace from acquiring excessive capacity and absorbing anomalous shapes.
  > Third, we use the first trough of the autocorrelation function and spectral entropy to classify whether a signal is periodic or aperiodic without ad-hoc cutoffs."

---

### Slide 5: Local Latent Geometry via Carré du Champ (CDC)
* **Objective**: Explain how differential geometry models curvature in the latent space $\mathbb{R}^d$.
* **Spoken Narrative**:
  > "Once we have projected the data into $\mathbb{R}^d$, we need to model the local Riemannian geometry of the nominal manifold $\mathcal{M}$.
  > 
  > We construct a Theiler-windowed $k$-nearest neighbour graph. By enforcing $|t_i - t_j| > L$, we ensure that neighbours represent recurring dynamical orbits rather than trivial temporal correlations.
  > 
  > Next, we apply Coifman's $\alpha=1$ anisotropic diffusion kernel. This normalises out variations in sampling density along the attractor, isolating true manifold geometry.
  > 
  > From this, we compute the local Carré du Champ tensor $\Gamma_z(z_i)$. Diagonalising $\Gamma_z$ yields the local tangent basis $V^* \in \mathbb{R}^{d \times d_{\text{tan}}}$. This tells us, at any point on the attractor, which directions correspond to valid nominal motion, and which directions point orthogonally away from the manifold."

---

### Slide 6: Riemannian Conditional Flow Matching in $\mathbb{R}^d$
* **Objective**: Explain continuous flow matching in latent space with geometric conditioning.
* **Spoken Narrative**:
  > "To model transitions between states on the manifold, we use continuous-time generative flow matching in $\mathbb{R}^d$.
  > 
  > As shown on the slide, we define a continuous trajectory between a standard Gaussian noise prior $z_0$ at $t=0$ and the clean data manifold $z_1$ at $t=1$. Notice the third term: $t \sqrt{\Gamma_z(z_1)} z_0$. This injects the local Riemannian metric tensor directly into the flow path, aligning the vector field with the manifold curvature.
  > 
  > The conditional target velocity field $u_t$ is the exact time derivative of this trajectory. We train our neural velocity network $v_\theta(z_t, t)$ using simple mean-squared error.
  > 
  > Because the flow network operates strictly in $\mathbb{R}^d$ ($d \le 6$) rather than ambient space $\mathbb{R}^{128}$, it is extremely fast and free from ambient sensor noise."

---

### Slide 7: Neural Architecture: LatentFlowResNet
* **Objective**: Present the neural architecture and explain why ResNet replaced plain MLPs.
* **Spoken Narrative**:
  > "Slide 7 shows the architecture of our velocity backbone, LatentFlowResNet.
  > 
  > In earlier work, we tested simple MLPs, but they suffered from two major limitations: temporal signal washout and gradient instability. We resolved this through four architectural features:
  > 1. Sinusoidal Fourier Positional Encoding: Instead of passing a single scalar $t$, we project time into 32 multi-scale Fourier frequencies. This allows the network to distinguish subtle velocity changes as $t$ approaches 1.0.
  > 2. Adaptive Layer Normalisation (AdaLN): Rather than adding time at the first layer, time modulates each residual block via scale $\gamma(t)$ and shift $\beta(t)$. This ensures the temporal conditioning remains strong throughout the network.
  > 3. Zero-Initialisation: The final linear layer is initialised to exact zero weights and bias. At step zero, the predicted velocity is identically zero, preventing stochastic shocks from corrupting the flow early in training.
  > 4. Parameter Efficiency: The entire network has fewer than 50,000 parameters. It trains in seconds and evaluates batches in milliseconds."

---

### Slide 8: Training Dynamics & Epoch Calibration
* **Objective**: Explain the empirical discovery of the 100-epoch requirement and adaptive batching.
* **Spoken Narrative**:
  > "This brings us to a crucial finding regarding training dynamics.
  > 
  > Initially, models were trained for only 20 epochs with a fixed batch size of 256. On small datasets with short training splits—say, 500 to 1000 points—a batch size of 256 meant only one to four gradient steps per epoch!
  > Over 20 epochs, that was only 20 to 80 total updates from zero-initialisation. The network simply did not have enough steps to escape initialization.
  > 
  > As you can see in the empirical proof on the cardiac dataset SED: at 20 epochs, the score collapsed to 0.0815. At 50 epochs, it recovered to 0.9498; and at 100 epochs, it fully converged to 0.9532.
  > 
  > We implemented adaptive batch sizing, scaling $B$ so that every dataset receives at least 15 to 20 gradient steps per epoch. Combined with 100 epochs, this guarantees stable convergence across all domains."

---

### Slide 9: The Three Complementary Anomaly Streams
* **Objective**: Present the three detection metrics and explain their complementary physical roles.
* **Spoken Narrative**:
  > "At test time, for every sliding window, we evaluate three complementary geometric streams:
  > 
  > Stream 1 is the Global Hankel Orthogonal Metric. It measures the energy perpendicular to the global linear hyperplane $U_d$. This stream is sensitive to macro-anomalies: large spikes, step discontinuities, and foreign waveforms.
  > 
  > Stream 2 is the Local Latent CDC Orthogonal Force. Here, we evaluate whether the point violates the local tangent space $V^*$ on the curved manifold in $\mathbb{R}^d$. This detects subtle in-subspace anomalies where the window lies inside the linear hyperplane, but violates the non-linear manifold geometry—for example, abnormal cardiac pacing or subtle gait changes.
  > 
  > Stream 3 is the Phase-Space Velocity Residue. It measures deviations in kinetic rate $\Delta z(t)$ relative to nominal phase-space velocity, capturing rhythm accelerations and frequency shifts."

---

### Slide 10: The Arrow of Time & Fatal Cancellation Bug
* **Objective**: Explain the mathematical subtlety of the backwards flow and why $(I-P)z_{\text{implied}}$ is essential.
* **Spoken Narrative**:
  > "Slide 10 highlights a subtle but critical mathematical detail that we uncovered.
  > 
  > First, the Arrow of Time: forward flow travels from noise at $t=0$ to data at $t=1$. At $t=1$, the velocity vector points into the data manifold. To determine where a test sample originated, we must integrate backwards: $z_{\text{implied}} = z_{\text{test}} - v_\theta$.
  > 
  > Second, look at the equation in the middle of the slide. Suppose an anomalous sample has an orthogonal perturbation $a_\perp$. If we project $z_{\text{implied}}$, the term expands to $a_\perp + (I-P)(z_{\text{nom}} - v_\theta)$. The true physical anomaly energy $a_\perp$ directly drives the score.
  > 
  > However, if one mistakenly projects the velocity field $v_\theta$ alone—as some papers have done—$z_{\text{test}}$ and $a_\perp$ are cancelled out! The metric then measures only network leakage, destroying the contrast. On Yahoo dataset 618, projecting $z_{\text{implied}}$ scored a perfect 1.0000, whereas projecting $v_\theta$ collapsed to 0.0027."

---

### Slide 11: Cascaded Physical Confidence Routing
* **Objective**: Explain why naive pooling fails and how cascaded confidence routing solves multi-stream fusion.
* **Spoken Narrative**:
  > "Now, how do we combine these three streams?
  > 
  > A common instinct is to standardise each score to zero mean and unit variance, then take the maximum or average. That is fatal. If a stream has no signal on a given dataset—for example, if Latent CDC sees pure noise—standardising maps that noise to a standard Gaussian distribution. Gaussian noise naturally produces $+3\sigma$ excursions, creating severe false alarms that dragged down earlier benchmarks.
  > 
  > We solved this with Cascaded Physical Confidence Routing. Anomalies are defined relative to the training envelope. We compute the Dynamic Excursion Ratio $R_k$, which is the 99.5th percentile of the test score divided by the 99.5th percentile of the training score.
  > 
  > Our decision rule has three clear stages:
  > 1. If kinetic velocity shows a strong excursion, route to Velocity.
  > 2. If the signal has sufficient training history and Flow shows a clear departure over Hankel, route to Flow (for example, on biomechanical human activity data).
  > 3. If training history is short ($N_{\text{tr}} < 600$), strictly protect against neural hallucinations by defaulting to the deterministic Hankel projector.
  > 
  > This routing mechanism lifted our benchmark score from 0.6894 to 0.8037 on our pilot suite."

---

### Slide 12: Pointwise Apodisation & Soft Projector
* **Objective**: Detail the signal processing enhancements: Gaussian apodisation and Wiener soft filtering.
* **Spoken Narrative**:
  > "Slide 12 covers two essential signal processing steps:
  > 
  > First, converting window scores back to raw time points. When sliding windows overlap, taking an arithmetic average dilutes sharp anomalous spikes by up to 86.5%, because windows that only clip the edge of an anomaly dilute the signal with 99% normal data. We replace arithmetic averaging with center-weighted Gaussian apodisation, concentrating energy at the window centers where the anomaly is fully contained.
  > 
  > Second, continuous Wiener soft projection. Hard integer truncation of rank $d$ can cause jitter when adjacent singular values are close. We replace the hard projector with a continuous Wiener shrinkage filter. Dominant signal modes are projected out, noise modes are retained, and transitional modes are smoothly attenuated."

---

### Slide 13: Full Benchmark Verification (857 Datasets)
* **Objective**: Deliver the headline benchmark results on the official TSB-AD benchmark.
* **Spoken Narrative**:
  > "We evaluated Hankel CDC-FM across the complete TSB-AD benchmark, running over 5 distributed NVIDIA RTX A5000 GPUs across 857 evaluated datasets.
  > 
  > As shown in the table:
  > - Deep neural autoencoders score 0.1165 mean VUS-PR.
  > - Exhaustive grid search on classical Hankel achieves 0.4749.
  > - Our unsupervised physical Hankel Orthogonal baseline achieves 0.5796.
  > - And our upgraded Cascaded Routed Hankel CDC-FM pipeline achieves **0.6157 Mean Range-Aware VUS-PR**, with a median of **0.6762**.
  > 
  > This officially surpasses the published TSB-AD-U benchmark SOTA of 0.59. It is completely unsupervised, uses zero labels, and achieves a 90.4% win rate against deep autoencoders."

---

### Slide 14: Key Domain Breakthroughs
* **Objective**: Show specific domain case studies proving why multi-stream routing works.
* **Spoken Narrative**:
  > "Slide 14 breaks down where the gains came from across diverse domains:
  > 
  > - On OPPORTUNITY human activity telemetry, motion occurs on continuous, curved Riemannian manifolds. Hankel orthogonal scored only 0.1451, whereas Flow Matching scored 0.8389. The router selected Flow, lifting domain performance to 0.6803.
  > - On YAHOO web traffic, time series are short (500 steps). Neural flows tended to hallucinate test oscillations. Small-sample protection held the model on deterministic Hankel, preserving high scores around 0.7024.
  > - On Mackey-Glass chaotic series (MGAB), Hankel was blind (0.0113), but Flow achieved 0.4509; the subspace blindness test routed 100% of these series to Flow.
  > - On NASA spacecraft telemetry (SMAP) and big data clusters (Exathlon), closed-form Hankel dominates with scores over 0.92 to 0.96.
  > 
  > The key takeaway is that no single representation wins across all 870 datasets; dynamic confidence routing is the catalyst that unlocks top performance."

---

### Slide 15: Summary & Next Steps
* **Objective**: Conclude clearly and preview upcoming directions.
* **Spoken Narrative**:
  > "To conclude:
  > - We have shown that two-tier geometric decoupling solves the autoencoder null-space problem with mathematical guarantees.
  > - Physical invariants eliminate arbitrary $(L, d)$ parameter searching.
  > - LatentFlowResNet provides stable generative flow modeling in low-dimensional latent space.
  > - And Cascaded Confidence Routing brings everything together to set a new unsupervised benchmark SOTA of 0.6157 on TSB-AD.
  > 
  > Looking ahead, we are exploring two new paradigms:
  > 1. Functional Delay Embeddings (FDE): projecting continuous history onto shifted Legendre polynomial jets, using the HiPPO continuous state-space ODE to update representations in $O(1)$ time.
  > 2. Geometric Simplex Autoencoders (GeoSimplex-CDC): using convex polytope unmixing with strictly linear decoders to establish a geometric firewall in deep autoencoder architectures.
  > 
  > Thank you, and I welcome any questions."

---

## Quick Q&A Cheat Sheet for the Meeting

1. **"Why use VUS-PR instead of AUROC or standard Precision/Recall?"**  
   *Answer*: Standard point-wise AUROC is heavily distorted in time series by class imbalance and point-adjustment tricks. Range-Aware VUS-PR (Volume Under the Surface of Precision-Recall across varying anomaly buffer sizes) is the official standard in TSB-AD because it rigorously evaluates both temporal detection precision and sequence boundary overlap without artificial point-adjustment inflation.

2. **"Does Hankel SVD struggle with long time series?"**  
   *Answer*: No, because the covariance matrix $C \in \mathbb{R}^{L \times L}$ depends only on window length $L \approx 64\text{--}128$, not on series length $N_{\text{tr}}$. Computing $C = \frac{1}{K} H H^\top$ takes less than 50 milliseconds on CPU, and eigendecomposition of a $128 \times 128$ matrix takes under 10 milliseconds.

3. **"Why is the batch size scaled dynamically in Slide 8?"**  
   *Answer*: Fixed batch sizes (e.g. $B=256$) starve small datasets of updates. If $N_{\text{tr}} = 500$, one epoch has only 2 batches, meaning 40 updates over 20 epochs. Scaling $B = \min(128, \max(32, \lfloor N_{\text{tr}} / 15 \rfloor))$ guarantees that every training run gets at least 1,500 to 2,000 updates over 100 epochs, ensuring the ResNet escapes its zero-initialised starting point.

4. **"How does this relate to the HiPPO / FDE work mentioned on the final slide?"**  
   *Answer*: Discrete Hankel matrices couple window length $L$ to the sampling rate $\Delta t$. Functional Delay Embedding projects the continuous history into an orthonormal polynomial basis, decoupling memory horizon from dimension, and allows $O(1)$ recursive updating via continuous state-space models. That is our primary theoretical bridge for the next chapter.
