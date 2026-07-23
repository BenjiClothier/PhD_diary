# Weekly Log: [Date Range]

## 🎯 Focus of the Week
Investigate and justify CDC, evaluate a more challenging baseline, and conduct ablation studies on the effectiveness of the Markov transition matrix $P$ (comparing no matrix, $W$, $\tilde{W}$, and $P$).
---

## 📝 To-Do List
- [ ] Justify CDC
- [x] Find and evaluate a harder baseline (Note: this task took up all of this week's time)
- [ ] Perform ablation study on the effectiveness of the transition matrix $P$ (test: no $P$, use $W$, use $\tilde{W}$, use $P$)

---

## 🔬 Progress & Experiments

[TSB-AD Webpage](https://thedatumorg.github.io/TSB-AD/#:~:text=Example%20time%20series%20from%20TSB%2DAD%2C%20with%20anomalies,performance.%20Last%20updated%3A%20July%201%2C%202026.%20Abstract.)

### Why CDC and not LPCA

- **Euclidean Proximity vs Diffusion Geometry**
    $$\Delta_{x_i} = x_i - x$$
    $$X_{\text{PCA}} = \begin{bmatrix} 
            \frac{1}{\sqrt{K}} \, \Delta_{x_1}^T \\
            \vdots \\
            \frac{1}{\sqrt{K}} \, \Delta_{x_K}^T
            \end{bmatrix}$$
    $$C = X_{\text{PCA}}^T X_{\text{PCA}}$$


    $$\tilde{X}_{\text{CDC}} = \begin{bmatrix} 
            \sqrt{P_{{x} \rightarrow x_1}} \, \Delta_{x_1}^T \\
            \vdots \\
            \sqrt{P_{{x} \rightarrow x_K}} \, \Delta_{x_K}^T
            \end{bmatrix}$$

    $$G = \tilde{X}_{\text{CDC}}^T \tilde{X}_{\text{CDC}}$$

    $$\underbrace{\Gamma(f, h) = \tilde{X}_{\text{CDC}}^T \tilde{X}_{\text{CDC}} \quad \xrightarrow{\text{limit}} \quad g(\nabla f, \nabla h)}_{{\text{Provable convergence to the Riemannian Metric Tensor}}}$$
---

# CDC for Time-Series: Phase Space Anomaly Detection

This document outlines the architectural pipeline for adapting the **Carré du Champ (CDC) Riemannian Manifold** framework to univariate time-series data using the TSB-AD benchmark. This approach upgrades traditional linear Subsequence PCA into a dynamically dimensioned, non-linear topological model.

## The Architecture Pipeline

### Phase 1: Phase-Space Embedding (Taken's Theorem)
Instead of processing time-series as isolated scalars, we embed the temporal dynamics into a high-dimensional geometric space.
1. **Sliding Window:** Slice the 1D training time-series into overlapping windows of length $L$ (e.g., $L=100$).
2. **Vectorization:** Each window becomes a single point in $\mathbb{R}^L$ space. The continuous sequence of these points traces out the "Strange Attractor" (the healthy manifold) of the dynamical system.

### Phase 2: The Vector Bank & Density De-biasing
Unlike spatial image features, 1D time-series windows are computationally lightweight. We do not need a Greedy k-Center Coreset to compress the space.
1. **Full Memory Bank:** We retain *all* overlapping $\mathbb{R}^L$ phase-space windows from the training set. This complete set becomes our dense **Vector Bank**.
2. **Theiler Window Exclusion:** In time-series, overlapping windows are trivially identical. When constructing the graph and searching for neighbors, we must mathematically enforce a "Theiler Window" (forbid matches within $\pm L$ timesteps). This forces the manifold to map *structural cycles* rather than just sliding along the exact same temporal wave.
3. **Coifman-Lafon Graph:** Compute the Heat Kernel affinities between structurally distant vectors and normalize them by local density ($q_i = \sum W_{ij}$). This produces the Markov Transition Matrix ($P$), ensuring the geometry is immune to the sampling density of common frequencies.

### Phase 3: Dynamic Tangent Planes
For every vector in the dense Memory Bank, we calculate its local geometric surface.
1. **Anchored Neighborhood SVD:** Find the $k=25$ nearest neighbors in $\mathbb{R}^L$ space (respecting the Theiler Window). Instead of using the mean-centered neighborhood covariance (which blurs anomaly boundaries), we compute the displacement vectors exactly anchored at the healthy center point ($\Delta = X_{neighbors} - X_{center}$).
2. **Markov Weighting:** We weight these displacement vectors using the Coifman-Lafon row-stochastic Markov transition matrix ($P$) and compute the Singular Value Decomposition to form a rigid, strictly anchored Tangent Plane.
3. **Participation Ratio (PR):** Use the continuous PR metric on the eigenvalues to find the intrinsic dimensionality $d$. The top $d$ eigenvectors define the Tangent Plane (the structural physics of the signal), while the remaining $L-d$ vectors represent ambient orthogonal noise (sensor noise).

### Phase 4: Inference & Orthogonal Scoring
When evaluating a test time-series for anomalies:
1. Slice the test sequence into the same $L$-length windows.
2. For each test window, find its nearest Healthy Vector in the Memory Bank.
3. Project the test window onto the vector's pre-computed Tangent Plane.
4. Calculate the Orthogonal distance $||V_{ortho}||$.
5. **Reconstruction:** Map the window-level anomaly scores back to point-level scores by averaging the orthogonal distances of all windows that overlap a specific timestamp.

## 1. Principal Component Analysis (Global Variance)

Principal Component Analysis (PCA) operates under the assumption that the healthy dynamical system can be approximated by a single, global, low-dimensional linear subspace. 

Given a matrix of healthy phase-space windows $\mathbf{W} \in \mathbb{R}^{M \times L}$, where $M$ is the total number of extracted windows and $L$ is the window length. PCA calculates the global mean vector $\bar{w}$ and centres the data. It then computes the global covariance matrix:
$$ \Sigma = \frac{1}{M-1} (\mathbf{W} - \bar{w})^T (\mathbf{W} - \bar{w}) $$

By performing Eigenvalue Decomposition on $\Sigma$, we extract the ordered eigenvectors (principal components). The first $d$ components define a global hyper-plane that captures the majority of the system's variance. 

During inference, a test window $w_{test}$ is projected onto this global hyper-plane. The anomaly score is defined as the orthogonal reconstruction error, the Euclidean distance between the original point and its projection on the plane.

### 1.1 Mathematical Correction of the PCA Baseline
In the foundational TSB-AD benchmark, the reported baseline performance for PCA across the entire corpus was $0.42$ VUS-PR. However, our evaluation yielded a significantly higher baseline performance of $0.51$ VUS-PR. 

This discrepancy arises from a geometric flaw in the standard library adaptation used by the original benchmark. The original code computes the outlier decision score by measuring the raw pairwise Euclidean distance between the data window and the unit eigenvectors themselves:
$$ \text{Score}_{\text{original}} = \sum_{j} || x_i - v_j || $$
where $v_j$ is the $j$-th eigenvector. Computing the Euclidean distance to a unit vector representing a directional axis is meaningless in the context of phase-space reconstruction.

```python
self.decision_scores_ = np.sum(
    cdist(X, self.selected_components_) / self.selected_w_components_,
    axis=1).ravel()
```

To provide a robust baseline, we implemented true **Subsequence PCA**. We project the phase-space window onto the hyper-plane formed by the top $K$ principal components, reconstruct the window, and measure the exact orthogonal rejection:
$$ \text{Score}_{\text{robust}} = || x_i - \text{Proj}_{\text{span}(v_1, \dots, v_K)}(x_i) || $$

```python
    # Project data onto the top k eigenvectors
    coeffs = torch.matmul(X_c, Vh_k.T)
    # Reconstruct the data using only those k planes
    X_recon = torch.matmul(coeffs, Vh_k)
    # The anomaly score is the orthogonal distance from the plane
    error = torch.norm(X_c - X_recon, dim=1)
```

By correcting this geometric fallacy, our robust PCA implementation achieves a $0.51$ VUS-PR, serving as a formidable global variance benchmark against which our local CDC operator is evaluated.

## 2. The Carré du Champ Operator (Local Differential Geometry)

To overcome the stationary limitations of global PCA, we adapt the **Carré du Champ (CDC)** operator, a concept derived from Riemannian geometry and Markov diffusion processes. Rather than fitting a single plane to the entire dataset, CDC computes a Tangent Plane for every data point.

### 2.1 The Density-Invariant Markov Graph
To construct the local geometry, we first calculate the pairwise Euclidean distances between all windows in the phase space. To ensure the geometry is strictly structural (ignoring the density of frequently repeating time-series cycles), we construct a graph using the Coifman-Lafon adaptive bandwidth kernel:
$$ K(x, y) = \exp \left( - \frac{||x - y||^2}{\sigma_x \sigma_y} \right) $$
where $\sigma_x$ is the distance from $x$ to its $k$-th nearest neighbour. This kernel is normalised by the local degree $q(x) = \sum K(x,y)$ to produce the row-stochastic Markov transition matrix $P$.

### 2.2 The Anchored Local SVD
For a specific healthy point $x_0$, we isolate its $k$-nearest neighbours $\mathcal{N}(x_0)$. Unlike local PCA, which would centre this neighbourhood around its local mean, CDC anchors the geometry at $x_0$. 

We construct a displacement matrix $\Delta$ where each row is the vector $(x_i - x_0)$ for $x_i \in \mathcal{N}(x_0)$. These displacement vectors are scaled by their Markov transition probabilities from $P$, and a Singular Value Decomposition (SVD) is performed.

The top $d$ eigenvectors form a localised Tangent Plane $T_{x_0}\mathcal{M}$ that descirbes the curvature of the manifold at the coordinate in phase space.

### 2.3 Orthogonal Distance as Anomaly Scoring
During inference, a test window $w_{test}$ searches the memory bank for its closest healthy neighbour $x_0$. The test point is projected onto the pre-computed Tangent Plane $T_{x_0}\mathcal{M}$. The anomaly score is calculated as the orthogonal rejection:
$$ \text{Score}(w_{test}) = || (w_{test} - x_0) - \text{Proj}_{T_{x_0}\mathcal{M}}(w_{test} - x_0) || $$

### 2.4 Global Baseline Performance
Across the entire TSB-AD benchmark corpus (comprising 870 diverse datasets), the raw Carré du Champ (CDC) operator achieved a global average performance of $0.488$ VUS-PR. While this global average is slightly lower than our highly-optimised Subsequence PCA baseline ($0.511$ VUS-PR) on short, stationary datasets, CDC's true geometric power is revealed in the presence of complex structural anomalies and massive concept drift (detailed in Section 10), where local geometry outperforms global statistical methods.

Remarkably, this $0.488$ global average was achieved without tuning parameters per dataset. The exact same universal geometry was applied to all 870 datasets:
*   **Phase-Space Window ($L$):** 50
*   **Local Neighbourhood ($k$-NN):** 25
*   **SVD Variance Threshold:** $90\%$ (Dynamic Dimensionality)
*   **Window Z-Norm:** Disabled (Raw Spatial Geometry)

---

# Phase-Space Embedding: Dimensionality, Delay, and Over-Embedding

The performance of any geometric anomaly detector is fundamentally bound by the quality of the phase space into which the time-series is embedded. If the strange attractor is improperly constructed, the subsequent SVD Tangent Planes will fail to capture the true structural physics of the signal. In this section, we analyse the constraints of Takens' Time-Delay Embedding and propose a novel frequency-based Over-Embedding strategy to stabilise local topological computations.

## 3. The Takens' Time-Delay Dilemma

Takens' embedding theorem dictates that a multi-dimensional strange attractor can be reconstructed from a 1D observation $x(t)$ using a time-delay parameter $\tau$ and an embedding dimension $L$, such that a point in phase space is defined as:
$$ w_t = [x_t, x_{t+\tau}, x_{t+2\tau}, \dots, x_{t+(L-1)\tau}] $$

In standard machine learning implementations (such as convolutional autoencoders or moving-average models), it is universally standard to use strictly contiguous windows where $\tau = 1$. 

### 3.1 The Collinearity Feature
From a dynamical systems perspective, $\tau=1$ is flawed for highly-sampled physical data (e.g., ECG sampled at 500 Hz). The temporal distance between $x_t$ and $x_{t+1}$ is so small that the physical system has not changed state, leading to extreme collinearity within the $\mathbb{R}^L$ vector. 

Our preliminary ablations suggest an interesting trade-off: **In certain anomaly detection regimes, this collinearity may act as a feature rather than a bug.** 
While calculating a dynamic optimal $\tau$ (via the first zero-crossing of the Autocorrelation Function) decorrelates the axes and technically satisfies Takens' theorem, it creates a practical physical flaw. To maintain a rich $L$-dimensional space with a large $\tau$, the total physical time spanned by a single window becomes massive. For example, a window with $L=50$ and $\tau=18$ spans 883 timestamps. 
If an anomaly is a brief transient spike lasting 10 timestamps, an 883-point window completely swallows it, severely degrading point-wise detection precision. 

By enforcing $\tau=1$ with a large embedding dimension ($L=50$), the extreme collinearity forces the primary SVD eigenvectors to strictly trace the continuous shape of the physical wave. We hypothesise that this provides the CDC operator with a sufficiently rich space to compute curvature, while maintaining a 50-point physical resolution that remains highly sensitive to localised shape deformations.

## 4. Geometric Over-Embedding (DCT-CDC)

The trade-off in Takens' embedding is the conflict between SVD stability (which requires a high embedding dimension $L$) and temporal resolution (which limits the physical span of the window). If a dataset requires a small physical span (e.g., 10 points) to capture brief anomalies, setting $L=10$ starves the SVD algorithm. Computing a robust, local 12-neighbour Tangent Plane inside an $\mathbb{R}^{10}$ space often lacks the topological complexity to isolate subtle anomalies.

To solve this, we introduce **Geometric Over-Embedding** using the Discrete Cosine Transform (DCT-II). 

Instead of embedding raw temporal amplitudes, we map a large physical window (e.g., 50 contiguous points) into the frequency domain. We then truncate the output to the lowest $L$ coefficients:
$$ c_k = \sum_{n=0}^{N-1} x_n \cos \left( \frac{\pi k (2n+1)}{2N} \right) \quad \text{for } k=0, \dots, L-1 $$

By setting $N=50$ and $L=10$, we mathematically compress the dominant structure of a 50-point temporal horizon into a highly dense $\mathbb{R}^{10}$ space. 

### 4.1 Empirical Validation
This strategy operates as a natural topological low-pass filter, discarding high-frequency sensor jitter while preserving the core structural variance. 

**The Success (Structural Datasets):** In benchmarking against UCR Medical and ECG datasets, DCT-CDC (compressing $L=50$ to 10 coefficients) consistently outperformed the raw 50-dimensional temporal embedding, demonstrating up to a $+6\%$ boost in VUS-PR. By dropping high-frequency noise, the SVD Tangent Plane was perfectly stabilised to detect subtle macroscopic structural deformations.

**The Failure (High-Frequency Datasets):** Conversely, on datasets such as TAO (environmental) and OPPORTUNITY (human activity motion), Geometric Over-Embedding resulted in a minor performance degradation (dropping VUS-PR by roughly $1\% - 2\%$). In these categories, the "high-frequency jitter" is not ambient noise—it is the anomaly itself. Truncating the high-frequency DCT coefficients effectively destroyed the topological signatures of the transient anomalies, blinding the CDC operator. 

Ultimately, Geometric Over-Embedding decouples the physical resolution of the window from the mathematical dimensionality required for SVD, providing a foundation for localised differential geometry, but it must only be applied to systems where anomalies manifest as structural deformations rather than high-frequency transients.

---

# Manifold Stability & Topological Denoising

Even when successfully embedded into a high-dimensional phase space, raw physical time-series often contain inherent stochasticity, background sensor noise, or structural non-stationarity. These factors can distort Euclidean distances and fragment the local $k$-nearest neighbour graph, undermining the Carré du Champ (CDC) operator. In this section, we analyse two fundamental geometric vulnerabilities, amplitude shifts and topological noise, and propose mathematical solutions to stabilise the manifold.

## 5. Addressing Amplitude Shifts: Window-Level Standardisation

Because the CDC operator anchors its Tangent Planes using local Euclidean displacement vectors, it is fundamentally highly sensitive to raw magnitude. While CDC is intrinsically scale-invariant relative to global dataset variance (due to the Coifman-Lafon adaptive bandwidth), it remains mathematically vulnerable to localised *baseline shifts*.

In datasets characterised by stochastic point-spikes or perfect flatlines (e.g., WebService logs), the system's underlying dynamics may not change, but a sudden shift in the absolute amplitude creates massive Euclidean displacement in $\mathbb{R}^L$ space. This shatters the $k$-NN graph, causing the CDC operator to fail to find true structural neighbours.

### 5.1 Window Normalisation: Raw Amplitude vs. Structural Shape
To map structural variance independently of amplitude, window-level Z-Score normalisation can be applied prior to computing the distance matrix:
$$ \hat{w}_t = \frac{w_t - \mu(w_t)}{\sigma(w_t) + \epsilon} $$

However, window standardisation introduces a trade-off depending on the anomaly type:
1.  **Amplitude Anomalies:** For anomalies defined by magnitude spikes (e.g., Financial market shocks), Z-Norm standardises the variance to $1.0$, reducing their detectability in the phase space.
2.  **Structural Anomalies:** For structural deformations obscured by baseline drift (e.g., Human Activity state changes), Z-Norm centres the data, isolating the underlying waveform shape.

**Empirical Validation:**
To evaluate this trade-off, we conducted an ablation over 689 TSB-AD datasets, computing Tangent Planes for both raw amplitude (Raw CDC) and standardised structure (Z-Norm CDC), and outputting the maximum anomaly score per point.

The aggregated results are:
*   **Average CDC Raw:** $0.521$ VUS-PR
*   **Average CDC Z-Norm:** $0.331$ VUS-PR
*   **CDC Max (Raw vs. Z-Norm):** $0.601$ VUS-PR

The aggregate increase to $0.601$ VUS-PR indicates that time-series anomalies manifest across both amplitude and structural domains. The category breakdown supports this: in Financial data, Z-Norm underperformed Raw CDC in all instances due to the amplitude-dependent nature of the anomalies. Conversely, in Human Activity and WebService datasets, Z-Norm outperformed Raw CDC in the majority of cases by mitigating the effect of wandering baselines.

## 6. Graph Diffusion via Ricci Curvature Flow

In many real-world datasets, the phase-space geometry suffers from poor structural separation. The healthy data forms broad, diffuse clusters, while anomalous points often reside in sparse, ambiguous boundary regions. To improve the fundamental geometric representation of the data before SVD projection, we apply discrete graph diffusion using Ricci Curvature Flow. This process actively manipulates the underlying metric space by compressing dense clusters of healthy data and expanding sparse regions, effectively pushing anomalies further away from the healthy manifold.

### 6.1 Forman-Ricci vs. Bakry-Émery vs. The CDC Operator
While both Forman and Bakry-Émery (BE) evaluate discrete Ricci curvature to manipulate the metric space, they approach the geometry from entirely different mathematical paradigms:

1.  **Forman-Ricci (Combinatorial):** This method is purely topological and geometric. It evaluates curvature by simply counting integer combinations of node degrees, shared edges, and triangles (2-cells). It is highly computationally efficient but completely ignores the diffusion dynamics of the time-series.
2.  **Bakry-Émery Ricci (Probabilistic):** This method is rooted in Markov diffusions and optimal transport. It evaluates how much two random walks (starting at nodes $x$ and $y$) naturally converge towards each other. This is computed using the 1st Wasserstein distance ($W_1$) between their respective transition probability distributions, defined as $W_1(m_x, m_y) = (1 - \text{Ricci}(x, y)) \cdot d(x, y)$. If the probability distributions heavily overlap, the Earth Mover's (Wasserstein) distance is small, yielding a positive curvature that identifies a dense, tightly bound cluster.

By applying the Ricci flow equation, the distance matrix is iteratively updated:
$$ D_{t+1}(x,y) = D_t(x,y) - \alpha \cdot \text{Ricci}(x,y) \cdot D_t(x,y) $$
This non-linear diffusion dynamically shrinks positively curved edges (tightening the healthy manifold) while expanding negatively curved edges (pushing stochastic outliers further away).

---

# Multi-Scale Geometry & The Aggregation Paradox

A constraint in anomaly detection is the trade-off between macro-structural sensitivity and micro-transient precision. A large embedding window smoothly captures complex structural waves but inevitably blurs brief, localised point-spikes. Conversely, a tight temporal window perfectly isolates point-spikes but lacks the physical horizon to detect slow structural deformations. 

To create a mathematically universal anomaly detector, we proposed a **Hierarchical Multi-Scale Cascade** that computes dual-scale CDC embeddings and fuses their topological rejections.

## 7. The Dual-Scale CDC Cascade

By utilising Takens' Time-Delay Embedding, we generate two distinct phase-space manifolds simultaneously:
1.  **Scale A (Macro):** Engineered to map the long-term structural "background" manifold. We set $L=10$ and $\tau=10$, capturing a broad physical span of 91 timestamps.
2.  **Scale B (Micro):** Engineered to map high-frequency, transient geometry. We set $L=10$ and $\tau=1$, capturing a tight 10-point physical span.

Because both scales share the same mathematical dimensionality ($L=10$), we strictly stabilise the SVD by fixing the local neighbourhood to $k_{nn} = 12$. For a given test point, orthogonal anomaly scores are independently calculated for both Scale A and Scale B. 

## 8. The Aggregation Difficulty

A mathematical challenge of multi-scale modelling lies in fusing the independent signals. The aggregation function must dynamically determine which topological scale contains the true anomaly without amplifying the background noise of the alternate scale. Our empirical tests isolated two diametrically opposed aggregation behaviours, revealing a fundamental challenge in multi-scale signal processing.

### 8.1 The Harmonic Mean (Logical AND)
Initially, the dual scores were normalised to $[0, 1]$ and aggregated using a Harmonic Mean. This function strictly operates as a "Logical AND" gate—it heavily penalises the final score if either scale outputs a value near zero.

*   **The Success:** On stochastic datasets dominated by point-spikes (e.g., Yahoo WebService data), the Harmonic Mean succeeded, boosting VUS-PR from $0.68$ (Raw CDC) to **$0.90$**. Because a point-spike registers as an anomaly in *both* the Micro and Macro scales, the Harmonic Mean spiked violently. During healthy periods, the Micro scale naturally outputs zero, perfectly silencing any false-positive noise generated by the Macro scale.
*   **The Failure:** On Medical datasets characterised by 50-point structural shape deformations, performance collapsed. While the Macro scale successfully detected the deformation, the Micro scale (spanning only 10 points) was physically blind to the macroscopic shape change, scoring the anomaly as a healthy $0.0$. The Harmonic Mean subsequently pulled the final score to zero, effectively allowing the Micro scale to veto the Macro scale.

### 8.2 The Maximum Filter (Logical OR)
To prevent the Micro scale from blinding the detector to structural anomalies, we replaced the Harmonic Mean with a Maximum Filter ($\max(\text{Scale}_A, \text{Scale}_B)$). This operates as a "Logical OR" gate.

*   **The Success:** On the Medical dataset, the Maximum filter allowed the Macro scale's detection to pass through uninhibited, recovering the detection accuracy that the Harmonic Mean had destroyed.
*   **The Failure:** On the Yahoo point-spike dataset, performance crashed from $0.90$ down to $0.62$. Because the 91-point Macro window is too wide to cleanly isolate point-spikes, its baseline output is inherently noisy. The Maximum Filter preserved all of the Macro scale's false-positive noise, completely destroying the model's Precision.

### 8.3 Conclusion on Multi-Scale Aggregation
The Hierarchical Cascade proved that topological anomalies exist simultaneously on completely different physical scales, and CDC is fully capable of mathematically isolating them. However, a static scalar aggregation function is fundamentally insufficient to universally resolve the **Aggregation Difficulty**. 

For true multi-scale robustness without neural attention networks, frequency-based methods—such as the DCT-Embedded CDC (Section 2)—remain the superior geometric solution. By packing all scales of variance into a unified frequency space and dropping the high-frequency jitter, the manifold naturally achieves multi-scale sensitivity within a single, unified phase space.

---

# Robustness to Concept Drift: The Massive Benchmark

A test of any time-series anomaly detection algorithm is its stability over long temporal horizons. In highly-sampled industrial or biological settings, the underlying physical system rarely maintains a strict, static baseline. Sensors experience thermal drift, biological metrics undergo circadian shifts, and mechanical components wear down over time. This phenomenon, known as **Concept Drift**, forces the healthy state of the system to physically migrate across the phase space.

To test the resilience of our topological framework against concept drift, we executed a sweep across the TSB-AD benchmark, specifically isolating datasets exceeding 25,000 continuous timestamps. We evaluated the geometric degradation of Global PCA against the stability of Local CDC.

## 9. The Collapse of Global Variance

As established in Section 1, Principal Component Analysis (PCA) relies on a single, global covariance matrix $\Sigma$ calculated relative to a static, global mean vector $\bar{w}$. 

When a dataset experiences concept drift, the phase-space coordinates of the healthy manifold slowly migrate away from the original global mean. Because PCA's hyper-plane is rigidly locked to this static origin, the migrating healthy data is geometrically interpreted as moving orthogonally away from the plane. 

### 9.1 Empirical Degradation
Our benchmark empirically proved this mathematical vulnerability. On small, stationary datasets, raw PCA performed exceptionally well (averaging $\sim0.62$ VUS-PR), effectively identifying simple stochastic anomalies. However, as the temporal horizon expanded past 25,000 points and concept drift was introduced, PCA's performance collapsed. 
The rigid global hyper-plane began flagging normal, drifting baselines as highly anomalous, flooding the evaluation metrics with false positives and dragging the massive dataset average down to roughly **$0.25$ VUS-PR**. 

## 10. The Resilience of Local Differential Geometry

Conversely, the Carré du Champ (CDC) operator constructs its geometry using strictly localised neighbourhoods. For any given test point $w_{test}$, the algorithm queries the memory bank for its nearest neighbour $x_0$, completely agnostic to the global mean of the entire dataset. 

The Tangent Plane $T_{x_0}\mathcal{M}$ is constructed using only the $k$-nearest neighbours surrounding $x_0$. 

### 10.1 Empirical Stability
The empirical results of the massive dataset sweep definitively validated this structural resilience. While PCA collapsed under the weight of concept drift, Raw CDC remained remarkably stable, maintaining an average of **$\sim0.36$ VUS-PR** across the massive, non-stationary datasets—nearly double the performance of PCA. 

## 📊 Results & Insights

# VUS-PR Analysis Report

## Overall Performance

| Method | Average VUS-PR |
|--------|----------------|
| PCA_Raw | 0.5115 |
| CDC_Raw | 0.4821 |
| CDC_ZNorm | 0.2960 |
| CDC_1Bit_Max | 0.5472 |

## Category (Domain) Breakdown
| Domain        |   PCA_Raw |   CDC_Raw |   CDC_ZNorm |   CDC_1Bit_Max |
|:--------------|----------:|----------:|------------:|---------------:|
| Environment   | 0.599488  |  0.548702 |    0.378246 |       0.589365 |
| Facility      | 0.436605  |  0.505226 |    0.152978 |       0.518637 |
| Finance       | 0.910269  |  0.907291 |    0.803092 |       0.907291 |
| HumanActivity | 0.0904221 |  0.154902 |    0.150016 |       0.215394 |
| Medical       | 0.203449  |  0.404115 |    0.222763 |       0.420933 |
| Sensor        | 0.451731  |  0.534218 |    0.230278 |       0.5658   |
| Synthetic     | 0.648156  |  0.635569 |    0.360936 |       0.656227 |
| Traffic       | 0.307643  |  0.288037 |    0.171368 |       0.289161 |
| WebService    | 0.698078  |  0.474506 |    0.370481 |       0.615789 |

## Source Breakdown
| Source      |   PCA_Raw |     CDC_Raw |   CDC_ZNorm |   CDC_1Bit_Max |
|:------------|----------:|------------:|------------:|---------------:|
| CATSv2      | 0.347584  | nan         | nan         |     nan        |
| Daphnet     | 0.422074  |   0.391939  |   0.0376178 |       0.391939 |
| Exathlon    | 0.816916  |   0.774509  |   0.360584  |       0.799929 |
| IOPS        | 0.591494  |   0.397754  |   0.0640213 |       0.405601 |
| LTDB        | 0.264065  |   0.425731  |   0.354734  |       0.458837 |
| MGAB        | 0.0108533 |   0.66579   |   0.508845  |       0.665913 |
| MITDB       | 0.0524074 |   0.10357   |   0.0638095 |       0.114652 |
| MSL         | 0.477136  |   0.489645  |   0.211238  |       0.535098 |
| NAB         | 0.374729  |   0.343555  |   0.188544  |       0.360214 |
| NEK         | 0.868902  |   0.913572  |   0.340908  |       0.913572 |
| OPPORTUNITY | 0.059175  |   0.091157  |   0.181147  |       0.2007   |
| Power       | 0.0778971 |   0.0879427 |   0.110295  |       0.110295 |
| SED         | 0.0503627 |   0.772896  |   0.88215   |       0.88215  |
| SMAP        | 0.594024  |   0.599822  |   0.324732  |       0.644777 |
| SMD         | 0.764946  |   0.759334  |   0.0549994 |       0.759371 |
| SVDB        | 0.113108  |   0.208349  |   0.08525   |       0.208349 |
| SWaT        | 0.112363  | nan         | nan         |     nan        |
| Stock       | 0.910269  |   0.907291  |   0.803092  |       0.907291 |
| TAO         | 0.918831  |   0.916066  |   0.810628  |       0.916066 |
| TODS        | 0.756913  |   0.834429  |   0.820569  |       0.839543 |
| UCR         | 0.192389  |   0.363957  |   0.18055   |       0.380596 |
| WSD         | 0.733009  |   0.53519   |   0.0238866 |       0.53519  |
| YAHOO       | 0.696129  |   0.494317  |   0.512841  |       0.671575 |
---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*
