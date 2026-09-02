# Weekly Log: [19 - 26 August]

## 🎯 Focus of the Week
Learning the hyperparameters

---

## 📝 To-Do List
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

---

## 🔬 Progress & Experiments
# Unsupervised Hyperparameter Selection for CDC-FM and Neural CDC

In the context of unsupervised anomaly detection (AD), relying on test-set metrics (like PR-AUC or ROC-AUC) to perform grid searches for optimal hyperparameters (Window Size `L`, Delay `tau`, and Neighbors `k_nn`) is "oracle cheating." In a real-world scenario, only the normal training data is available.

To select these optimal parameters without labels, we must rely on criteria that evaluate the "quality," "predictability," or "topological coherence" of the delay-embedded phase space generated entirely from the training data.

Here are five promising strategies for unsupervised parameter selection:

---

## 1. The "Predictability" Criterion (Forward Prediction Error)

If a delay-embedded phase space is constructed correctly (using optimal `L` and `tau`), the underlying dynamical system unfolds cleanly, making the time series deterministic and predictable. If it is constructed poorly, the state space collapses, trajectories intersect, and the system appears as stochastic noise.

**Implementation:**
- Split the training data chronologically (e.g., 80% train, 20% hold-out).
- Train a lightning-fast, lightweight regressor (e.g., Ridge Regression, a tiny MLP, or a Random Forest) to predict the next point $X_{t+1}$ given the current state window $X_t$.
- **Selection Metric:** Pick the `L` and `tau` configuration that yields the lowest Mean Squared Error (MSE) on the 20% hold-out set.

## 2. Spectral Gap / Intrinsic Dimension Sharpness

Normal time-series data typically resides on a low-dimensional manifold embedded in the higher-dimensional ambient space $L$. If the phase space is constructed optimally, the local tangent spaces (measured via the CDC metric) should exhibit a sharp, defined cut-off between "signal" dimensions and "noise" dimensions.

**Implementation:**
- For a given grid configuration of `(L, tau, k_nn)`, compute the local CDC metric across the training data.
- Extract the average local eigenvalue spectrum (which is already calculated during the intrinsic dimension estimation phase).
- **Selection Metric:** Pick the parameters that maximize the "Spectral Gap" (e.g., the difference between the 1st and 2nd eigenvalues, or the ratio of variance explained by the top `d` dimensions versus the trailing noise dimensions). A sharper gap implies a cleaner, more coherent manifold.

## 3. Topological Coherence (Minimizing False Nearest Neighbors)

False Nearest Neighbors (FNN) is a classic chaos theory concept. If the window size `L` is too small, segments of the time series that are actually far apart in the true underlying system will be projected on top of each other, creating "false" neighbors.

**Implementation:**
- Unroll the training data into phase space. For every point, find its nearest neighbor.
- Artificially increase the embedding dimension `L` by 1. If the distance between those two neighbors suddenly explodes, they were "false" neighbors caused by projection overlap.
- **Selection Metric:** Pick the minimum `L` (and corresponding `tau`) where the percentage of False Nearest Neighbors drops to near zero or plateaus.

## 4. Tangent Space Smoothness (Curvature Minimization)

This is particularly relevant for selecting `k_nn`. If `k_nn` is too small, the manifold graph is jagged, disconnected, and hypersensitive to noise. If `k_nn` is too large, the local tangent planes encompass too much curvature, blurring the fine-grained geometry.

**Implementation:**
- Compute the tangent plane (the subspace spanned by the top `d` eigenvectors of the CDC metric) for each point.
- Measure the principal angles (subspace distance) between the tangent planes of chronologically adjacent points.
- **Selection Metric:** Pick the `k_nn` (and `L`) that minimizes the variance of these angular distances. A well-embedded, well-sampled manifold should feature smoothly varying tangent spaces without erratic jumps.

## 5. Flow Matcher / Diffusion Validation Loss

If the computational budget allows, the most direct proxy to final anomaly detection performance is the network's own ability to learn the manifold structure.

**Implementation:**
- Split the training data into an 80/20 train/validation split.
- Train the `CDCConditionalFlowMatcher` on the 80% split for a set number of steps.
- **Selection Metric:** Evaluate the Flow Matching loss on the 20% validation split. Better geometries are intrinsically easier for the network to model, reliably resulting in a lower validation loss.

---
*Note: While Strategy 5 is the most theoretically sound for a Neural/Flow-Matching framework, Strategies 1 and 2 are vastly more computationally efficient and can evaluate thousands of grid configurations in seconds.*

---

# Selecting L: Summary of Approaches

## Background

The window size `L` (also called the embedding dimension) is the most sensitive hyperparameter in the CDC-FM anomaly detection pipeline. It controls the dimension of the delay-embedded phase space: each time series point becomes an `L`-dimensional vector `[x_t, x_{t-τ}, ..., x_{t-(L-1)τ}]`. Too small and the manifold collapses (trajectories cross); too large and the manifold is sparse, noisy, and computationally expensive.

The oracle grid search ([best_mass_ensemble_grid_results.csv](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/best_mass_ensemble_grid_results.csv)) established the performance ceiling by sweeping `L ∈ {15, 30, 50, 100, 256, 512}` across 871 datasets with `τ=1` and evaluating the Orthogonal Noise ROC-AUC against ground truth labels. This is "oracle cheating" — it uses test labels to pick L — but it provides the upper bound we're trying to match with unsupervised methods.

---

## Approach 1: Signal Processing Heuristics (ACF / AMI)

**Script:** [compute_L.py](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/compute_L.py)

**Idea:** Classical time-delay embedding theory (Takens' theorem) suggests setting `L` based on the decorrelation time of the signal:
- **ACF method:** Find the first lag where the autocorrelation drops below `1/e` or crosses zero.
- **AMI method:** Find the first minimum of the Average Mutual Information.

Both are clamped to `[15, 150]`.

**Result:** Mixed. The [analysis](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/analysis_L_vs_tau.md) comparing dynamic L & τ vs fixed L=50 showed a near 50/50 split (107 wins each across 223 datasets). However, when τ was fixed to 1 and only L was swept via grid search, the grid approach won 135 vs 74 datasets against fixed L=50, confirming that L selection matters but simple ACF/AMI heuristics aren't reliably choosing the right value.

---

## Approach 2: Spectral Feature-Based ML Prediction (Random Forest)

**Scripts:** [train_L_predictor.py](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/train_L_predictor.py), [predict_L.py](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/predict_L.py)

**Idea:** Train a Random Forest classifier to predict the oracle-optimal L class from unsupervised signal features computed on the training data alone:
- **Log Spectrum Slope** — slope of the power spectral density in log-log space (via Welch's method). Captures whether the signal is white-noise-like (slope ≈ 0, needs small L) or has strong low-frequency structure (steep negative slope, benefits from large L).
- **Sample Entropy (SampEn)** — measures regularity/complexity. High entropy signals are noisy and need small L; low entropy signals are structured and tolerate large L.

An earlier version also included `d_fnn_ami` (false nearest neighbours via AMI), but it was dropped because it only contributed ~16% feature importance while being computationally expensive. Log Spectrum Slope (44%) and SampEn (40%) dominated.

**Result:** The trained model is saved as [l_predictor_rf.pkl](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/l_predictor_rf.pkl) and exposed via the `UnsupervisedLPredictor` class in `predict_L.py`. The class can be used at inference time:

```python
from predict_L import UnsupervisedLPredictor
predictor = UnsupervisedLPredictor()
L = predictor.predict(time_series)
```

---

## Approach 3: DiffusionGeometry Spectral Dimension Heuristic (d=2 Criterion)

**Scripts:** [evaluate_d2_heuristic.py](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/evaluate_d2_heuristic.py), [test_dg_heuristic_L.py](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/test_dg_heuristic_L.py)

**Idea:** Use the DiffusionGeometry library to construct the diffusion geometry on the delay-embedded training data, extract the local metric tensor (Γ), and compute the median spectral dimension `d` via the eigengap heuristic. Sweep `L ∈ {16, 32, 64, 128, 256, 512}` and pick the L where the spectral dimension `d = 2` — the hypothesis being that a well-unfolded 1D time series manifold should appear 2-dimensional (one for the signal, one for the dynamics).

The method picks the smallest and/or largest L where d=2 is achieved.

**Result:** Evaluated on 30 sampled datasets ([d2_heuristic_evaluation.csv](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/d2_heuristic_evaluation.csv)). Performance was mixed — many datasets never achieved d=2 at any candidate L, and when they did, the selected L often did not match the oracle. The gap between the heuristic-selected L and the oracle L was significant in many cases.

---

## Approach 4: Intrinsic Dimension Interpolation (d_spectral)

**Script:** [run_low_pass_experiment.py](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/run_low_pass_experiment.py) (`load_intrinsic_dimensions()`)

**Idea:** Rather than selecting L, this approach selects the intrinsic dimension `d` used for the tangent space projection, by interpolating between two estimators based on the signal's spectral character:
- **α = clip(|log_spectrum_slope| / 2, 0, 1)** — confidence that the spectrum has structure
- **d_spectral = (1-α) × d_kaiser + α × d_eigengap** — blends the Kaiser criterion (conservative, noise-aware) with the eigengap heuristic (aggressive, structure-sensitive)

This was used in the mass grid experiments to set `d` per-dataset while sweeping L separately.

---

## Approach 5: Flow Matching Validation Loss

**Script:** [test_unsupervised_val.py](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/test_unsupervised_val.py)

**Idea:** The most direct proxy — split the training data 80/20, train the CDCConditionalFlowMatcher for 100 epochs on the 80% split, and evaluate the flow matching MSE loss on the 20% validation split. The configuration (L, k_nn) that yields the lowest validation loss should correspond to the best geometry for anomaly detection.

**Result:** This was tested on a single dataset (YAHOO #701) with L=15 and k_nn ∈ {15, 50, 80}. It is computationally expensive (requires training a neural network per candidate configuration) but is the most theoretically grounded approach, as it directly measures how well the learned flow field generalises.

---

## Proposed Strategies (Not Yet Implemented)

The [unsupervised_hyperparameter_selection.md](file:///home/clothibenj/Documents/git/myGit/CDC_drift_AD/project_cdc_fm/fm_cdc_L_heuristic/unsupervised_hyperparameter_selection.md) document outlines five strategies, of which approaches 2, 3, and 5 above have been partially explored. Two remain untested:

1. **Forward Prediction Error** — train a lightweight regressor (Ridge/MLP) to predict `x_{t+1}` from the delay-embedded state; pick L that minimises hold-out MSE.
2. **Tangent Space Smoothness** — measure principal angles between tangent planes of chronologically adjacent points; pick L/k_nn that minimises variance of these angles.

---

## Key Findings

| Finding | Evidence |
|---|---|
| L selection significantly impacts performance | Grid L (τ=1) won 135/223 datasets vs fixed L=50 |
| No single L dominates | Oracle L varies across {15, 30, 50, 100, 256, 512} depending on the dataset |
| ACF/AMI heuristics are unreliable | 50/50 split vs fixed L=50 baseline |
| Log Spectrum Slope + SampEn are the most informative unsupervised features | 44% and 40% feature importance respectively |
| The d=2 spectral heuristic has poor coverage | Many datasets never achieve d=2 at any candidate L |
| Validation loss is theoretically sound but expensive | Requires training a full FM model per configuration |


---


# Diffusion Geometry for Time-Series Window Selection

## 1. The Core Problem: Takens' Theorem and the $L$ Trade-off
To detect anomalies in a 1D time series using generative models like Carré du Champ Flow Matching (CDC-FM), we must first reconstruct the underlying system's "phase space." We do this via **time-delay embedding**, transforming the 1D signal into a multi-dimensional trajectory in $\mathbb{R}^L$ using sliding windows of length $L$.

Selecting $L$ is a delicate balancing act:
*   **Under-embedding (Small $L$):** The phase space is "squashed" and insufficiently unfolded. The manifold self-intersects, causing completely different historical states of the time series to artificially overlap and appear similar (projection collapse).
*   **Over-embedding (Large $L$):** The "curse of dimensionality" takes over. The embedded space becomes sparse, spatial distances lose meaning, and high-frequency noise dominates the local geometry.

Standard signal processing heuristics (like Autocorrelation or Average Mutual Information) often fail because they analyse linear statistical dependencies, ignoring the actual *geometry* of the resulting embedded manifold.

## 2. The Flaw in Naive DG: The Missing Theiler Window
Our previous attempts to use Diffusion Geometry (specifically, trying to find an intrinsic dimension of $d=2$ by applying an eigengap heuristic to the Carré du Champ local metric tensor) failed due to a fundamental oversight in $k$-NN graph construction on time-series data. 

Because sliding windows share $L-1$ data points with their immediate chronological neighbours, a standard Euclidean $k$-NN search will overwhelmingly select $t-1, t+1, t-2, t+2$ as the nearest spatial neighbours. The resulting Markov chain degenerates into a trivial 1D chronological line. The graph is entirely blind to the global geometry of the time series (i.e., it fails to connect repeating patterns that occur far apart in time).

**The Fix:** We introduce a **Theiler Window** mask during $k$-NN construction. By explicitly forbidding windows within $\pm L$ time steps from being considered spatial neighbours, we force the $k$-NN graph to connect a point to *genuinely similar repeating patterns* that occurred in the past or future. This reveals the true geometric structure of the data.

---

## 3. Theoretical Framework: The Three DG Metrics

With a topologically valid $k$-NN graph, we extract the local geometry using the discrete **Carré du Champ operator ($\Gamma$)**. For a point $p$ and functions $f, h$, the carré du champ is defined as the local covariance:
$$ \Gamma_p(f, h) = \frac{1}{2} \sum_{j} P_{pj} (f_j - f_p)(h_j - h_p) $$
where $P_{pj}$ are the density-normalised Markov transition probabilities. 

We construct three geometric metrics to evaluate candidate $L$ values:

### Metric 1: Dirichlet Energy of Time (Penalising Self-Intersections)
If $L$ is too small, the embedded manifold is squashed, causing the trajectory to artificially cross itself. In the $k$-NN graph, these "false neighbours" will link points from vastly different times. 
Let $T \in \mathbb{R}^N$ be the global time-index function ($T_i = i / N$). We compute the local carré du champ of the time index:
$$ \mathcal{E}_p(T) = \Gamma_p(T, T) = \sum_{j \in \text{k-NN}(p)} P_{pj} (T_j - T_p)^2 $$
**Theory:** This computes the variance of time within a spatial neighbourhood. If $L$ is too small, false neighbours bridge distant historical epochs, resulting in a massive spike in $\mathcal{E}_p(T)$. Conversely, a properly unfolded manifold will have low Dirichlet energy because the spatial neighbours represent genuinely similar repeating patterns (e.g., the peaks of two different heartbeats), meaning the progression of time along those parallel tracks behaves consistently.

**Implementation 1.0**:


1. **State Space and K-Nearest Neighbors:** The time series is embedded into vectors of length $L$. For every point $X_t$, the algorithm finds its $k$-nearest geometric neighbors. (It specifically uses an adaptive Theiler window to explicitly ignore points that are immediately adjacent in time, forcing the algorithm to look for distant neighbors).
2. **Diffusion Transition Weights (Markov Chain):** For every point $X_t$ and its neighbors $X_{t'}$, it computes a transition probability weight $P(X_t \to X_{t'})$. This is calculated using a standard $\alpha=1$ Diffusion Maps Gaussian kernel with an adaptive bandwidth (based on distance to the 8th neighbor). This weight represents the probability that a random walker on the manifold would step from $X_t$ to $X_{t'}$.
3. **Treating Time as a Function on the Manifold:** It defines a function on the manifold $f(X_t) = t/N$, which simply maps every point to its normalized chronological time step.
4. **Compute Local Dirichlet Energy:** For every point, it computes the expected squared temporal jump a random walker would experience taking a single step in the geometry:$$E_t = \sum_{t' \in \text{KNN}(t)} P(X_t \to X_{t'}) \cdot \left(\frac{t' - t}{N}\right)^2$$ 
   If geometric neighbors $X_{t'}$ are extremely far away in time, the temporal difference $(t' - t)$ is large, and this local energy shoots up.
5. **Aggregate:** Finally, it sums the local energies across all valid points and averages them to produce the `mean_dirichlet` score.

When evaluating $L$ candidates, a high Dirichlet Energy of Time indicates poor quality because the phase space is severely tangled, causing the geometric random walk to jump erratically back and forth through time.

**Critisims**:


### 1. The Recurrence Paradox (Punishing Valid Dynamics)
*   **The Fault:** The metric heavily penalizes points that are close in space but distant in time.
*   **Why it matters:** Consider a perfectly embedded, completely noise-free **sine wave** or a **limit cycle** (like a beating heart). In these periodic systems, the state perfectly revisits identical coordinates in the phase space every period $T$. 
    Because the algorithm uses a **Theiler window** to intentionally blind itself to immediate chronological neighbors, the true nearest neighbors it finds *will inherently be points from previous or future cycles*. Therefore, the temporal jump $(t' - t)$ will be $T, 2T, 3T$, etc. The Dirichlet Energy will square these large jumps and severely penalize the embedding, misinterpreting natural, healthy periodicity as a "folded" or "tangled" manifold.

### 2. Chaotic Systems are Falsely Penalized
*   **The Fault:** It contradicts Poincaré Recurrence.
*   **Why it matters:** In non-linear chaotic systems (like the Lorenz attractor), trajectories are bounded and naturally loop back to pass arbitrarily close to previous states after long, unpredictable periods of time. A high-quality embedding of a chaotic system *should* feature neighbors that are very far apart chronologically. By strictly penalizing long temporal jumps, this metric basically declares that all chaotic attractors are "bad embeddings."

### 3. Bias Toward Infinite Stretching (The "Straight Line" Cheat)
*   **The Fault:** The metric inherently favors embeddings that never loop back on themselves.
*   **Why it matters:** As you increase the window size $L$, you are increasing the dimensionality of the delay embedding. If you make $L$ aggressively large, you essentially stretch the time series out into a massive, non-intersecting, high-dimensional filament. Because the state vector is so long, it never matches anything else in history. 
    The nearest neighbors will just be the points immediately sitting outside the Theiler boundary ($t \pm L_{\text{theiler}}$). The algorithm will think, *"Great, the neighbors are as close in time as legally allowed, no tangling!"* and give it a perfect score. The metric doesn't actually find a good embedding; it just stretches the data until the natural geometry is destroyed.

### 4. Vulnerability to Dataset Length ($N$)
*   **The Fault:** The metric normalizes the temporal distance by dividing by the total length of the dataset ($N$): $\left(\frac{t' - t}{N}\right)^2$
*   **Why it matters:** This makes the metric highly unstable across different sample lengths. If you evaluate a 10-second clip of a repeating signal, and then evaluate a 1-hour clip of the exact same signal, the underlying geometry hasn't changed. But in the 1-hour clip, $N$ is exponentially larger, which artificially squashes the $\left(\frac{t' - t}{N}\right)^2$ fraction closer to zero. The algorithm will give drastically different Dirichlet scores to the exact same dynamical system simply because the recording was left on longer.

### 5. Confusion Between Non-Stationarity and Good Geometry
*   **The Fault:** Random walks or data with a strong underlying drift will score exceptionally well.
*   **Why it matters:** If a time series has a strong upward linear trend (e.g., stock market data), the geometry never loops back on itself. The $k$-nearest neighbors for any point $t$ will *always* be the points chronologically closest to it (just outside the Theiler window). The metric will yield a very low Dirichlet energy, implying an excellent embedding. In reality, the embedding might be practically useless for phase space reconstruction; the metric is just rewarding the lack of recurrence.

### Summary
The Dirichlet Energy of Time is excellent at detecting if a **monotonic, non-recurrent** trajectory has been twisted by a bad embedding. However, for periodic or chaotic data, the metric effectively treats the laws of physics as "geometric errors." To fix this, you would need to measure the *flow* of time (e.g., do the neighbor's trajectories travel in parallel directions?) rather than strictly measuring absolute temporal distance.


>>**Poincaré Recurrence Theorem**:
This theorem states that in any bounded, deterministic system, a trajectory is mathematically guaranteed to eventually return to a state arbitrarily close to its initial starting point. Because our algorithms construct low-dimensional phase spaces (e.g., embedding periodic physiological rhythms or chaotic mechanical vibrations), these recurrences happen frequently and naturally.


**Implementation 2.0**:

#### 1. The Problem: The Recurrence Paradox
Previously, our metric evaluated embedding quality by measuring how well the spatial geometry mapped to chronological time. It computed the expected temporal jump between geometric neighbours:
>> $E_t = \sum P(X_t \to X_{t'}) \cdot \left(\frac{t' - t}{N}\right)^2$

While this effectively detects severely tangled manifolds, it fundamentally misunderstands standard dynamical systems. In systems with limit cycles (e.g. biological signals) or chaotic attractors, trajectories naturally revisit the same regions of phase space over time. Under the old metric, if a geometric neighbour belonged to a previous cycle, the large temporal jump $(t' - t)$ was heavily penalised. The algorithm was essentially treating the laws of physics and Poincaré recurrence as "geometric errors". Furthermore, it artificially rewarded overly large embedding dimensions ($L$) that simply stretched the data into a non-recurrent straight line.

#### 2. The Solution: Measuring Local Flow
To fix this, we drew on Takens' Embedding Theorem. A high-quality delay embedding "unfolds" a manifold such that trajectories do not illegally cross one another. Therefore, if two points are geometrically close (true neighbours), their trajectories must be flowing in parallel directions. If their trajectories cross like an 'X' (false neighbours), their flows will clash.

Instead of measuring absolute time, we now measure the **phase space velocity** or local flow: $\Delta X_t = X_{t+1} - X_t$. 

We no longer care if a neighbour is from 5 seconds ago or 5 years ago; we only ask: *are these two neighbouring points travelling in the same direction?*

#### 3. Integration into the Diffusion Framework
We have integrated this concept into our existing diffusion maps framework with the following steps:

1. **Define the Velocity Field:** For every point $X_t$, we calculate its forward difference $V_t = X_{t+1} - X_t$. We normalise these to unit vectors ($v_t = V_t / \|V_t\|$) to ensure we are strictly evaluating directional alignment, not magnitude.
2. **Preserve Diffusion Mechanics:** We retain our existing Markov transition probability matrix $P(X_t \to X_{t'})$, calculated via the $\alpha=1$ Gaussian kernel and bounded by our adaptive Theiler window.
3. **Compute the Dirichlet Energy of the Flow:** We replace the temporal distance fraction with a measurement of flow disagreement, utilising Cosine Distance: $1 - (v_t \cdot v_{t'})$. 
   
   The new local energy is computed as:
   > $E_{\text{flow}}(t) = \sum_{t' \in \text{KNN}(t)} P(X_t \to X_{t'}) \cdot \left[ 1 - (v_t \cdot v_{t'}) \right]$

4. **Aggregate:** We sum and average this local energy across all valid points to produce our new `mean_flow_dirichlet` score.

#### 4. Key Advantages Realised
* **Respects Chaos and Periodicity:** A recurrent heartbeat signal that loops back to the exact same geometric location will now correctly receive a perfect score, as the old and new cycles are travelling in identical directions (cosine distance $\approx 0$).
* **Scale Invariant:** Because cosine distance is strictly bounded between 0 and 2, the metric is entirely agnostic to the dataset length ($N$). It will not artificially deflate scores just because a recording was left running longer.
* **Prevents $L$-Overfitting:** It naturally penalises arbitrary high-dimensional stretching (the "straight-line cheat"), providing a much more robust heuristic for selecting the optimal window size $L$.

**Next Steps:**
I will be benchmarking this new flow-based Dirichlet metric against our baseline oracle data to quantify the improvement in our $L$-selection accuracy.


### Metric 2: Tangent Space Smoothness (Penalising Kinks via the Levi-Civita Connection)
If a manifold is squashed due to a small $L$, the trajectory will navigate sharp, unnatural folds. We penalise this using principles derived from the **Levi-Civita connection**.

*Understanding the Levi-Civita Connection:* In differential geometry, if you have a curved surface (like a sphere), you cannot simply subtract a tangent vector at point $A$ from a tangent vector at point $B$ because they live in different tangent planes. The Levi-Civita connection ($\nabla_X Y$) solves this by defining "parallel transport"—a mathematically consistent way to slide a vector along a path (the direction $X$) so it can be compared to vectors at a new location. It essentially measures how much the space "twists" or "bends" as you move along a trajectory.

**Theory:** We approximate this concept discretely. By taking the eigendecomposition of the local covariance matrix $\Gamma_p$, we extract the local $d$-dimensional tangent basis $V_t$ at time $t$. To measure the "twist" (the connection), we measure the principal angles between chronologically adjacent tangent planes ($V_t$ and $V_{t+1}$). 
If $L$ is too small, the trajectory is forced through sharp projection artifacts, causing the tangent plane to flip violently (high angular change). Selecting an $L$ that minimises the maximum principal angle ensures we choose a manifold that is smooth and naturally un-kinked.

**Implementation 1.0**:

1. **Extract Local Tangent Planes:** For every point $x_t$ in the embedded phase space, the algorithm finds its $k$-nearest neighbors and computes their local covariance matrix. It then performs an eigendecomposition to find the top $d$ eigenvectors (where $d$ is the intrinsic dimension). These $d$ vectors form an orthonormal basis matrix $V_t \in \mathbb{R}^{L \times d}$ representing the "tangent plane" at time $t$.
2. **Pair Chronological Neighbors:** The algorithm lines up all adjacent tangent planes in chronological order, forming pairs $(V_t, V_{t+1})$.
3. **Compute the Projection Matrix:** For each pair, it computes the inner product matrix $M_t = V_t^T V_{t+1}$. This $d \times d$ matrix represents how much the two tangent planes overlap.
4. **Calculate Principal Angles via SVD:** It takes the Singular Value Decomposition (SVD) of $M_t$ to extract its singular values $S$. Because both planes are orthonormal bases, the principal angles $\theta$ between the two planes are given by exactly $\theta = \arccos(S)$.
5. **Extract the Maximum Angle:** At each time step, there are $d$ principal angles. The algorithm selects the maximum principal angle (`max_angles = angles.max()`), which represents the vector within the tangent plane that shifted the most radically between $t$ and $t+1$.
6. **Average Over Time:** Finally, it averages this maximum angle across all time steps to produce a single `mean_angle` score.

When the algorithm sweeps across different candidate $L$ values, it aims to minimize this `mean_angle` score. A lower score means that as you travel forward in time, the tangent plane glides smoothly rather than twisting violently, indicating a high-quality embedding.

**Critisims:**

### 1. The $k$-NN Paradox and the Curse of Dimensionality
*   **The Fault:** As the algorithm sweeps across larger candidate $L$ (embedding dimension) values, the volume of the phase space grows exponentially.
*   **Why it matters:** To calculate a valid covariance matrix for a tangent plane, you need a fixed minimum number of neighbors $k$. In low dimensions, these $k$ neighbors are truly "local." But in high dimensions, the data becomes sparse. To find $k$ neighbors, the algorithm has to reach very far across the phase space. The resulting tangent plane is no longer a local approximation; it becomes a global chord slicing across the curvature of the manifold, severely distorting the eigenvectors. 

### 2. Extreme Brittleness to Noise (The `max()` Problem)
*   **The Fault:** Extracting the maximum principal angle (`max_angles = angles.max()`) acts as a strict infinity-norm penalty. 
*   **Why it matters:** Eigendecomposition is notoriously sensitive to noise, particularly in the lower-variance eigenvectors. If $d=3$, two of the principal directions might align perfectly with the flow of time. However, if the third vector catches high-frequency measurement noise, it will rotate wildly. Because the algorithm takes the *maximum* angle, it will declare the embedding terrible based entirely on the most noisy, unstable dimension, completely ignoring the smooth alignment of the dominant dynamics.

### 3. The Unpenalized $L$ Sweep (Risk of Trivial Solutions)
*   **The Fault:** The objective is simply to "minimize this mean_angle score" by sweeping $L$, but there is no regularization or penalty for making $L$ larger.
*   **Why it matters:** In phase space reconstruction, increasing $L$ stretches the attractor out. Eventually, if $L$ is large enough, the data is projected into such a high-dimensional space that the trajectory simply looks like a straight line (or perfectly smoothed curve), artificially driving the `mean_angle` to 0. Without a penalty term for higher dimensions, the algorithm will likely "over-embed" and just tell you to pick the highest $L$ you offered it.

### 4. Assumption of High Sampling Frequency
*   **The Fault:** Comparing $V_t$ directly to $V_{t+1}$ assumes a very high temporal resolution (a small $\Delta t$).
*   **Why it matters:** The assumption that a tangent plane "glides smoothly" is only true if the time step between $t$ and $t+1$ is tiny compared to the speed of the system's dynamics. If your data is sampled sparsely, or if the system exhibits fast chaotic mixing, the state at $t+1$ might legitimately be on a radically different part of the attractor's curve. The algorithm would penalize this as a "bad embedding," when in reality, it is just a low sampling rate.

### 5. The Fixed Intrinsic Dimension ($d$)
*   **The Fault:** The algorithm requires $d$ to be known *a priori* and assumes it is constant everywhere on the attractor.
*   **Why it matters:** In many real-world systems, the local intrinsic dimensionality varies. Some parts of the phase space might be 2D (a smooth spiral), while others might be 3D (a chaotic fold). If you force the algorithm to extract $d=3$ everywhere, then in the 2D regions, the 3rd eigenvector is composed entirely of mathematical noise. When paired chronologically, these noise-vectors will be orthogonal to each other, generating massive principal angles and ruining your `mean_angle` score. 

### How to fix some of these faults:
1.  **Replace `max()` with a weighted average:** Instead of just taking the max angle, weight the angles by their corresponding singular values. This ensures that the dominant directions of the tangent plane matter more than the noisy, low-variance directions.
2.  **Add an $L$ penalty:** Introduce a regularization term (e.g., $Score = \text{mean\_angle} + \lambda L$) so it finds the *minimum necessary* embedding dimension, analogous to the Minimum Description Length (MDL) principle.
3.  **Adaptive Neighborhoods:** Instead of a fixed $k$, use a fixed radius $\epsilon$ to find neighbors, ensuring the tangent plane remains strictly local regardless of dimension (though this requires dropping points that don't have enough neighbors).




### Metric 3: Orthogonal Thickness Ratio (Penalising Over-embedding)
As $L$ grows excessively large, the data becomes too sparse to confidently define a clean $d$-dimensional tangent space. 
The eigenvalues $\lambda_1, \dots, \lambda_L$ of $\Gamma_p$ represent the local variance in the embedded space. 
**Theory:** The sum of the top $d$ eigenvalues represents the variance captured by the true, structured tangent space. The sum of the remaining $L-d$ eigenvalues represents the "thickness"—the ambient, orthogonal noise. 
$$ \text{Thickness Ratio} = \frac{\sum_{i=1}^{L-d} \lambda_i}{\sum_{i=1}^L \lambda_i} $$
Minimising this ratio combats the curse of dimensionality. It ensures the manifold remains dense, tightly structured, and lower-dimensional relative to the embedding space, rather than inflating into a cloud of noise.

---

## 4. Practical Implementation

This implementation is highly efficient because it operates entirely on the training data without needing to train a neural network or compute the full Hodge Laplacian.

**Algorithm:**
1.  **Candidate Sweep:** Define a list of candidate $L$ values (e.g., $L \in \{15, 30, 50, 100, 256\}$).
2.  **Delay Embedding:** For each $L$, construct the $N \times L$ phase space matrix.
3.  **Theiler-Masked $k$-NN:** Compute batched pairwise distances. Mask out the diagonal band where $|i - j| \le L$ with infinity. Extract the top-$k$ nearest neighbours.
4.  **Diffusion Kernel:** Compute the anisotropic, density-corrected Markov transition weights $P_{pj}$.
5.  **Compute Metrics:**
    *   Compute $\mathcal{E}_p(T)$ using the transition weights and the chronological time indices.
    *   Construct the local matrix $C_p$ (the discrete $\Gamma$ tensor), compute its eigendecomposition, and calculate the Thickness Ratio.
    *   Extract the top $d$ eigenvectors ($V_t$) and compute the SVD of $V_t^\top V_{t+1}$ to find the principal angles between tangent planes.
6.  **Composite Scoring:** Normalise the three metrics (Dirichlet Energy, Angle Variance, and Thickness) to $[0,1]$ across all candidate $L$'s. Sum them into a single composite score.
7.  **Selection:** The optimal $L$ is the one that minimises the composite score.

---
## Time Delay Embedding
To reconstruct the phase space of the underlying dynamical system, we employ time-delay embedding (Takens, 1981). Rather than following the classical approach of selecting a sparse time-delay ($\tau > 1$) to minimise the embedding dimension ($L$), we adopt a dense-sampling regime typical of modern machine learning. We fix $\tau = 1$ and treat the sequence length $L$ as the embedding dimension. In this regime, the window size $L$ dictates the total embedding window (or receptive field) of the phase space vector. Selecting the optimal $L$ is equivalent to finding the minimal temporal history required to successfully unfold the manifold and prevent trajectory self-intersections, which we measure geometrically using the Carré du Champ operator.

---

### Theoretical Foundations: Hodge Decomposition for $L$ Estimation

To understand why the Hodge Decomposition acts as the perfect "Global-Local Bridge" for estimating the optimal receptive field ($L$), we must translate the abstract differential geometry into its physical counterpart: **Fluid Dynamics**. 

When a time series is delay-embedded into a phase space of dimension $L$, the chronological progression from $X_t$ to $X_{t+1}$ creates a vector field ($V_t$). You can visualise this as a fluid flowing through a high-dimensional space.

Hodge Theory states that *any* flow field on a manifold (specifically, its metric-dual 1-form $V^\flat$) can be perfectly decomposed into three orthogonal, non-overlapping components:
$$V^\flat = d\beta + \delta\alpha + \gamma$$

Here is the intuition behind each component and how they interact to solve the window-size problem.

---

### 1. The Exact Part ($d\beta$): The Local Tangle (Divergence)
*   **Fluid Intuition:** Imagine water bursting out of a spring (a source) or draining into a sinkhole. This component measures pure expansion and contraction. 
*   **Time-Series Meaning:** In a healthy dynamical system, trajectories should flow parallel to one another. If the embedding dimension $L$ is too small, the manifold is "crushed" and folds over itself. Points that are far apart in time are illegally projected into the same spatial location. 
*   **The Effect:** When these false neighbours try to step forward in time, their trajectories clash, diverge wildly, or crash into one another. This generates massive divergence. 
*   **Algorithm Objective:** **Minimise the Exact norm.** By doing so, the algorithm mathematically guarantees that local false intersections have been eliminated and the manifold is locally unfolded.

### 2. The Coexact Part ($\delta\alpha$): The Local Eddy (Curl/Vorticity)
*   **Fluid Intuition:** Imagine small whirlpools or eddies swirling in a river. This measures microscopic, localised rotation.
*   **Time-Series Meaning:** This captures high-frequency, local oscillations or chaotic mixing. While useful for classifying the *type* of dynamic (e.g., chaotic vs. periodic), it is less critical for bounding $L$, as local vorticity can exist in both good and bad embeddings.

### 3. The Harmonic Part ($\gamma$): The Global Cycle (Macroscopic Topology)
*   **Fluid Intuition:** Imagine a river flowing smoothly around a massive, central island. The flow is not bursting from a source (divergence-free), nor is it swirling in tiny whirlpools (curl-free). Instead, it forms a massive, stable, global loop governed by the physical shape of the landmass. 
*   **Time-Series Meaning:** In topology, that "island" is a hole. A periodic, seasonal time series (like daily server loads or heartbeat ECGs) forms a closed loop in the phase space (topologically isomorphic to a circle, $S^1$). 
*   **The Effect:** According to Hodge Theory, the dimension of the harmonic space is exactly equal to the number of topological holes in the manifold (the Betti numbers). If the time series has a stable global period, it possesses a macroscopic topological hole. The harmonic component $\gamma$ isolates the pure, smooth chronological flow circulating around this global period.
*   **Algorithm Objective:** **Maximise the Harmonic norm.** 

---

### The Global-Local Bridge in Action

Neural networks (like Transformers or CNNs) require a window size $L$ large enough to "see" an entire seasonal cycle. Diffusion Geometry inherently looks at local nearest neighbours. 

Hodge Decomposition bridges this gap seamlessly by evaluating the global topology from local transition weights. Here is how the fluid behaves as the algorithm sweeps across candidate $L$ values:

#### Scenario A: Stochastic Noise (e.g., Financial Ticks)
If the data is a random walk, there is no macroscopic topological hole. The fluid is just local Brownian motion. 
*   The Harmonic part ($\gamma$) remains near zero for all $L$.
*   The algorithm focuses solely on minimising the Exact part ($d\beta$) to stop trajectories from clashing.
*   **Result:** It correctly selects a small $L$ (e.g., $L=15$), smoothing the immediate local noise without subjecting the neural network to the curse of dimensionality.

#### Scenario B: Periodic Signals (e.g., Facility Power Cycles)
If the data is highly periodic with a cycle length of $T=400$, and we evaluate $L=20$:
*   **At $L=20$:** The embedding is too short to capture the wave. The "circle" of the phase space is crushed into a tangled mess. The exact part ($d\beta$) is high because trajectories cross. The harmonic part ($\gamma$) is low because the topological hole is crushed and illegible.
*   **At $L=400$:** The embedding window finally engulfs the period. The manifold snaps into a pristine, unfolded topological ring. 
*   **The Mathematical Signature:** The Exact part ($d\beta$) drops to zero (no false crossings). Simultaneously, because a perfect topological hole has manifested in the phase space, the Harmonic part ($\gamma$) experiences a massive spike. 
*   **Result:** The algorithm detects this spike in $\gamma$ and selects $L=400$ (or $512$). It has deduced the global macroscopic period purely by analysing the algebraic topology of the local nearest-neighbour fluid dynamics.

### Conclusion

By computing the Hodge Decomposition of the delay-embedded flow, you are no longer relying on brittle Fast Fourier Transforms or hardcoded intrinsic dimensionality ($d$). You are explicitly measuring:
1.  **Is the geometry locally safe?** (Low Exact Norm).
2.  **Is there a macroscopic cycle the neural network needs to see?** (High Harmonic Norm).

This provides a mathematically rigorous, parameter-free mechanism for estimating the exact receptive field required by deep learning architectures.


---

## Flow Roughness Metric

### Theory
The "Flow Roughness" metric evaluates whether a time-series delay embedding has successfully unfolded the underlying manifold. According to Takens' Embedding Theorem, in a valid phase space, trajectories cannot illegally intersect. Therefore, if two points are geometrically close (true neighbours), their subsequent temporal trajectories must be parallel. 

In differential geometry, the smoothness of a vector field across a manifold is quantified by the Connection Laplacian (or Bochner Laplacian) $\nabla^*\nabla$. If an embedding dimension $L$ is too small, the manifold folds over itself, placing chronologically distant and directionally opposed trajectories adjacent to one another. This causes the local vector field to twist violently, resulting in a high Connection Laplacian eigenvalue (high roughness). 

### The Connection Laplacian (Bochner Laplacian)

**Definition and Purpose**
The Connection Laplacian, denoted as $\nabla^*\nabla$, is a second-order differential operator acting on vector fields (or general tensor fields) over a Riemannian manifold. While the standard Laplace-Beltrami operator measures the spatial dispersion of a *scalar function*, the Connection Laplacian measures the spatial dispersion of a *vector field*. It strictly quantifies the intrinsic "roughness" of the field, defined as the degree to which the vectors deviate from being perfectly parallel across the geometry.

**Mathematical Components**
1. **The Covariant Derivative ($\nabla$):** 
   The operator $\nabla$ represents the Levi-Civita connection. When applied to a vector field $X$, $\nabla X$ computes the directional derivative of $X$ along all tangent directions. It produces a tensor field that encodes exactly how the vector $X$ tilts, stretches, or rotates as one moves infinitesimally along the manifold.
2. **The Formal Adjoint ($\nabla^*$):** 
   The operator $\nabla^*$ is the metric adjoint (dual) to the covariant derivative with respect to the global $L^2$ inner product on the manifold. It acts as a divergence-like operator on tensor fields, mapping the derivative information back down into a vector field.

**The Concept of an Adjoint in $L^2$ Space**
In linear algebra, the adjoint (or transpose) of a matrix $A$ is defined by the algebraic relationship: the dot product of $Ax$ and $y$ equals the dot product of $x$ and $A^*y$. 

In differential geometry and functional analysis, this concept is elevated from finite vectors to continuous vector fields across a manifold. The "dot product" becomes the global $L^2$ inner product, which is the integral of the local geometric inner products over the entire volume of the manifold $M$. 

The formal adjoint $\nabla^*$ is defined strictly as the unique operator that satisfies this exact integral equality:
$$\int_M \langle \nabla X, S \rangle \, d\mu = \int_M \langle X, \nabla^* S \rangle \, d\mu$$
where $X$ is a vector field, $S$ is a tensor field, $\langle \cdot, \cdot \rangle$ is the local Riemannian metric, and $d\mu$ is the volume measure.

**Mapping Up and Mapping Down**
Differential operators move objects between different mathematical spaces based on their tensor rank (complexity).
1.  **Mapping Up ($\nabla$):** The covariant derivative takes a vector field (a rank-1 object). It computes the rate of change of that vector field in every possible tangent direction. The output is a rank-2 tensor field, which contains strictly more structural information than the original vector.
2.  **Mapping Down ($\nabla^*$):** To construct the Connection Laplacian ($\nabla^*\nabla$), the mathematics requires an operator that can return the rank-2 tensor back to the original vector space. The adjoint $\nabla^*$ achieves this by taking the rank-2 tensor $S$ and collapsing it back into a rank-1 vector field. 

**Why it is a "Divergence-like" Operator**
In classical vector calculus, the gradient operator ($\nabla f$) maps a scalar (rank-0) to a vector (rank-1). Through integration by parts, the adjoint of the classical gradient is proven to be the negative divergence ($-\text{div}$). The divergence takes a vector and sums its rates of change to collapse it back into a scalar, representing the net flux (source or sink) at a point.

The covariant derivative $\nabla$ is the higher-dimensional generalisation of the gradient; it maps vectors to tensors. 
Consequently, its adjoint $\nabla^*$ is the higher-dimensional generalisation of the divergence. It performs a tensor contraction (a generalized trace). It sums over the directional components of the rank-2 tensor field to calculate its net geometric accumulation, collapsing it back into a single directional vector at each point.

**The Operator ($\nabla^*\nabla$)**
Applying the connection followed by its adjoint yields the Connection Laplacian: $\nabla^*\nabla X$. 
In flat Euclidean space, this operator simplifies to applying the standard scalar Laplacian to each individual Cartesian component of the vector field. On a curved Riemannian manifold, it actively accounts for the intrinsic curvature of the space, ensuring that the measurement of vector deviation relies strictly on geometric parallel transport.

**Relationship to Dirichlet Energy and Flow Roughness**
The primary utility of the Connection Laplacian in differential geometry and graph signal processing lies in its relationship to the Dirichlet energy of a vector field. By integration by parts on a closed manifold, the global inner product of a vector field $X$ with its Connection Laplacian equates to the total squared norm of its covariant derivative:

$$\int_M \langle \nabla^*\nabla X, X \rangle \, d\mu = \int_M ||\nabla X||^2 \, d\mu$$

The term $||\nabla X||^2$ represents the local variance of the flow. If a vector field consists of perfectly parallel trajectories ($\nabla X = 0$), the Connection Laplacian evaluates to zero. High values indicate areas where the vector field twists violently or where adjacent trajectories point in contradictory directions. 

**The Weitzenböck Identity**
The Connection Laplacian is mathematically distinguished from the Hodge Laplacian ($\Delta_H = d\delta + \delta d$) via the Weitzenböck identity. For a vector field $X$ (identified with a 1-form), the relationship is:

$$\Delta_H X = \nabla^*\nabla X + \text{Ric}(X)$$

where $\text{Ric}(X)$ is the Ricci curvature tensor. This identity proves that the topological properties of a vector field (measured by the Hodge Laplacian) are governed entirely by the sum of its flow roughness (the Connection Laplacian) and the underlying curvature of the physical space.

By measuring the local variance in trajectory directions, the algorithm detects topological false crossings. A low flow roughness mathematically guarantees that the local phase space is structurally sound.

### Maths
**1. Phase Space Velocity and Tangent Projection:**
Initial logic and parameters for phase space velocity ($V_t = X_{t+1} - X_t$) and unit normalisation are validated. The ambient vectors are projected into an $m$-dimensional local tangent space yielding $v_i \in \mathbb{R}^m$.

**2. Discrete Connection Laplacian (Intrinsic Flow Roughness):**
To evaluate intrinsic topological flow, neighbour velocities are aligned to the query point's tangent space using the orthogonal parallel transport matrix $O_{ij} \in \mathbb{R}^{m \times m}$. The local roughness at point $i$ is calculated as the diffusion-weighted expected variance in the intrinsically aligned flow direction:
$$E_i = \frac{1}{2} \sum_{j \in \text{KNN}(i)} P_{ij} ||v_i - O_{ij}v_j||^2$$

The global intrinsic flow roughness is the average of these local energies across all $N$ valid points:
$$\text{Roughness} = \frac{1}{N} \sum_{i=1}^N E_i$$

### Implementation
**1. Ambient Velocity and Tangent Extraction**
Standard processing applied to compute normalised ambient velocities $V$. Local tangent bases $B \in \mathbb{R}^{N \times L \times m}$ are extracted via the dominant eigenvectors of the spatial Gram matrix.

**2. Parallel Transport Matrix ($O_{ij}$)**
```python
alignment_matrix = torch.matmul(B_i.transpose(2, 3), B_j)
U, _, Vh = torch.linalg.svd(alignment_matrix)
O_ij = torch.matmul(U, Vh)
```

**3. Intrinsic Geometric Disagreement and Energy Aggregation**
```python
v_i_tangent = torch.einsum('nli,nl->ni', B, V)
v_j_tangent = torch.einsum('nkli,nkl->nki', B_j, V[knn_idx])
v_j_transported = torch.einsum('nkij,nkj->nki', O_ij, v_j_tangent)

v_diff_sq = torch.sum((v_i_tangent.unsqueeze(1) - v_j_transported)**2, dim=2)
intrinsic_flow_roughness = torch.mean(0.5 * torch.sum(P * v_diff_sq, dim=1)).item()
```
*   `O_ij` rotates the tangent-projected neighbour velocities into the coordinate frame of the query point.
*   `torch.sum(..., dim=1)` computes the discrete Connection Laplacian trace $E_i$ using Markov weights $P$.
*   `torch.mean(...)` yields the macroscopic intrinsic flow roughness scalar.

## Slide 1

### The Core Intuition
Imagine your time series is a long piece of string, and every data point is a bead numbered sequentially (1, 2, 3, 4...) representing its chronological timestamp. 

When you embed this string into a low-dimensional phase space (a small window size $L$), you are essentially crumpling the string into a tangled ball. 

The **Dirichlet Energy of Time** simply asks: *"If I stand on a bead and take a tiny step to my closest physical neighbour in this tangled ball, how far did I just time-travel?"*

### Breaking Down the Math
**1. The Temporal Jump ($t_j - t_i$):**
This looks at the difference in the chronological timestamps between a point ($i$) and its geometric neighbour ($j$). 

**2. Squaring the Jump $( \dots )^2$:**
If your nearest geometric neighbour is bead 10 and you are bead 11, the jump is small ($1^2 = 1$). But if the string is looped over itself, your nearest physical neighbour might be from a previous seasonal cycle—say, bead 400. Squaring this difference ($390^2$) acts as a massive, ruthless penalty against "time-travelling" shortcuts.

**3. The Diffusion Weight ($P_{ij}$):**
This ensures we only care about points that are genuinely close in the geometry. If a point is physically far away, the transition probability $P_{ij}$ is nearly zero, so we don't care what its timestamp is.

### Why This is Crucial for Finding $L$ (The Macro-Stretcher)
This metric serves a very specific purpose: **It forces the Neural Network's receptive field to be large enough to see entire macroscopic cycles.**

If your data has a daily seasonality of 400 steps, but you use a small window size of $L=20$, the phase space will loop over itself 20 times. Geometric neighbours will constantly jump across cycles, and the Dirichlet Energy will explode. 

The only mathematical way for the algorithm to minimise this energy is to **increase the window size $L$ until the window is longer than the wave itself.** Once $L$ is large enough, the tangled ball stretches out into a long, non-intersecting ribbon. At that point, the only physical neighbours available are your immediate chronological neighbours, and the energy drops to zero.

## Slide 2

### The Core Intuition
Imagine drawing a series of parallel arrows on a flat sheet of rubber. As long as the sheet remains flat, all the arrows point in the exact same direction. If you violently crumple the sheet of rubber into a ball, arrows that are now physically touching each other will point in wildly contradictory directions.

In time-series embedding, the "arrows" are the chronological trajectories (velocities) of the data points. If the window size $L$ is too small, the phase space is crumpled. **Flow Roughness** measures the severity of this crumpling. It projects the trajectory vectors onto the local surface and calculates the variance in their directions. A high roughness score guarantees that the manifold is folded over itself, indicating that the neural network receptive field is insufficient.

### How $O_{ij}$ (Parallel Transport) is Calculated
To accurately compare an arrow at point $i$ with an arrow at point $j$ on a curved surface, one cannot simply subtract their ambient 3D coordinates. The arrow at $j$ must be "carried" along the surface to point $i$ without adding any artificial twisting. This mathematical translation is performed by the orthogonal parallel transport matrix, $O_{ij}$.

It is computed in four strict linear algebra steps:

**1. Extract Local Tangent Bases ($B_i$ and $B_j$)**
The algorithm computes the local covariance matrix of the spatial neighbourhood. The dominant eigenvectors of this matrix form an orthogonal basis ($B \in \mathbb{R}^{L \times m}$) representing the flat tangent plane resting on the manifold at that specific point.

**2. Compute the Alignment Matrix**
To see how the tangent plane at $j$ is tilted relative to the tangent plane at $i$, the algorithm computes their inner product:
$$A = B_i^T B_j$$
This yields an $m \times m$ matrix capturing the raw geometric overlap between the two planes.

**3. Singular Value Decomposition (SVD)**
The alignment matrix is decomposed into its fundamental rotational and stretching components:
$$A = U \Sigma V^T$$

**4. Isolate the Pure Rotation ($O_{ij}$)**
Because $O_{ij}$ must represent a pure, rigid transport without stretching or distorting the vector, the singular values ($\Sigma$) are discarded. The parallel transport matrix is constructed strictly from the left and right singular vectors:
$$O_{ij} = U V^T$$

When the neighbour's velocity vector ($v_j$) is multiplied by $O_{ij}$, it is seamlessly rotated into the coordinate frame of $v_i$, allowing the algorithm to compute the true, intrinsic directional disagreement: $||v_i - O_{ij}v_j||^2$.

## Slide 3

### The Core Intuition
Imagine trying to forecast a system based on historical records. If today's exact conditions perfectly match five specific days from the past, you can average what happened on the days immediately following those historical events to reliably predict what will happen tomorrow. 

In a properly unfolded, deterministic dynamical system, history repeats itself. In the time-series phase space, a query point's geometric neighbours represent those exact historical matches. **Local Predictability Error** tests whether averaging the immediate futures of those historical neighbours successfully predicts the immediate future of the current state. 

### Breaking Down the Math
**1. The Diffusion-Weighted Prediction ($\hat{v}_i$)**
The algorithm retrieves the velocity vectors ($v_j$) of the $k$-nearest spatial neighbours—representing where the time series travelled immediately after those historical moments. It calculates a weighted average of these vectors using the Markov transition probabilities ($P_{ij}$). Historical moments that are geometrically closer to the current state carry more weight in the prediction. The result is normalised into a single predicted unit vector.

**2. The Prediction Error ($1 - \langle v_i, \hat{v}_i \rangle$)**
This computes the cosine distance between the query point's actual velocity ($v_i$) and the predicted velocity ($\hat{v}_i$). If the historical neighbours perfectly forecast the trajectory, the dot product evaluates to $1$, and the error drops to $0$. If the neighbours flow in completely unrelated directions, the vectors cancel out, the dot product approaches $0$, and the error rises to $1$.

### Why This is Crucial for Finding $L$ (The Starvation Check)
This metric is a direct mathematical implementation of Sugihara's S-Map constraint. It strictly penalises two distinct geometric failures:

1.  **Too Small (The Tangle):** If $L$ is insufficient, the manifold folds. Points that represent completely different underlying states are crushed together as false neighbours. Because they belong to different parts of the system, their future trajectories violently diverge. The prediction fails.
2.  **Too Large (Dimensional Starvation):** As $L$ becomes excessively large, the curse of dimensionality empties the phase space. To fulfill the $k$-NN requirement, the algorithm is forced to connect the query point to completely unrelated, distant historical states. Their random future trajectories cancel each other out during the weighted average. The prediction fails.

Minimising the Local Predictability Error isolates the exact optimal window size: the point where the manifold is perfectly unfolded, but strictly before the high-dimensional space becomes too sparse to extract meaningful patterns.

## Slide 4

### The Core Intuition
Imagine trying to forecast a system based on historical records. If today's exact conditions perfectly match five specific days from the past, you can average what happened on the days immediately following those historical events to reliably predict what will happen tomorrow. 

In a properly unfolded, deterministic dynamical system, history repeats itself. In the time-series phase space, a query point's geometric neighbours represent those exact historical matches. **Local Predictability Error** tests whether averaging the immediate futures of those historical neighbours successfully predicts the immediate future of the current state. 

### Breaking Down the Math
**1. The Diffusion-Weighted Prediction ($\hat{v}_i$)**
The algorithm retrieves the velocity vectors ($v_j$) of the $k$-nearest spatial neighbours—representing where the time series travelled immediately after those historical moments. It calculates a weighted average of these vectors using the Markov transition probabilities ($P_{ij}$). Historical moments that are geometrically closer to the current state carry more weight in the prediction. The result is normalised into a single predicted unit vector.

**2. The Prediction Error ($1 - \langle v_i, \hat{v}_i \rangle$)**
This computes the cosine distance between the query point's actual velocity ($v_i$) and the predicted velocity ($\hat{v}_i$). If the historical neighbours perfectly forecast the trajectory, the dot product evaluates to $1$, and the error drops to $0$. If the neighbours flow in completely unrelated directions, the vectors cancel out, the dot product approaches $0$, and the error rises to $1$.

### Why This is Crucial for Finding $L$ (The Starvation Check)
This metric is a direct mathematical implementation of Sugihara's S-Map constraint. It strictly penalises two distinct geometric failures:

1.  **Too Small (The Tangle):** If $L$ is insufficient, the manifold folds. Points that represent completely different underlying states are crushed together as false neighbours. Because they belong to different parts of the system, their future trajectories violently diverge. The prediction fails.
2.  **Too Large (Dimensional Starvation):** As $L$ becomes excessively large, the curse of dimensionality empties the phase space. To fulfill the $k$-NN requirement, the algorithm is forced to connect the query point to completely unrelated, distant historical states. Their random future trajectories cancel each other out during the weighted average. The prediction fails.

Minimising the Local Predictability Error isolates the exact optimal window size: the point where the manifold is perfectly unfolded, but strictly before the high-dimensional space becomes too sparse to extract meaningful patterns.


## Slide 5

### The Core Intuition
If the **Edge Flow ($W_{ij}$)** represents the volume of fluid moving through a single pipe between two points, the **Codifferential ($\delta W$)** measures the **net accumulation or depletion of fluid at a specific junction (node).**

In standard vector calculus, this concept is known as **Divergence**. 

### The Mathematical Mechanics
In differential geometry, the exterior derivative ($d$) maps a 0-form (a scalar function on nodes) to a 1-form (values on edges). The **codifferential** ($\delta$) is its exact mathematical adjoint—it runs the machinery in reverse. It takes a 1-form (edge flows) and maps it back down to a 0-form (a single scalar value at each node).

The equation from the slide achieves this on a discrete graph:
$$(\delta W)_i = \sum_{j \in \mathcal{N}(i)} P_{ij} W_{ij}$$

1.  **$W_{ij}$**: The flow along the edge connecting point $i$ to neighbour $j$. 
2.  **$P_{ij}$**: The Markov transition weight, which dampens the impact of geometric outliers.
3.  **The Summation ($\sum$)**: It aggregates the weighted flows across all pipes connected to node $i$.

### The Geometric Result (Sources, Sinks, and Transit)
When you sum the flows around a specific node $X_i$, three things can happen:

1.  **Positive Net Flux (A Source):** Trajectories are spontaneously bursting outward from this spatial location in multiple conflicting directions.
2.  **Negative Net Flux (A Sink):** Multiple disconnected trajectories are violently crashing into this single spatial location. 
3.  **Zero Net Flux (Transit):** The flow entering the node from the "past" perfectly balances the flow exiting the node towards the "future". 

### Application to Receptive Field ($L$) Estimation
In a valid, fully unfolded time-series embedding, the chronological trajectory acts like a single continuous river. Every point should just be a transit node (Zero Net Flux). 

If the window size $L$ is too small, the manifold is crushed, forcing chronologically distant trajectories to overlap in Euclidean space. These false intersections act as massive artificial sources and sinks. By computing the codifferential, the algorithm identifies these topological collisions. Summing and squaring these values yields the **Exact Energy**, which the algorithm minimises to guarantee that the manifold has unfolded enough to eliminate false crossings.

## Slide 6

### The Core Intuition
Imagine a complex fluid flow in a river network. You can decompose this flow into two distinct behaviours: water draining into sinkholes or bursting from springs (divergence), and water circulating smoothly around a large central island (harmonic flow). 

In a time-series phase space, a highly periodic, seasonal pattern forms a closed topological loop—the equivalent of that central island. **Harmonic Energy** isolates and measures the pure, uninterrupted circulation around this global loop. It actively filters out local topological collisions (sources and sinks) to reveal the true macroscopic periodicity of the underlying data.

### Breaking Down the Math
**1. The Poisson Equation ($(I - P)f = \delta W$)**
To isolate the cyclic flow, the algorithm must first identify the "downhill" flow caused by topological intersections. It solves the discrete Poisson equation using the graph Laplacian ($I - P$) and the codifferential ($\delta W$). This computes a scalar potential function ($f$), mapping the exact geometric pressure of every source and sink in the network.

**2. The Residual Flow ($R_{ij} = W_{ij} - (f_j - f_i)$)**
The algorithm calculates the gradient of the potential function across each edge ($f_j - f_i$). This represents the pure "Exact" flow—the trajectories crashing into one another. By subtracting this from the raw edge flow ($W_{ij}$), the algorithm strips away all local topological errors. The remainder ($R_{ij}$) is completely divergence-free. 

**3. The Harmonic Energy Norm**
The algorithm squares and averages this divergence-free residual flow across the graph. Because a 1D time series forms a 1D trajectory ribbon, it lacks the geometry to support true 2D localised whirlpools (coexact flow). Therefore, this residual energy is almost exclusively comprised of pure Harmonic flow ($\gamma$) circling the global topological hole.

### Why This is Crucial for Finding $L$
This metric acts as a rigorous mathematical detector for deep learning context windows. 

If the window size $L$ is smaller than the dataset's seasonal cycle, the topological loop is crushed into a tangled mass. False intersections dominate, the Exact flow consumes all the energy, and the Harmonic Energy remains near zero.

When $L$ expands to perfectly encapsulate the entire macroscopic period, the false intersections vanish. The phase space snaps open into a pristine, non-intersecting topological ring. The sources and sinks disappear, and the pure cyclic circulation dominates. By seeking the $L$ that **maximises the Harmonic Energy**, the algorithm blindly deduces the exact global receptive field required by the neural network to observe the dataset's full seasonality.


## Slide 7

### The Core Intuition
Imagine striking a bell or a drum. Its physical shape dictates the exact acoustic frequencies at which it naturally resonates. In differential geometry, the shape of a data manifold dictates its resonant frequencies—represented mathematically by the eigenvalues of the graph Laplacian. 

A perfectly closed, uniform circle (a pure periodic time series) possesses a highly unique resonant signature: its frequencies always appear in perfectly identical pairs. The **Spectral Gap** measures the difference between these frequencies. If the gap drops to exactly zero, the time-series phase space has physically resolved itself into a flawless, unbroken topological loop.

### Breaking Down the Math
**1. The Random Walk Laplacian ($\Delta = I - P$)**
This operator models how information or "heat" diffuses across the $k$-NN graph. It is computed by subtracting the Markov transition matrix ($P$) from the Identity matrix ($I$). 

**2. The Eigenvalue Spectrum ($0 = \lambda_0 \leq \lambda_1 \leq \lambda_2 \dots$)**
The Laplacian yields a sequence of eigenvalues representing the fundamental frequencies of the graph.
*   $\lambda_0$ is always exactly $0$.
*   $\lambda_1$ (Algebraic Connectivity) dictates how easily the graph can be severed. If $\lambda_1 \approx 0$, the manifold is shattering into disconnected fragments (dimensional collapse). A healthy, highly connected graph maintains $\lambda_1 \gg 0$.

**3. The Spectral Gap ($\lambda_2 - \lambda_1$)**
This calculates the numerical distance between the first and second active frequencies. According to spectral graph theory, a pure cycle graph uniquely guarantees that these two eigenvalues share a multiplicity of 2 (they are mathematically identical).

### Why This is Crucial for Finding $L$
This metric acts as a flawless, noise-free detector for strict macroscopic periodicity.

If the embedding window $L$ is too small to contain the full seasonal cycle, the manifold folds over itself. The spatial nearest neighbours form false structural "short-circuits" across the phase space. These shortcuts destroy the perfect symmetry of the circle, forcing the resonant frequencies to split and resulting in a large Spectral Gap.

When $L$ expands to perfectly encapsulate the entire macroscopic period, the false bridges vanish. The space snaps into a pristine, non-intersecting topological ring. The frequencies instantly align, and the Spectral Gap plummets to exactly $0.00000$. By scanning for the first $L$ where the gap hits zero (whilst ensuring $\lambda_1 > 0$ to guarantee the graph hasn't just shattered), the algorithm mathematically proves the resolution of a periodic cycle and locks in the exact receptive field the neural network requires.

## Slide 8

### The Core Intuition
Imagine holding a compass pointing north, walking along a triangle drawn on a perfectly flat piece of paper, and returning to your starting point. The compass will still point in exactly the same direction. Now, perform the same walk on the surface of a globe—starting at the equator, walking to the North Pole, down to the equator, and back to the start. Because of the sphere's curvature, your compass will have rotated. 

This net rotation is called **Holonomy**. In time-series embeddings, it measures the intrinsic curvature of the phase space. If the manifold is smooth, returning to the start of a local geometric loop yields no rotation. If the window size $L$ forces the manifold to twist violently, the local curvature spikes, and the holonomy explodes.

### Breaking Down the Math
**1. Parallel Transport across 3-Cycles ($H_{ijk} = O_{ij} O_{jk} O_{ki}$)**
The algorithm isolates closed triangles within the $k$-NN graph (points $i$, $j$, and $k$ that are all mutual geometric neighbours). It takes a tangent vector and sequentially applies the orthogonal parallel transport matrices ($O$) to carry the vector along the edges of the triangle until it returns to its origin. The product of these matrices is the local holonomy operator, $H_{ijk}$.

**2. The Holonomy Error ($||H_{ijk} - I_m||^2$)**
If the local geometry is flat and smooth, the vector returns completely unchanged, meaning $H_{ijk}$ perfectly equals the identity matrix ($I_m$). This metric calculates the squared Frobenius norm of the difference between the holonomy operator and the identity matrix. It measures exactly how violently the vector was rotated by the local curvature, averaging this rotational error across every triangle in the graph.

### Why This is Crucial for Finding $L$
This metric acts as a strict boundary detector for high-frequency or highly stochastic data, identifying the **Holonomy Wall**.

At small window sizes, a high-frequency time series embeds as a relatively smooth, flat 1D string in the phase space. The triangles formed by local neighbours have zero area and zero curvature, yielding a low holonomy error. 

As $L$ expands, the sliding window begins to swallow multiple erratic, high-frequency oscillations simultaneously. To accommodate this, the 1D string must twist and crumple into a dense knot. The intrinsic curvature spikes to mathematical extremes, causing the holonomy error to violently explode. 

Neural networks process data using flat Euclidean operations. If the intrinsic manifold twists too violently, the neural network cannot map the Euclidean input to the underlying dynamics. By selecting the $L$ exactly preceding this holonomy explosion, the algorithm mathematically caps the receptive field, providing the neural network with the absolute maximum temporal context achievable before the geometric structure irreparably breaks down.

## 📊 Results & Insights
*What were the results? What broke? What worked?*
- Insight 1: 
- Insight 2: 

*(Tip: You can use markdown to embed images like `![Result Plot](/path/to/plot.png)`)*

---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*
