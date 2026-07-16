# Weekly Log: [Date Range]

## 🎯 Focus of the Week
*Briefly describe the overarching goal for this week.*

---

## 📝 To-Do List
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

---

## 🔬 Progress & Experiments

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
2. **Theiler Window Exclusion (CRITICAL):** In time-series, overlapping windows are trivially identical. When constructing the graph and searching for neighbors, we must mathematically enforce a "Theiler Window" (forbid matches within $\pm L$ timesteps). This forces the manifold to map *structural cycles* rather than just sliding along the exact same temporal wave.
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



## 📊 Results & Insights
*What were the results? What broke? What worked?*
- Insight 1: 
- Insight 2: 

*(Tip: You can use markdown to embed images like `![Result Plot](/path/to/plot.png)`)*

---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*
