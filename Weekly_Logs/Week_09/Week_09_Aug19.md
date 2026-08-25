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

## 📊 Results & Insights
*What were the results? What broke? What worked?*
- Insight 1: 
- Insight 2: 

*(Tip: You can use markdown to embed images like `![Result Plot](/path/to/plot.png)`)*

---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*
