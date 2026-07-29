# Weekly Log: [22nd - 28th July]

## 🎯 Focus of the Week
*Briefly describe the overarching goal for this week.*

---

## 📝 To-Do List
- [x] Further analyse CDC TSB-AD results, focusing on extreme failures.
- [ ] Brainstorm ideas as to how to implement geometrically aware deep learning with CDC/Diffusion Geometry

---

## 🔬 Progress & Experiments

### Geometric Flow Matching Anomaly Detection: Evaluation Metrics

This document outlines the five distinct evaluation metrics used to quantify anomalies in the time-series phase space. The methodology leverages a Time-Conditioned Multilayer Perceptron (MLP) trained via Continuous Normalizing Flows, utilizing the Carré du Champ (CDC) operator to enforce geometric constraints derived from local tangent planes. 

During evaluation, time-series windows are projected forward in time to $t=0.95$ (or $t=0.99$), and anomaly scores are derived from the network's ability to reconstruct the state or from the geometric properties of the resulting error.

---

### 1. Recon_CDC (Geometric Reconstruction Error)

This metric evaluates the model's ability to denoise a sample that has been corrupted not just by isotropic Gaussian noise, but by anisotropic noise structured along the data manifold's tangent plane.

*   **Tangent Projection**: The noise $x_0 \sim \mathcal{N}(0, I)$ is scaled by the square root of the local CDC operator $\Gamma$. Mathematically, this is computed via the eigendecomposition $\Gamma = V \Lambda V^T$.
*   **Forward Process**: The noisy state $x_t$ is constructed at $t=0.95$ using the CDC-guided interpolation: $x_t = t x + (1 - t) x_0 + t \sqrt{\Gamma} x_0$
*   **Prediction**: The network predicts the vector drift $v_\theta(x_t, t)$, and the reconstructed window $x_{1, \text{pred}}$ is computed by reversing the specific noise components.
*   **Scoring**: The final anomaly score is the standard Euclidean distance (L2 norm) between the original window and the CDC reconstruction: $||x - x_{1, \text{pred}}||_2$

### 2. Recon_Standard (Standard Flow Reconstruction)

This metric serves as a non-geometric baseline. It computes the reconstruction error using a standard, unconstrained probability path typical of standard Flow Matching or Denoising Diffusion.

*   **Forward Process**: The test window is corrupted with pure isotropic Gaussian noise $x_0$ without any tangent plane projection: $x_t = t x + (1 - t) x_0$
*   **Prediction**: The network predicts the drift, yielding the standard reconstruction $x_{1, \text{pred}} = v_\theta(x_t, t) + x_0$.
*   **Scoring**: The anomaly score is the L2 norm of the reconstruction error: $||x - x_{1, \text{pred}}||_2$

### 3. Recon_Ensemble (Stochastic Average Reconstruction)

Because the reconstruction process involves sampling random noise $x_0$, a single evaluation can be subject to high stochastic variance. This metric stabilizes the geometric reconstruction error.

*   **Ensemble Averaging**: The `Recon_CDC` process is repeated $N=5$ times for each time-series window using different random noise samples.
*   **Scoring**: The anomaly score is the arithmetic mean of the L2 reconstruction errors across all ensemble passes. This is theoretically the most robust metric for capturing the true expected reconstruction error.

### 4. Orthogonal_Error_Projection (Mahalanobis Tangent Error)

Rather than treating all directions of reconstruction error equally, this metric measures how far the error deviates specifically relative to the local geometric structure of the healthy data. It computes the Mahalanobis distance of the reconstruction error within the tangent space.

*   **Error Projection**: The raw geometric reconstruction error $e = x - x_{1, \text{pred}}$ is projected onto the local eigenvectors of the healthy anchor: $e_{\text{proj}} = V^T e$
*   **Scoring**: The projected error is scaled by the inverse of the local eigenvalues $\Lambda$ (with a small $\epsilon = 10^{-4}$ for numerical stability). 
*   **Formulation**: $\sum (e_{\text{proj}}^2 / (\Lambda + \epsilon))$

### 5. Latent_Noise_Energy (Implied High-t Energy)

Unlike the reconstruction metrics, this approach does not simulate a forward noise process. Instead, it queries the neural network at the extreme end of the temporal domain ($t=0.99$) using the raw, uncorrupted test data.

*   **Evaluation**: The network expects data at $t=0.99$ to be almost entirely pure signal. If an anomalous window is passed, the network's drift prediction $v_\theta(x, t=0.99)$ will misalign with the identity mapping.
*   **Implied Noise**: The latent residual is extracted as the implied noise: $z_{\text{implied}} = x - v_\theta(x, 0.99)$
*   **Scoring**: The anomaly score is the L2 norm of this implied noise vector: $||z_{\text{implied}}||_2$

---

### Time-Series Aggregation & Final Evaluation

Because the Flow Matching model operates on phase-space windows of size $L = 50$, the anomaly scores are initially computed per-window. 

*   **Point-wise Aggregation**: To generate a continuous time-series anomaly score, the window-level scores are accumulated and averaged for each individual time step.
*   **TSB-AD Metrics**: The final point-wise anomaly scores are evaluated against the ground-truth binary labels using Volume Under the Surface (VUS) metrics.
*   **VUS-ROC**: Volume Under the Receiver Operating Characteristic surface.
*   **VUS-PR**: Volume Under the Precision-Recall surface.
---

## 📊 Results & Insights
# Comprehensive Anomaly Detection Benchmark Report

**Total Datasets Evaluated:** 870

This report summarises the performance of three baseline anomaly detection models across a massive benchmark suite.

## 1. Global Averages (VUS-PR)
| Method | Average VUS-PR |
|---|---|
| **CDC_Raw** | 0.4779 |
| **PCA** | 0.5094 |
| **P_norm (d=1, k=15)** | 0.5344 |

## 2. Global Win Rates
Number of datasets where the method achieved the absolute highest score:
- **PCA Wins:** 350 (40.2%)
- **P_norm Wins:** 316 (36.3%)
- **CDC_Raw Wins:** 204 (23.4%)

### Head-to-Head Comparisons
- CDC_Raw strictly outperformed PCA on **426** datasets.
- P_norm strictly improved upon CDC_Raw on **550** datasets.

## 3. Domain Analysis
**Average performance broken down by dataset origin/domain:**

| Domain | Count | CDC_Raw | PCA | P_norm | Best Method |
|---|---|---|---|---|---|
| Exathlon | 32 | 0.7745 | 0.8169 | 0.9041 | **P_norm** |
| Medical | 40 | 0.2786 | 0.1302 | 0.3357 | **P_norm** |
| NAB | 28 | 0.3436 | 0.3747 | 0.3912 | **P_norm** |
| Other | 84 | 0.4686 | 0.4083 | 0.4995 | **P_norm** |
| SMD | 38 | 0.7593 | 0.7649 | 0.8010 | **P_norm** |
| SWaT | 1 | 0.1030 | 0.1124 | 0.1030 | **PCA** |
| Sensor | 29 | 0.5593 | 0.5493 | 0.6024 | **P_norm** |
| Stock | 20 | 0.8167 | 0.8190 | 0.9097 | **P_norm** |
| UCR | 228 | 0.3592 | 0.1924 | 0.3463 | **CDC_Raw** |
| WSD | 111 | 0.5352 | 0.7330 | 0.6757 | **PCA** |
| YAHOO | 259 | 0.4943 | 0.6961 | 0.5774 | **PCA** |

**Average Baseline VUS-PR by Category:**

| Category      |   CDC_Raw |   PCA_VUS_PR |   VUS_PR_P_norm_d1_k15 |
|:--------------|----------:|-------------:|-----------------------:|
| Environment   |    0.5487 |       0.5995 |                 0.4848 |
| Facility      |    0.4946 |       0.4366 |                 0.5295 |
| Finance       |    0.8167 |       0.8190 |                 0.9097 |
| HumanActivity |    0.1549 |       0.0904 |                 0.1621 |
| Medical       |    0.4041 |       0.2034 |                 0.4140 |
| Sensor        |    0.5217 |       0.4517 |                 0.5566 |
| Synthetic     |    0.6356 |       0.6482 |                 0.6403 |
| Traffic       |    0.2880 |       0.3076 |                 0.3757 |
| WebService    |    0.4745 |       0.6981 |                 0.6007 |

## 4. Key Takeaways
- **PCA Dominance:** As an extremely robust global baseline, standard PCA achieved the highest win rate. This implies that many anomalies in this benchmark heavily disrupt the global linear subspace.
- **P_norm Improvement:** The local tangent plane geometry (`CDC_Raw`) struggles in areas of high noise. However, applying the transition probability matrix `P_norm` successfully smoothed out the manifold, yielding a higher average and substantially more wins than raw CDC.
- **Next Steps:** Given the limitations of purely local geometric approaches against global PCA, employing a non-linear global regulariser—such as the newly developed **Carré du Champ Conditional Flow Matcher (CDC-FM)**—is the natural evolution to bridge the gap and achieve state-of-the-art results on datasets where both local and global linear baselines fail.
---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*
