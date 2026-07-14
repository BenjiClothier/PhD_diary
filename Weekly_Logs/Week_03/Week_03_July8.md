# Weekly Log: [Date Range]

## 🎯 Focus of the Week
Validate and document the discrete Carré du Champ (CDC) orthogonal projection method for anomaly detection, demonstrating its geometric superiority over baselines and showcasing real-world performance.

---

## 📝 To-Do List
- [ ] Toy Example Construction: Create a toy dataset and visual example demonstrating a scenario where the k-NN metric fails for anomaly detection, but the CDC orthogonal displacement succeeds.

- [x] Fix $V_{\text{ortho}}: Currently the model only projects onto the largest singular value dimension, due to normalising $V_{rawtan}$. Fix with SVD on the weighted distance matrix.

- [ ] Theory & Methodology Write-Up: Draft the formal theory and methodology sections explaining the CDC decomposition and the mathematical logic behind the architecture.

- [ ] MVTec Evaluation: Compile, format, and present the anomaly detection results using CDC-drifting on the MVTec dataset.

---

## 🔬 Progress & Experiments

### The Original Architecture & Its Geometric Flaw
The initial implementation attempted a "matrix-free" projection using a weighted sum of neighbor step vectors: $V_{\text{tan\_raw}} = \sum P_{x \rightarrow m} (V \cdot \Delta_m) \Delta_m$. However, this formulation suffered from a geometric flaw:

1. The 1D Covariance CollapseThe matrix-free bypass was mathematically equivalent to multiplying $V$ by the local covariance matrix $G$. This introduced two severe distortions:Eigenvalue Scaling: Instead of treating all tangent directions equally, it scaled the projection by the local variance (the eigenvalues).Dimensionality Collapse: By normalizing the output into a single direction ($V_{\text{tan\_dir}}$), the algorithm collapsed the multi-dimensional tangent space into a single 1D line. This caused the method to "leak" tangential movement in all other dimensions, incorrectly classifying normal on-manifold flow as anomalous.

2. The step vectors were originally calculated relative to the anomalous test point out in ambient space: $\Delta_m = X_m - \text{anomaly}$. Geometrically, this built a tangent plane at the anomaly itself. Because the neighbors $X_m$ sit on the manifold, these step vectors formed a cone pointing from the anomaly back down to the dataset. Consequently, the anomaly's vertical vector $V$ aligned perfectly with this cone, causing the projection to incorrectly absorb the anomaly's magnitude and destroying the anomaly/normal separation ratio.

To resolve these issues, we restructured the algorithm to execute a true, unweighted orthogonal projection onto a dynamically sized, tangent plane.

1. Instead of building the tangent plane at the anomaly in empty space, we shifted the origin of our step vectors to the anomaly's nearest neighbor in the training set ($x_{nn}$).$$\Delta_m = X_m - x_{nn}$$This guarantees the basis vectors span a perfectly flat plane that accurately reflects the true surface of the manifold. When an anomaly vector $V$ crashes into this flat plane, it is correctly recognized as orthogonal.

2. To fix the 1D covariance collapse, we bypassed the $D \times D$ covariance matrix entirely by performing Singular Value Decomposition (SVD) directly on the probability-weighted difference matrix:$$\tilde{X} = \sqrt{P_{x \rightarrow m}} \Delta_m$$ By extracting the right singular vectors of this small $K \times D$ matrix, we obtain the true, unweighted orthonormal basis vectors of the tangent space, removing the distorting effect of the local data density.

3. Because high-dimensional feature spaces (like Wide-ResNet50) may have varying intrinsic dimensionality across different regions of the data structure, hardcoding the tangent space dimension ($m=5$) discards valid tangential flow. We introduced a dynamic variance mask by squaring the singular values ($\sigma_i^2$) to calculate the explained variance. The algorithm now dynamically selects the exact number of basis vectors required to capture 90% of the local geometric variance for each individual point in the batch. This follows the ad-hoc dimension estimation of Ansuini et. al[^1]. 

4. This heuristic is producing good results so far, but it would be better to replace with the Participation Ratio. After an eigen-gap analysis the data showed that the local spectra are smooth power-law curves. Instead of searching for a gap or setting a percentage, we can the entire spectrum to calculate a continuous intrinsic dimension:$$PR = \frac{(\sum_{i=1}^n \lambda_i)^2}{\sum_{i=1}^n \lambda_i^2}$$
- If the variance is concentrated in just 1 dimension, $PR = 1$. 
- If the variance is spread perfectly evenly across all $n$ dimensions, $PR = n$.
- For a smooth curve, it naturally outputs the "effective" number of dimensions without requiring manual thresholds.

```python
# Magic number heuristic
cumulative_variance = torch.cumsum(eigenvalues, dim=1) / total_variance
mask = (cumulative_variance <= 0.90).float()
```

```python
# Calculate continuous intrinsic dimension mathematically
pr = (torch.sum(eigenvalues, dim=1)**2) / torch.sum(eigenvalues**2, dim=1)

# Round to the nearest integer dimension
d_intrinsic = torch.round(pr).long()

# Create a dynamic mask that keeps the first 'd_intrinsic' eigenvectors
N, max_d = eigenvalues.shape
indices = torch.arange(max_d, device=DEVICE).expand(N, max_d)
mask = (indices < d_intrinsic.unsqueeze(1)).float()
```

### Method

**Phase 1: Feature Extraction**
- WideResNet-50: Multiple layers from a pre-trained encoder are dynamically selected depending on category and concatenated. This creates a high-dimensional representation

- $3 \times 3$ Average Pooling: We apply a sliding $3\times3$ Average Pool ($stride=1, padding=1$) across the feature maps.

**Phase 2: Local Geometry (The Tangent Space)**


**Phase 3: The Generative Vector Field**


**Phase 4: Inference & Orthogonal Anomaly Scoring**

---

## 📊 Results & Insights

# MVTec AD Benchmark: PatchCore Baseline vs CDC Optimized

The following table demonstrates the state-of-the-art performance improvements achieved by imposing **Carré du Champ (CDC) constraints** on the generative drift vector field, compared to the standard Euclidean PatchCore baseline.

| Category | PatchCore Baseline | CDC Optimized | Improvement |
| :--- | :--- | :--- | :--- |
| **bottle** | 1.0000 | 1.0000 | 0.0000 |
| **cable** | 0.9968 | 0.9893 | -0.0075 |
| **capsule** | 0.9792 | 0.9924 | **+0.0132** |
| **carpet** | 0.9859 | 0.9872 | +0.0013 |
| **grid** | 0.9791 | 1.0000 | **+0.0209** |
| **hazelnut** | 1.0000 | 1.0000 | 0.0000 |
| **leather** | 1.0000 | 1.0000 | 0.0000 |
| **metal_nut** | 0.9990 | 1.0000 | +0.0010 |
| **pill** | 0.9667 | 0.9861 | **+0.0194** |
| **screw** | 0.9877 | 1.0000 | **+0.0123** |
| **tile** | 0.9949 | 1.0000 | +0.0051 |
| **toothbrush**| 1.0000 | 1.0000 | 0.0000 |
| **transistor**| 0.9987 | 0.9867 | -0.0120 |
| **wood** | 0.9912 | 0.9912 | 0.0000 |
| **zipper** | 0.9950 | 0.9947 | -0.0003 |
| **Mean** | **0.9916** | **0.9952** | **+0.0036** |

## Conclusion
The CDC optimization pushes the overall mean AUROC from **0.9916 to 0.9952**.

It successfully achieves perfect or near-perfect performance on the hardest topological categories where the baseline failed most notably:
*   **Grid:** (0.9791 $\rightarrow$ 1.0000)
*   **Pill:** (0.9667 $\rightarrow$ 0.9861)
*   **Screw:** (0.9877 $\rightarrow$ 1.0000)
*   **Capsule:** (0.9792 $\rightarrow$ 0.9924)

---

## ⏭️ Next Steps


## References

[^1]: Ansuini, A., Laio, A., Macke, J. H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. *Advances in Neural Information Processing Systems*, 32. [arXiv:1905.12784](https://arxiv.org/abs/1905.12784)

