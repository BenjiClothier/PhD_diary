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

1. The "Landing Zone" AnchorInstead of building the tangent plane at the anomaly in empty space, we shifted the origin of our step vectors to the anomaly's nearest neighbor in the training set ($x_{nn}$).$$\Delta_m = X_m - x_{nn}$$This guarantees the basis vectors span a perfectly flat plane that accurately reflects the true surface of the manifold. When an anomaly vector $V$ crashes into this flat plane, it is correctly recognized as orthogonal.

2. Local SVD for a True Orthonormal BasisTo fix the 1D covariance collapse, we bypassed the $D \times D$ covariance matrix entirely by performing Singular Value Decomposition (SVD) directly on the probability-weighted difference matrix:$$\tilde{X} = \sqrt{P_{x \rightarrow m}} \Delta_m$$ By extracting the right singular vectors of this small $K \times D$ matrix, we obtain the true, unweighted orthonormal basis vectors of the tangent space, removing the distorting effect of the local data density.

3. Dynamic Intrinsic Dimension via Variance MaskingBecause high-dimensional feature spaces (like Wide-ResNet50) have varying intrinsic dimensionality across different regions of the manifold, hardcoding the tangent space dimension ($m=5$) discards valid tangential flow.We introduced a dynamic variance mask by squaring the singular values ($\sigma_i^2$) to calculate the explained variance. The algorithm now dynamically selects the exact number of basis vectors required to capture 90% of the local geometric variance for each individual point in the batch.

---

## 📊 Results & Insights


---

## ⏭️ Next Steps

