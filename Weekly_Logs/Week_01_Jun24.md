# Weekly Log: 24 June 2026

## 🎯 Focus of the Week
Investigating the structural limitations of K-Nearest Neighbours (K-NN) in high-dimensional spaces, establishing rigid anomaly detection baselines, and proving that generative manifold drifting (via the Carré-du-Champ operator) can generalise sparse topologies.

---

## 📝 To-Do List
- [x] **Evaluate CDC Tangential Metric:** Implement the Carré-du-Champ mathematical operator as an evaluation metric to explicitly calculate tangential vs. orthogonal flow vectors on baseline models.
- [x] **Geometric Score Function:** Understand and document the mathematical derivation for the geometric interpretation of the score function.
- [x] **Investigate K-NN Breakdown:** Analyse the exact mathematical point at which K-NN fails by systematically analysing the ratio of feature dimensionality to the number of available data points.
- [x] **Establish rigid K-NN Baseline:** Finalise a solid baseline for Anomaly Detection (AD) that relies purely on K-NN (e.g., Euclidean distance tracking), ensuring we have a strictly defined performance threshold to beat.
- [ ] **Develop Sub-Manifold Generalisation:** Train a generative neural network that structurally outperforms the K-NN baseline by successfully learning to interpolate and generalise the continuous sub-manifold geometry between data points.

---

## 🔬 Progress & Experiments

# Carre du Champ Operator For Manifold Aware Anomaly Detection

## The Carre du Champ Operator: A Summary
The carré du champ (French for "square of the field") is a fundamental mathematical operator that bridges probability theory, stochastic analysis, and differential geometry. The CDC operator, denoted by $\Gamma$, allowa us to extract the local curvature of a scape purely from how things spread within that space.

For a given mathematical operator $L$ and two functions $f$ and $g$, the carré du champ is defined as:
```math
\Gamma(f,g) = \frac{1}{2}(L(fg) - fL(g) - gL(f))
```

When analysing a single function (measuring its own internal variation), we evaluate $f$ against itself:
```math
\Gamma(f,g) = \frac{1}{2}(L(f^2) - 2fL(f))
```

The operator $L$ is known as the infinitesimal generator (often a Markov generator or a Laplacian operator, like $\Delta$). $L$ describes how a process, like heat, a random walk or a probability distribution, evolves over time.

$f$ and $g$ are scalar functions defined on the state space. They are the observables of the space. By feeding these functions into the generator $L$, we observe how these specific properties change as the system evolves.

To understand why the equation is structured the way it is, we have to look at the Leibniz product rule from standard calculus. For a standard, first-order derivative (let's call it $D$), the product rule states:

```math
D(fg) = fD(g) + gD(f)
```
If we rewrite this to equal zero, we get:

```math
D(fg) - fD(g) - gD(f) = 0
```

The generator $L$ (like a diffusion operator) is typically a second-order differential operator. Because it is second-order, it fails to obey the first-order Leibniz product rule perfectly. The carré du champ equation calculates the amount $L$ breaks this rule. The result measures the variance or squared gradient of the system.

If we set the generator $L$ to be the standard Laplacian ($\Delta$) from Euclidean geometry, we can see exactly what the CDC operator extracts.Using the identity $\Delta(f^2) = 2f\Delta(f) + 2|\nabla f|^2$, we plug it into the CDC equation:
```math
\Gamma(f, f) = \frac{1}{2} \big( 2f\Delta(f) + 2|\nabla f|^2 - 2f\Delta(f) \big)
```
```math
\Gamma(f, f) = |\nabla f|^2
```

## Calculating the CDC operator and orthogonal flow anomaly metric
For our work, calculating the CDC operator means evaluating how a specific vector interacts with the local geometry of your data manifold. We then use this to measure the orthogonal movement from the data manifold.

Before we can project our vector, we must define the shape of the manifold beneath it. This is done by building a discrete transition matrix.

### Step 1: Construct the Local Geometry (The Graph Laplacian)
- **Find Neighbours**: For a given point, find its $K$-nearest neighbours in our training data 'bank'.
- **Apply Kernel**: Convert the distances to those neighbours into similarity weights using a Gaussian kernel with a fixed bandwidth parameter ($\epsilon$).
- **Normalise for Density**: Apply Coifman-Lafon density normalisation. This ensures that regions with heavily clumped data do not artificially skew the geometry. This gives a set of robust transition probabilities ($P$) for the local neighbourhood.
  ```math
  W_{i,j} = \exp(-\frac{||x_i - x_j||^2}{\epsilon})
  ```
  - **Estimate the local density**($q$): Count how much weight is around point $i$
  ```math
  q_i = \sum_j W_{i,j}
  ```
  - **Normalise the kernel**($\tilde{W}$): Divide the pairwise similarities by the densities of both points.
  ```math
  \tilde{W}_{i,j} = \frac{W_{i,j}}{q_iq_j}
  ```
  - **Create the Markov transition matrix**($P$): Row-normalise the new density-free weights so they sum to 1.
  ```math
  P_{ij} = \frac{\tilde{W}_{ij}}{\sum_k \tilde{W}_{ik}}
  ```
  The matrix $P$ represents an unbiased random walk on the training manifold.

#### Tying it back to the Carre du Champ($\Gamma$)
How does the matrix $P$ fit into the carre du champ operator?
```math
\Gamma(f,g) = \frac{1}{2}(L(fg) - fL(g) - gL(f))
```
First, we need the infinitesimal generator $L$. In a discrete Markov chain, the generator that describes how functions change over one time step is $L = P - I$ (where $I$ is the identity matrix).

When you apply this discrete $L$ to a function $f$ evaluated at point $x_i$, it looks like this:
```math
L(f)(x_i) = \sum_j P_{ij} \big( f(x_j) - f(x_i) \big)
```
The change in $f$ is the weighted average of how much $f$ differs between you and your neighbours.
For a single function $\Gamma(f,f)$:
```math
\Gamma(f, f)(x_i) = \frac{1}{2} \Big( \sum_j P_{ij} \big( f(x_j)^2 - f(x_i)^2 \big) - 2f(x_i) \sum_j P_{ij} \big( f(x_j) - f(x_i) \big) \Big)
```
Simplify the terms in the brackets:
```math
f(x_j)^2 - 2f(x_i)f(x_j) + f(x_i)^2
```
This is a perfect square. $(a^2 - 2ab + b^2 = (a-b)^2)$. So the entire Carré du Champ equation collapses into:
```math
\Gamma(f, f)(x_i) = \frac{1}{2} \sum_j P_{ij} \big( f(x_j) - f(x_i) \big)^2
```

### Step 2: Define the Local Chords
n the abstract derivation above, $f$ represents a single scalar function. However, in our model, our points exist in a high-dimensional feature space (e.g., $\mathbb{R}^D$). Therefore, we treat $f$ as the coordinate functions of our data.Rather than evaluating one dimension at a time, we evaluate all $D$ coordinate functions simultaneously. The term $\big( f(x_j) - f(x_i) \big)$ becomes the vector difference between a point and its neighbors. We call these the local chords ($\Delta$):
```math
\Delta = x_j - x_i
```
These chords represent the discrete geometric vectors that stitch the local surface of the manifold together.

### Step 3: Compute the Raw CDC Direction
First let's define our drift vector $\nu$:
```math
\nu = f_{\theta}(x) - x
```
This is the vector from our input to our trained network output.

With the local surface defined by the transition matrix ($P$) and the local chords ($\Delta$), we can evaluate how our drift vector ($\nu$) interacts with the manifold.
The CDC operator extracts the gradient information by projecting the incoming vector onto the local chords, weighted by the transition probabilities. We calculate the dot product of $\nu$ with each chord, scale it by $P_{ij}$, and sum the results to find the raw tangent direction:
```math
\nu_{\text{raw\_tan}} = \sum_j P_{ij} \langle \Delta, \nu \rangle \Delta
```
This operation squashes the vector $\nu$ flat against the local geometry established in Step 1.

### Step 4: Enforce Strict Orthogonal Projection
Because discrete approximations of manifolds scale inherently with the intrinsic dimensionality of the data, $\nu_{\text{tan\_raw}}$ gives the mathematically correct direction of the tangent space, but its magnitude is not a strict projection.

We normalise $\nu_{\text{tan\_raw}}$ in to a unit vector ($\nu_{\text{tan\_dir}}$) and explicitly project our original drift vector $\nu$ onto it:
```math
\nu_{\text{tan}} = \langle \nu, \nu_{\text{tan\_dir}} \rangle \nu_{\text{tan\_dir}}
```
This gives us the tangential flow, the movement that safely adheres to the structural rules of the training data.

### Step 5: Extract the Orthogonal Flow (Measuring the Anomaly)
Finally, because we enforced a strict orthogonal projection, we can use simple vector subtraction to isolate the component of the drift that tears away from the manifold. This is our anomaly metric:
```math
\nu_{\text{ortho}} = \nu - \nu_{\text{tan}}
```
A large $\nu_{\text{ortho}}$ magnitude indicates that the generative process is breaking away from the established data structure,

## Implementing a Tangental Repulsion field for Drifting

### Objective
In drifting our repulsive field is used to push generated points ($x_{\text{gen}}$) away from other generated points to prevent mode collapse. 


---
---

# Paper Summary: When Is "Nearest Neighbour" Meaningful?

**Authors:** Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, and Uri Shaft (1999)

## Core Contribution
The paper investigates the "curse of dimensionality" as it applies to the Nearest Neighbour (NN) problem. The authors mathematically prove and empirically demonstrate that, under a broad set of conditions, as dimensionality increases, the distance to the nearest data point converges to the distance to the farthest data point. Consequently, the concept of a "nearest" neighbour loses its meaningfulness because all points become virtually equidistant from the query point.

## The Mathematical Breakdown: "Instability"
* **Definition of Instability:** A nearest neighbour query is defined as "unstable" for a given $\epsilon$ if the distance to most data points is less than $(1+\epsilon)$ times the distance to the nearest neighbour. 
* **Theorem 1 (Distance Concentration):** The authors prove that if the variance of a scaled distance distribution converges to zero as the number of dimensions $m$ approaches infinity, then the ratio of the maximum distance to the minimum distance converges to 1.
* **Generality:** This breakdown occurs under conditions much broader than just strictly independent and identically distributed (IID) dimensions, affecting workloads with correlated attributes and varying dimensional variances.

## Empirical Thresholds
While the theorem describes behavior as dimensions approach infinity, the paper's experiments show that the collapse of distance contrast happens at surprisingly low dimensions.
* In synthetic workloads, the distinction between the nearest and farthest points drops rapidly; at 10 dimensions, the contrast can be reduced by 6 orders of magnitude compared to one dimension.
* By 20 dimensions, the farthest point in a uniform dataset might be only 4 times the distance of the closest point.
* The empirical results suggest that nearest neighbour queries can become unstable with as few as 10 to 15 dimensions.

## When is High-Dimensional NN Actually Meaningful?
The authors clarify that high-dimensional nearest neighbour queries are not always useless. They remain meaningful in specific scenarios:
* **Exact or Near-Exact Matches:** When the query point is an exact duplicate or exceptionally close to a specific data point.
* **Clustered Data:** In classification problems where data naturally groups into well-separated clusters, provided the query point falls within or very close to one of these clusters.
* **Implicitly Low Dimensionality:** When the actual, intrinsic dimensionality of the data is much lower than the ambient high-dimensional space it resides in.

## Why PatchCore Works (Despite High Dimensions)

PatchCore successfully uses Nearest neighbour (NN) for anomaly detection in high-dimensional spaces by exploiting specific geometric exceptions to the "curse of dimensionality." 

### 1. Implicitly Low Dimensionality
While PatchCore extracts features with high ambient dimensions (e.g., 1024-D from a pre-trained Wide ResNet-50), the actual meaningful variations of natural image features trace out a much smoother, lower-dimensional manifold. NN remains mathematically meaningful when the underlying dimensionality of the data is much lower than the actual dimensionality. 

### 2. The Clustered Data Exception
NN queries remain highly effective when data naturally falls into discrete classes or clusters in some potentially high dimensional feature space. PatchCore's normal training features (valid textures and edges) form  dense clusters. Anomaly detection simply measures whether a test patch falls safely inside one of these known clusters or is isolated out in empty space.

### 3. Greedy Coreset Subsampling
If PatchCore retained every training feature, dense regions could suffer from localised distance concentration. By using Greedy K-Center Coreset selection, the algorithm aggressively spaces out the saved features to map only the boundaries and structure of the normal manifold. This preserves geometric contrast and prevents the neighbourhood collapse that occurs when sample sizes overwhelm the space.

### 4. Neighbourhood-Penalised Scoring
Instead of relying solely on the absolute distance to a single nearest neighbour, PatchCore evaluates the local neighbourhood density of that match. By incorporating the distances to the $K$ nearest neighbours within the coreset, it penalises matches that belong to sparse, isolated regions. This prevents random high-dimensional noise from masking true anomalies.

---

## 📊 Results & Insights

## Average Performance Summary Across All MVTec Categories

| Method | Avg Init | Avg Map | Avg Drift | Avg $V_{tan}$ |
|:---|---:|---:|---:|---:|
| **`fullbank` (100% Data)** | **0.9806** | **0.9785** | 0.9515 | N/A |
| **`coreset0.01` (1% Data)** | 0.9676 | 0.9666 | **0.9537** | **0.9629** |
| **`cdc_coreset0.01` (1% + Tangent CDC)** | 0.9650 | 0.9605 | 0.9270 | 0.9510 |

## Top Performers by Category

This table compares the initial ambient AUROC (`Init AUROC`) against the absolute best performance achieved for each of the metrics (`Map`, `Drift`, and `VTan`), along with the specific method that achieved that top score.

| Category   | Init AUROC | Top Map | Top Drift | Top VTan |
|:-----------|-----------:|:--------|:----------|:---------|
| **bottle** | 0.9944 | 0.9976 (`fullbank`) | 0.9960 (`fullbank`) | 0.9889 (`cdc_coreset`) |
| **cable** | 0.9436 | 0.9457 (`fullbank`) | 0.8540 (`fullbank`) | 0.8782 (`coreset`) |
| **capsule** | 0.9892 | 0.9876 (`fullbank`) | 0.9573 (`coreset`) | 0.9741 (`coreset`) |
| **carpet** | 0.9097 | 0.9358 (`coreset`) | 0.9173 (`fullbank`) | 0.9374 (`coreset`) |
| **grid** | 1.0000 | 0.9883 (`cdc_coreset`) | 0.9741 (`cdc_coreset`) | 0.9858 (`coreset`) |
| **hazelnut** | 0.9993 | 0.9993 (`coreset`) | 1.0000 (`coreset`) | 0.9996 (`cdc_coreset`) |
| **leather** | 0.9728 | 0.9915 (`fullbank`) | 0.9901 (`coreset`) | 0.9878 (`coreset`) |
| **metal_nut** | 1.0000 | 0.9976 (`coreset`) | 0.9966 (`coreset`) | 0.9971 (`coreset`) |
| **pill** | 0.9689 | 0.9763 (`fullbank`) | 0.9394 (`coreset`) | 0.9269 (`coreset`) |
| **screw** | 0.9822 | 0.9570 (`coreset`) | 0.9406 (`coreset`) | 0.9328 (`cdc_coreset`) |
| **tile** | 1.0000 | 1.0000 (`cdc_coreset`) | 1.0000 (`coreset`) | 0.9996 (`cdc_coreset`) |
| **toothbrush** | 0.9944 | 0.9944 (`cdc_coreset`) | 0.9944 (`fullbank`) | 0.9611 (`coreset`) |
| **transistor** | 0.9888 | 0.9796 (`fullbank`) | 0.8767 (`cdc_coreset`) | 0.9237 (`cdc_coreset`) |
| **wood** | 0.9939 | 0.9921 (`coreset`) | 0.9930 (`coreset`) | 0.9930 (`cdc_coreset`) |
| **zipper** | 0.9950 | 0.9974 (`fullbank`) | 0.9850 (`fullbank`) | 0.9753 (`coreset`) |

### Quick Takeaways:
- **`coreset` dominates Tangential and Drift scores:** It almost universally holds the top spot for identifying tangential vs orthogonal outliers, proving that sampling 1% of the data retains almost all required structural geometry.
- **`fullbank` reigns in Map Density:** Because Map relies strictly on calculating standard KNN distances without filtering out noise natively, it usually performs slightly better when it has access to the full density graph (`fullbank`).
- **`cdc_coreset` is highly specialised:** It wins out in extremely regular structural patterns (Grid, Tile, Toothbrush).

## Unified Flow Evaluation Summary

This table tracks the highest AUROC values achieved during the Drift, Map, and Tangential evaluation stages against the ambient Init Space. **Bold** values indicate the top performer in that metric for each category.

| Category   | Method          | Init       | Best Map   |   Map Ep | Best Drift   |   Drift Ep | Best VTan   |   VTan Ep |
|:-----------|:----------------|:-----------|:-----------|---------:|:-------------|-----------:|:------------|----------:|
| bottle     | cdc_coreset0.01 | 0.9929     | 0.9944     |      500 | 0.9706       |       2500 | **0.9889**  |       750 |
| bottle     | coreset0.01     | 0.9929     | 0.9952     |      250 | 0.9952       |       1750 | 0.9873      |       250 |
| bottle     | fullbank        | **0.9944** | **0.9976** |     1750 | **0.9960**   |       1000 | NaN         |       nan |
| cable      | cdc_coreset0.01 | 0.8523     | 0.8789     |      100 | 0.8356       |        500 | 0.8671      |      1000 |
| cable      | coreset0.01     | 0.8847     | 0.8855     |     2750 | 0.8366       |       2000 | **0.8782**  |       250 |
| cable      | fullbank        | **0.9436** | **0.9457** |     2250 | **0.8540**   |        500 | NaN         |       nan |
| capsule    | cdc_coreset0.01 | 0.9629     | 0.9657     |     2500 | 0.9402       |        100 | 0.9533      |         0 |
| capsule    | coreset0.01     | 0.9793     | 0.9573     |     1000 | **0.9573**   |       1000 | **0.9741**  |         0 |
| capsule    | fullbank        | **0.9892** | **0.9876** |     2000 | 0.9561       |       2250 | NaN         |       nan |
| carpet     | cdc_coreset0.01 | **0.9097** | 0.9001     |     1750 | 0.9169       |        250 | 0.9306      |       250 |
| carpet     | coreset0.01     | 0.9065     | **0.9358** |     2750 | 0.9133       |       3000 | **0.9374**  |       500 |
| carpet     | fullbank        | 0.8864     | 0.9105     |     1250 | **0.9173**   |       2000 | NaN         |       nan |
| grid       | cdc_coreset0.01 | 0.9916     | **0.9883** |     2000 | **0.9741**   |        250 | 0.9799      |      1250 |
| grid       | coreset0.01     | 0.9791     | 0.9323     |     2500 | 0.8964       |        250 | **0.9858**  |       500 |
| grid       | fullbank        | **1.0000** | 0.9532     |     2750 | 0.8630       |        250 | NaN         |       nan |
| hazelnut   | cdc_coreset0.01 | **0.9993** | 0.9825     |     2750 | 0.9961       |        500 | **0.9996**  |       750 |
| hazelnut   | coreset0.01     | 0.9989     | **0.9993** |     1750 | **1.0000**   |       2500 | 0.9975      |        10 |
| hazelnut   | fullbank        | **0.9993** | **0.9993** |      750 | **1.0000**   |       1500 | NaN         |       nan |
| leather    | cdc_coreset0.01 | 0.9704     | 0.9681     |     1500 | 0.9677       |        500 | 0.9830      |       500 |
| leather    | coreset0.01     | 0.9660     | 0.9901     |     1000 | **0.9901**   |       2000 | **0.9878**  |      2250 |
| leather    | fullbank        | **0.9728** | **0.9915** |     2000 | 0.9891       |       2000 | NaN         |       nan |
| metal_nut  | cdc_coreset0.01 | 0.9995     | 0.9932     |     1000 | 0.9413       |        500 | 0.9775      |         0 |
| metal_nut  | coreset0.01     | **1.0000** | **0.9976** |      750 | **0.9966**   |       1250 | **0.9971**  |       100 |
| metal_nut  | fullbank        | **1.0000** | **0.9976** |     2750 | 0.9951       |        750 | NaN         |       nan |
| pill       | cdc_coreset0.01 | 0.9558     | 0.9414     |     2750 | 0.8819       |        500 | 0.9223      |       750 |
| pill       | coreset0.01     | 0.9577     | 0.9615     |     2750 | **0.9394**   |        500 | **0.9269**  |       250 |
| pill       | fullbank        | **0.9689** | **0.9763** |     2750 | 0.9364       |        500 | NaN         |       nan |
| screw      | cdc_coreset0.01 | 0.9471     | 0.9309     |     2000 | 0.8883       |        500 | **0.9328**  |       500 |
| screw      | coreset0.01     | 0.9613     | **0.9570** |      750 | **0.9406**   |       1750 | 0.9303      |       250 |
| screw      | fullbank        | **0.9822** | **0.9570** |      750 | 0.9297       |        750 | NaN         |       nan |
| tile       | cdc_coreset0.01 | 0.9996     | **1.0000** |     2250 | 0.9996       |        500 | **0.9996**  |         0 |
| tile       | coreset0.01     | 0.9996     | **1.0000** |       50 | **1.0000**   |        250 | **0.9996**  |         0 |
| tile       | fullbank        | **1.0000** | **1.0000** |      750 | 0.9996       |        500 | NaN         |       nan |
| toothbrush | cdc_coreset0.01 | 0.9833     | **0.9944** |      750 | 0.8472       |        500 | 0.8583      |      1500 |
| toothbrush | coreset0.01     | 0.9639     | 0.9861     |      500 | 0.9917       |        750 | **0.9611**  |       250 |
| toothbrush | fullbank        | **0.9944** | **0.9944** |      750 | **0.9944**   |        750 | NaN         |       nan |
| transistor | cdc_coreset0.01 | 0.9271     | 0.9021     |      250 | **0.8767**   |        500 | **0.9237**  |       750 |
| transistor | coreset0.01     | 0.9408     | 0.9133     |     1250 | 0.8733       |        750 | 0.9179      |       100 |
| transistor | fullbank        | **0.9888** | **0.9796** |     2250 | 0.8662       |        500 | NaN         |       nan |
| wood       | cdc_coreset0.01 | 0.9904     | 0.9904     |      500 | 0.9904       |       1000 | **0.9930**  |      2000 |
| wood       | coreset0.01     | 0.9904     | **0.9921** |      250 | **0.9930**   |       1000 | 0.9868      |         0 |
| wood       | fullbank        | **0.9939** | 0.9895     |     1250 | 0.9912       |        750 | NaN         |       nan |
| zipper     | cdc_coreset0.01 | 0.9924     | 0.9772     |     1250 | 0.8784       |       2250 | 0.9551      |      1500 |
| zipper     | coreset0.01     | 0.9926     | 0.9953     |     2000 | 0.9819       |       2000 | **0.9753**  |       100 |
| zipper     | fullbank        | **0.9950** | **0.9974** |     2000 | **0.9850**   |       2000 | NaN         |       nan |



---

## ⏭️ Next Steps
*(To be filled in as the week progresses)*
