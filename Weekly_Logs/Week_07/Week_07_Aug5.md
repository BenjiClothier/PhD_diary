# Weekly Log: [5th Aug - 12th Aug]

## 🎯 Focus of the Week
>MAKE SLIDES

---

## 📝 To-Do List
- [x] How do we submit to the TSB-AD leaderboard?
- [x] Find better automated parameter tweaks for UCR dataset
- [x] Make slides for LPCA vs CDC (similarities, differences)
- [x] Show derivation for conditional CDC
- [x] Implement a Metric Matching CDC network for Anomaly Detection

---

## 🔬 Progress & Experiments

### How to Submit to TSB-AD Leaderboard

- 1. Create a github for model.
- 2. Model's source code should be placed in the TSB_AD/models directory.
- 3. Submit a Pull Request. 
    - Fork the repository ([github.com/TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD)). 
    - Add your implementation and any necessary dependencies or documentation.
    - Open a Pull Request (PR) to the main branch of the official repository.
- 4. Evaluation by Maintainers: Once PR is submitted, the TSB-AD maintainers will independently run, test, and evaluate the algorithm against the full benchmark suite. They evaluate metrics like VUS-PR (Volume Under the Surface for Precision-Recall) to ensure fairness.

```python
    import numpy as np
    from sklearn.base import BaseEstimator



class MyCustomDimensionalityDetector(BaseEstimator):
    """
    Custom anomaly detector using Intrinsic Dimensionality.
    Compatible with the TSB-AD benchmark pipeline.
    """
    
    def __init__(self, window_size=50, max_tau=100, contamination=0.05):
        """
        Initialize the model hyperparameters.
        Note: DO NOT put data manipulation logic here. Only store parameters.
        """
        self.window_size = window_size
        self.max_tau = max_tau
        self.contamination = contamination
        
        # Internal state variables (populated during fit)
        self.is_fitted_ = False
        self.threshold_ = None
        
    def fit(self, X, y=None):
        """
        Fit the model to the training data.
        
        Parameters:
        X : np.ndarray of shape (n_samples, n_features)
            The training time series data. For univariate, n_features=1.
        y : np.ndarray, optional (default=None)
            Ground truth labels. Anomaly detection is usually unsupervised,
            so this is kept for API consistency but often ignored.
            
        Returns:
        self : object
        """
        # 1. Store/learn normal behavior or base dimensionality here
        # E.g., calculate the base intrinsic dimensionality of the training set
        
        self.is_fitted_ = True
        return self
        
    def decision_function(self, X):
        """
        Compute the raw anomaly scores for the input time series.
        
        Parameters:
        X : np.ndarray of shape (n_samples, n_features)
            The test time series data.
            
        Returns:
        scores : np.ndarray of shape (n_samples,)
            The anomaly score for each point. Higher scores = more anomalous.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling decision_function.")
            
        scores = np.zeros(len(X))
        
        # 2. Add your sliding window and KD-Tree logic here.
        # - Slide the window over X
        # - Calculate the local intrinsic dimensionality for each window
        # - The anomaly score is the deviation from the baseline dimension
        
        return scores

    def predict(self, X):
        """
        Generate binary anomaly labels (0 = normal, 1 = anomaly).
        
        Parameters:
        X : np.ndarray of shape (n_samples, n_features)
            
        Returns:
        labels : np.ndarray of shape (n_samples,)
            Binary labels based on the decision_function and contamination threshold.
        """
        scores = self.decision_function(X)
        
        # 3. Apply a threshold to convert raw continuous scores to binary labels.
        # Often derived automatically using the contamination rate (e.g., top 5% are anomalies).
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        labels = (scores >= self.threshold_).astype(int)
        
        return labels

    def predict_proba(self, X):
        """
        (Optional but highly recommended)
        Returns the probability of a sample being anomalous.
        """
        scores = self.decision_function(X)
        # Normalize scores to [0, 1] range using MinMax scaling or logistic function
        min_score, max_score = np.min(scores), np.max(scores)
        if max_score > min_score:
            probas = (scores - min_score) / (max_score - min_score)
        else:
            probas = np.zeros_like(scores)
        return probas
```

### Key Design Rules for Submission

- 1. Unsupervised by Default: Notice the fit(X, y=None) signature. While TSB-AD has ground truth labels, it evaluates models mostly as unsupervised algorithms. Your model should never peek at y during fit() unless explicitly marked as a supervised algorithm.
- 2. Scoring Direction: The decision_function() method must always return scores where higher values indicate higher anomaly severity. If your dimensionality algorithm returns lower values when an anomaly occurs (e.g., the manifold collapses), you must invert the score (e.g., score = -dimension or score = 1/dimension) before returning it.
- 3. Data Shape: The pipeline expects inputs as 2D NumPy arrays (n_samples, n_features). Even for univariate data (TSB-AD-U), expect an array shaped (N, 1).


# Neural Continuous Diffusion Convolution (Neural CDC)

This directory contains the implementation of **Neural CDC**, an advanced deep learning framework designed to learn the continuous Riemannian geometry of time-series data without relying on discrete $k$-NN graphs or explicit eigendecompositions.

## 1. Core Concept
In standard CDC, we construct a discrete nearest-neighbor graph, compute the Graph Laplacian, and perform an eigendecomposition to find the local tangent space of the data manifold. 
**Neural CDC** bypasses this discrete approximation entirely. It learns the topological structure of the data manifold directly through a continuous score-matching objective. 

Instead of an explicit graph, a neural network learns a low-rank Riemannian Metric tensor $\Gamma(x)$ that defines the tangent space at any point $x$ in the continuous space.

## 2. Architecture (`networks.py`)

### `MetricMatchingMLP`
The core neural architecture is a **FiLM-conditioned Residual MLP**.
* **Inputs**: 
  * $x \in \mathbb{R}^D$: A sliding window of the time series.
  * $\epsilon \in \mathbb{R}$: A continuous scalar representing the noise scale (diffusion time).
* **Outputs**:
  * $M \in \mathbb{R}^{r \times D}$: A low-rank projection matrix, where $r$ is the desired intrinsic dimension.
  * The full Riemannian metric is defined implicitly as $\Gamma = M^T M$.
* **Conditioning via FiLM**:
  * The noise scale $\epsilon$ is mapped to a high-dimensional space using **Fourier Features** (similar to positional encoding in transformers).
  * This Fourier embedding is passed into Feature-wise Linear Modulation (**FiLM**) layers.
  * The FiLM layers dynamically shift and scale the activations inside the Residual Blocks based on the current noise level $\epsilon$.

## 3. The Continuous Training Objective (`train_neural_cdc.py`)

Neural CDC is trained using a generalized Score Matching objective (specifically, a Conditional Metric Matching loss). 
The network learns the geometry by observing how the data density changes when smoothed by Gaussian noise.

1. **Noise Injection**: We sample a log-normal noise scale $\epsilon$ (using the Karras 2022 sampling strategy) and inject it into the raw data $X$: 
   $$Y = X + \sqrt{\epsilon} \cdot Z$$
2. **Metric Prediction**: The network predicts the projection matrix $M$ based on the noisy data $Y$.
3. **Loss Function**: We minimize the Conditional Metric Matching Loss:
   $$ \mathcal{L} = \mathbb{E} \left[ ||M_Y M_Y^T||_F^2 - \frac{1}{\epsilon} ||M_Y (X - Y)||^2 \right] $$
   * This is mathematically equivalent to regressing the Hessian of the smoothed data density! 
   * By forcing the network to output a rank-$r$ matrix $M$, we force it to discover the $r$-dimensional tangent plane that best explains the local density.

## 4. Anomaly Scoring (The Orthogonal Error)

Once the network is trained, it has successfully learned the tangent space projection matrix $M$ for any healthy data point. We use this to detect anomalies via Structural Drift.

For an incoming transition from time $t-1$ to $t$:
1. Compute the raw physical movement: $\Delta X = X_t - X_{t-1}$.
2. Query the network for the healthy tangent space at the anchor point $X_{t-1}$ (evaluated at a small noise scale, e.g., $\epsilon=0.01$).
3. Project the physical movement onto the tangent space: $\Delta \hat{X} = M^T (M \Delta X)$.
4. The final **Anomaly Score** is the Orthogonal Error (the magnitude of the movement that escaped the healthy tangent space):
   $$ Score = ||\Delta X - \Delta \hat{X}||^2 $$

## 5. Usage

To train and evaluate Neural CDC on a specific dataset:
```bash
python experiments/neural_cdc/train_neural_cdc.py --dataset path/to/dataset.csv --r 5 --epochs 3000
```
* `--r`: The rank of the projection matrix (the intrinsic dimension).
* `--eps_max`: The maximum noise scale used during Karras sampling.
* `--hidden_dim`: The width of the FiLM-conditioned ResBlocks.


---



## 📊 Results & Insights
# Grid Search Results Analysis

Here is the detailed breakdown of the hyperparameter sweep over different intrinsic dimension estimators and $\tau$ values (with locked 3-layer architecture `L3` to ensure a fair comparison).

## 1. Global Averages
| Estimator   |   Tau |   PR_Mean |   PR_Std |   PR_Max |   ROC_Mean |   ROC_Std |   ROC_Max |   Num_Datasets |
|:------------|------:|----------:|---------:|---------:|-----------:|----------:|----------:|---------------:|
| d_mle       |     1 |    0.5767 |   0.3439 |   1.0000 |     0.8871 |    0.1690 |    1.0000 |            870 |
| d_mle       |    10 |    0.2445 |   0.2534 |   0.9999 |     0.8013 |    0.2042 |    1.0000 |            869 |
| d_mle       |     2 |    0.4943 |   0.3073 |   1.0000 |     0.8856 |    0.1628 |    1.0000 |            870 |
| d_mle       |    30 |    0.1707 |   0.2127 |   0.9305 |     0.7016 |    0.2138 |    0.9995 |            740 |
| d_mle       |     5 |    0.3257 |   0.2780 |   0.9999 |     0.8413 |    0.1852 |    1.0000 |            870 |
| d_mle       |    50 |    0.1278 |   0.2125 |   0.9299 |     0.6447 |    0.2532 |    0.9977 |            588 |
| d_spectral  |     1 |    0.5687 |   0.3459 |   1.0000 |     0.8829 |    0.1720 |    1.0000 |            870 |
| d_spectral  |    10 |    0.2442 |   0.2553 |   0.9999 |     0.7982 |    0.2078 |    1.0000 |            869 |
| d_spectral  |     2 |    0.4811 |   0.3149 |   1.0000 |     0.8775 |    0.1673 |    1.0000 |            870 |
| d_spectral  |    30 |    0.2085 |   0.2594 |   0.9308 |     0.7068 |    0.2508 |    0.9994 |            247 |
| d_spectral  |     5 |    0.3220 |   0.2846 |   0.9999 |     0.8366 |    0.1864 |    1.0000 |            870 |

**Best Configuration by Mean VUS-PR:** `d_mle` at $\tau=1$ (Mean PR: 0.5767)

**Best Configuration by Mean VUS-ROC:** `d_mle` at $\tau=1$ (Mean ROC: 0.8871)

## 2. Dataset Group Breakdown
We can extract the dataset group from the filename (e.g., ECG, Sensor, OPPORTUNITY) to see if optimal $\tau$ is dependent on signal type.

| Group       | Estimator   |   TauValue |   VUS_PR |
|:------------|:------------|-----------:|---------:|
| CATSv2      | d_mle       |          1 |   0.3856 |
| CATSv2      | d_spectral  |          1 |   0.3976 |
| Daphnet     | d_mle       |          2 |   0.4955 |
| Daphnet     | d_spectral  |          1 |   0.5370 |
| Exathlon    | d_mle       |          1 |   0.9926 |
| Exathlon    | d_spectral  |          1 |   0.9919 |
| IOPS        | d_mle       |          1 |   0.5644 |
| IOPS        | d_spectral  |          1 |   0.5769 |
| LTDB        | d_mle       |          2 |   0.6121 |
| LTDB        | d_spectral  |          2 |   0.6111 |
| MGAB        | d_mle       |          1 |   0.6418 |
| MGAB        | d_spectral  |          1 |   0.6180 |
| MITDB       | d_mle       |          5 |   0.3838 |
| MITDB       | d_spectral  |          5 |   0.3841 |
| MSL         | d_mle       |          1 |   0.4877 |
| MSL         | d_spectral  |          1 |   0.4994 |
| NAB         | d_mle       |          5 |   0.5126 |
| NAB         | d_spectral  |          5 |   0.5202 |
| NEK         | d_mle       |          1 |   0.8717 |
| NEK         | d_spectral  |          1 |   0.8765 |
| OPPORTUNITY | d_mle       |          5 |   0.6171 |
| OPPORTUNITY | d_spectral  |          5 |   0.6373 |
| Power       | d_mle       |         10 |   0.5647 |
| Power       | d_spectral  |         10 |   0.6259 |
| SED         | d_mle       |          2 |   0.8301 |
| SED         | d_spectral  |          2 |   0.8299 |
| SMAP        | d_mle       |          2 |   0.7717 |
| SMAP        | d_spectral  |          2 |   0.7507 |
| SMD         | d_mle       |          1 |   0.8745 |
| SMD         | d_spectral  |          1 |   0.8691 |
| SVDB        | d_mle       |          2 |   0.7104 |
| SVDB        | d_spectral  |          2 |   0.7168 |
| SWaT        | d_mle       |          1 |   0.5286 |
| SWaT        | d_spectral  |          1 |   0.5357 |
| Stock       | d_mle       |          1 |   0.9078 |
| Stock       | d_spectral  |          1 |   0.9078 |
| TAO         | d_mle       |          1 |   0.9181 |
| TAO         | d_spectral  |          1 |   0.9186 |
| TODS        | d_mle       |          1 |   0.6983 |
| TODS        | d_spectral  |          1 |   0.7239 |
| UCR         | d_mle       |          2 |   0.4113 |
| UCR         | d_spectral  |          2 |   0.4078 |
| WSD         | d_mle       |          1 |   0.7853 |
| WSD         | d_spectral  |          1 |   0.7860 |
| YAHOO       | d_mle       |          1 |   0.5422 |
| YAHOO       | d_spectral  |          1 |   0.5119 |



## 3. Visualizations

### Global Performance Decay
This line plot demonstrates the strict monotonic decay in average performance as $\tau$ increases globally.
![Global Decay](images/tau_decay_lineplot.png)

### The Optimal Tau Shift
This grouped bar chart highlights the exact reason why STN-1D is necessary. Notice how the highest performance bar strictly shifts to the right depending on the signal family.
![Optimal Tau Shift](images/optimal_tau_shift_barplot.png)


### Global Heatmap
A dense summary of the full hyperparameter landscape.
![Heatmap](images/global_heatmap.png)

## 4. Advanced Metrics: Win-Rate Analysis
Average metrics can be heavily skewed by catastrophic failures on a few datasets. To isolate pure robustness, this table ranks every $\tau$ configuration strictly by its **Win-Rate** (i.e. what percentage of datasets it was ranked #1).

| Estimator   |   TauValue |   Wins |   Total |   Win_Rate_% |
|:------------|-----------:|-------:|--------:|-------------:|
| d_mle       |          1 |    515 |     870 |        59.2  |
| d_mle       |          2 |    172 |     870 |        19.77 |
| d_mle       |          5 |     83 |     870 |         9.54 |
| d_mle       |         10 |     58 |     869 |         6.67 |
| d_mle       |         30 |     20 |     740 |         2.7  |
| d_mle       |         50 |     16 |     588 |         2.72 |
| d_spectral  |          1 |    502 |     870 |        57.7  |
| d_spectral  |          2 |    184 |     870 |        21.15 |
| d_spectral  |          5 |     81 |     870 |         9.31 |
| d_spectral  |         10 |     88 |     869 |        10.13 |
| d_spectral  |         30 |     10 |     341 |         2.93 |

### The Positive Impact of Tau
This bar chart specifically highlights dataset families where locking $\tau=1$ leaves performance on the table, and structurally shifting the manifold ($\tau > 1$) yields a positive effect.
![(Positive Tau Shift)](images/positive_tau_effect_barplot.png)

## 5. The Oracle Performance Ceiling (Validating STN-1D)
If an "Oracle" (or a perfectly tuned STN-1D) were to dynamically select the exact optimal $\tau$ for each individual dataset rather than locking it globally, the overall macro-averaged performance would be:

* **`d_mle` Oracle**: **`0.6578`** Mean VUS-PR (vs. `0.5767` at global $\tau=1$)
* **`d_spectral` Oracle**: **`0.6411`** Mean VUS-PR (vs. `0.5687` at global $\tau=1$)

By dynamically unlocking the correct topological delay per dataset, we gain an enormous absolute improvement of **~8.1% in VUS-PR** across the entire 870-dataset suite. This explicitly proves that dynamically learning $\tau$ is not just a minor optimization, but structurally necessary to achieve true state-of-the-art robustness across diverse domains.

---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*
