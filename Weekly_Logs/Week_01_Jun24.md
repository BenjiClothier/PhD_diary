# Weekly Log: 24 June 2026

## 🎯 Focus of the Week
Investigating the structural limitations of K-Nearest Neighbours (K-NN) in high-dimensional spaces, establishing rigid anomaly detection baselines, and proving that generative manifold drifting (via the Carré-du-Champ operator) can generalise sparse topologies.

---

## 📝 To-Do List
- [x] **Evaluate CDC Tangential Metric:** Implement the Carré-du-Champ mathematical operator as an evaluation metric to explicitly calculate tangential vs. orthogonal flow vectors on baseline models.
- [x] **Geometric Score Function:** Understand and document the mathematical derivation for the geometric interpretation of the score function.
- [ ] **Investigate K-NN Breakdown:** Analyse the exact mathematical point at which K-NN fails by systematically analysing the ratio of feature dimensionality to the number of available data points.
- [x] **Establish rigid K-NN Baseline:** Finalise a solid baseline for Anomaly Detection (AD) that relies purely on K-NN (e.g., Euclidean distance tracking), ensuring we have a strictly defined performance threshold to beat.
- [ ] **Develop Sub-Manifold Generalisation:** Train a generative neural network that structurally outperforms the K-NN baseline by successfully learning to interpolate and generalise the continuous sub-manifold geometry *between* sparse data points.

---

## 🔬 Progress & Experiments

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
