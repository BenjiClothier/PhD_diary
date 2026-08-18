# Weekly Log: [12th - 19th Aug]

## 🎯 Focus of the Week
Simplify the hyperparameters of CDC-FM AD from dimension and tau to L (the receptive field). Begin taking notes on tools provided by Diffusion Geometry and their applications to Anomaly Detection.

---

## 📝 To-Do List
- [x] Receptive Field Ablation Study
- [x] Diffusion Geometry's applications for Anomaly Detection
- [ ] Apply FFT to calculate $\tau$

---

## 🔬 Progress & Experiments

### FFT $\rightarrow \tau$  

**Spectral Analysis:** Execute an FFT to calculate the Power Spectral Density (PSD) of the time series.

**Energy Threshold:** Identify the cut-off frequency $f_{c}$ below which 90% of the cumulative spectral energy is concentrated.

**Timescale Extraction:** Calculate the characteristic period of the dominant low-frequency dynamics using $T = 1 / f_{c}$.

**Delay Calculation:** Apply the quarter-period heuristic to establish the optimal lag, defined as $\tau \approx T / 4$.

---

## 📊 Results & Insights
# Analysis: Changing Window Size (L) vs Fixed L=50

Total datasets compared (present in all sets): **223**

## Overview (Dynamic L & Tau vs Fixed L=50)
- **Dynamic L & Tau Wins:** 107 datasets
- **Fixed L=50 Wins:** 107 datasets
- **Ties:** 9 datasets

## Top 15 Datasets where Dynamic L & Tau heavily beat Fixed L=50
These are cases where sweeping L uncovered significantly better geometry than keeping it fixed to 50.

| Dataset | Fixed L=50 PR | Dynamic L PR | Difference |
|---------|---------------|--------------|------------|
| 175_MITDB_id_6_Medical_tr_27274_1st_2737... | 0.2570 | 0.6968 | **+0.4398** |
| 072_WSD_id_44_WebService_tr_1691_1st_179... | 0.3804 | 0.6521 | **+0.2717** |
| 032_WSD_id_4_WebService_tr_4559_1st_1182... | 0.7409 | 0.9815 | **+0.2406** |
| 096_WSD_id_68_WebService_tr_1341_1st_144... | 0.7451 | 0.9616 | **+0.2165** |
| 103_WSD_id_75_WebService_tr_1375_1st_147... | 0.6119 | 0.8072 | **+0.1953** |
| 006_NAB_id_6_Traffic_tr_2579_1st_5839.cs... | 0.6058 | 0.7992 | **+0.1934** |
| 100_WSD_id_72_WebService_tr_4559_1st_102... | 0.7858 | 0.9567 | **+0.1709** |
| 027_NAB_id_27_Facility_tr_757_1st_2582.c... | 0.7426 | 0.8764 | **+0.1338** |
| 042_WSD_id_14_WebService_tr_3015_1st_311... | 0.8492 | 0.9799 | **+0.1307** |
| 039_WSD_id_11_WebService_tr_1746_1st_184... | 0.7337 | 0.8599 | **+0.1262** |
| 038_WSD_id_10_WebService_tr_4042_1st_414... | 0.8529 | 0.9737 | **+0.1208** |
| 022_NAB_id_22_Facility_tr_1007_1st_2980.... | 0.7361 | 0.8479 | **+0.1118** |
| 089_WSD_id_61_WebService_tr_1010_1st_111... | 0.8777 | 0.9826 | **+0.1049** |
| 108_WSD_id_80_WebService_tr_2368_1st_246... | 0.8252 | 0.9298 | **+0.1046** |
| 125_WSD_id_97_WebService_tr_2217_1st_231... | 0.8724 | 0.9760 | **+0.1036** |

## Top 15 Datasets where Fixed L=50 heavily beat Dynamic L & Tau
These are cases where sweeping L perhaps led to sub-optimal choices (e.g. overfitting L via heuristics) compared to a robust L=50.

| Dataset | Fixed L=50 PR | Dynamic L PR | Difference |
|---------|---------------|--------------|------------|
| 228_MGAB_id_4_Synthetic_tr_25000_1st_389... | 0.6630 | 0.0985 | **-0.5645** |
| 226_MGAB_id_2_Synthetic_tr_20000_1st_451... | 0.6711 | 0.1185 | **-0.5526** |
| 014_NAB_id_14_WebService_tr_500_1st_1045... | 0.7738 | 0.2558 | **-0.5180** |
| 005_NAB_id_5_Traffic_tr_594_1st_1645.csv... | 0.6534 | 0.1469 | **-0.5065** |
| 225_MGAB_id_1_Synthetic_tr_25000_1st_384... | 0.6007 | 0.1276 | **-0.4731** |
| 202_SMD_id_25_Facility_tr_6250_1st_21230... | 0.8496 | 0.3949 | **-0.4547** |
| 051_WSD_id_23_WebService_tr_3919_1st_515... | 0.7538 | 0.3200 | **-0.4338** |
| 142_MSL_id_3_Sensor_tr_1525_1st_4575.csv... | 0.8603 | 0.4312 | **-0.4291** |
| 093_WSD_id_65_WebService_tr_1125_1st_122... | 0.8636 | 0.4550 | **-0.4086** |
| 227_MGAB_id_3_Synthetic_tr_25000_1st_443... | 0.6270 | 0.2382 | **-0.3888** |
| 068_WSD_id_40_WebService_tr_4549_1st_133... | 0.6214 | 0.2392 | **-0.3822** |
| 109_WSD_id_81_WebService_tr_3175_1st_327... | 0.8880 | 0.5439 | **-0.3441** |
| 206_SMD_id_29_Facility_tr_6250_1st_21230... | 0.8633 | 0.5317 | **-0.3316** |
| 080_WSD_id_52_WebService_tr_1538_1st_163... | 0.6219 | 0.3206 | **-0.3013** |
| 224_LTDB_id_9_Medical_tr_4456_1st_4556.c... | 0.8786 | 0.5811 | **-0.2975** |

---
## Grid Search L (Fixed Tau=1) vs Fixed L=50 (Dynamic Tau)
- **Grid L (tau=1) Wins:** 135 datasets
- **Fixed L=50 (Dynamic tau) Wins:** 74 datasets

## Top 15 Datasets where Grid L beat Fixed L=50
| Dataset | Fixed L=50 (Dyn Tau) PR | Grid L (Tau=1) PR | Difference |
|---------|-------------------------|-------------------|------------|
| 080_WSD_id_52_WebService_tr_1538_1st_163... | 0.6219 | 0.8970 | **+0.2751** |
| 032_WSD_id_4_WebService_tr_4559_1st_1182... | 0.7409 | 0.9808 | **+0.2399** |
| 010_NAB_id_10_WebService_tr_500_1st_271.... | 0.5429 | 0.7706 | **+0.2277** |
| 096_WSD_id_68_WebService_tr_1341_1st_144... | 0.7451 | 0.9557 | **+0.2106** |
| 039_WSD_id_11_WebService_tr_1746_1st_184... | 0.7337 | 0.9298 | **+0.1961** |
| 082_WSD_id_54_WebService_tr_4437_1st_480... | 0.7423 | 0.9355 | **+0.1932** |
| 103_WSD_id_75_WebService_tr_1375_1st_147... | 0.6119 | 0.7958 | **+0.1839** |
| 135_WSD_id_107_WebService_tr_3779_1st_38... | 0.7410 | 0.9149 | **+0.1739** |
| 006_NAB_id_6_Traffic_tr_2579_1st_5839.cs... | 0.6058 | 0.7793 | **+0.1735** |
| 139_WSD_id_111_WebService_tr_1701_1st_18... | 0.7109 | 0.8628 | **+0.1519** |
| 072_WSD_id_44_WebService_tr_1691_1st_179... | 0.3804 | 0.5278 | **+0.1474** |
| 051_WSD_id_23_WebService_tr_3919_1st_515... | 0.7538 | 0.8963 | **+0.1425** |
| 066_WSD_id_38_WebService_tr_3320_1st_342... | 0.8084 | 0.9439 | **+0.1355** |
| 042_WSD_id_14_WebService_tr_3015_1st_311... | 0.8492 | 0.9767 | **+0.1275** |
| 079_WSD_id_51_WebService_tr_4559_1st_106... | 0.3808 | 0.5083 | **+0.1275** |

---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*
