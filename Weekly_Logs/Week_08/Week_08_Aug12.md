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
*What were the results? What broke? What worked?*
- Insight 1: 
- Insight 2: 

*(Tip: You can use markdown to embed images like `![Result Plot](/path/to/plot.png)`)*

---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*
