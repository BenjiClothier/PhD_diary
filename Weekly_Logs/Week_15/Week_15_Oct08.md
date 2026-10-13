# Weekly Log: [8 Oct - 14 Oct]

## 🎯 Focus of the Week
DG_AD (the learned geometric scorer) towards an ICML paper:
- prove what the score estimates;
- check the proof against shapes with known geometry;
- find out what holds the learned projector back;
- make sure the benchmark numbers are honest and properly compared.

Source of truth for every number: `DG_AD/docs/PROGRESS.md`; full write-ups in
`DG_AD/docs/reports/report_2026-10-12.md` and `report_2026-10-13.md`.

Protocol for every VUS-PR figure unless stated:
- TSB-AD-U, Tuning split (48 series) for development, Eva (350) run once on the frozen method;
- VUS-PR with the benchmark's own sliding window per series (`find_length_rank` on the whole series), matching TSB-AD's
  code (windows identical on all 48 Tuning and 350 Eva series; VUS-PR reproduced to 6 decimals on the 6 series
  checked, 12 Oct);
- networks seed 0 unless stated.

**The method in one line.** A denoiser $m(y)$ and a metric network $2\Gamma$ (a learned tangent projector) are
trained on healthy windows. The score is the orthogonal part of the residual, $\lVert(I-2\Gamma(m))(y-m)\rVert$.

---

## 📝 To-Do List
- [x] Proof of what the score estimates: exact error bound, rate, detection margin
- [x] Check the proof on shapes with known geometry: ideal quantities, then trained networks
- [x] Seed variance of the frozen method (30 seeds)
- [x] "Are we cheating?" check of the evaluation, and simple baselines run ourselves
- [x] Why the projector adds so little, and what holds it back
- [~] Introduction, Background, Method and Experiments-protocol drafted; results and theory sections not yet
- [~] Making the denoiser a true projection (step A/B): works on noise-free toys; step B diverges on the noisy sine
- [ ] Eva seeds: extra seeds of the frozen method, or a stated caveat (decision needed)

---

## 🔬 Progress & Experiments

**Proof note (8 Oct), `DG_AD/docs/paper/icml/proof.tex`.**
- Exact error bound $\lvert s-d\rvert\le\lvert N(m-p)\rvert+\lVert G-P\rVert\,\lvert r\rvert$, where:
  - $s$ is our score and $d$ the true distance to the healthy set;
  - $p$ is the nearest healthy point and $r=y-m$;
  - $N$ is the orthogonal projection at $p$;
  - $G$ and $P$ are the learned and true tangent projectors.

  In words: score error ≤ denoiser's orthogonal error + projector error × residual.
- A rate via Shen et al. (conditional), and a detection margin: when an anomaly is guaranteed to rank above every
  healthy window.
- Centring result: the centred target is needed; the uncentred one is blind at about $\sqrt\varepsilon$.

**Theory checks on shapes with known geometry (9 Oct).**
- With ideal (exact) quantities, the bound is never violated and is close to tight (median ratio 0.88). The healthy
  score equals the predicted curvature floor to within 1%. The score does not depend on sampling density to first
  order.
- With trained networks, the denoiser is close to ideal; the metric network's projector error is 5–18× ideal, mostly
  in eigenvalue *size*, not direction. Rescaling after training did not help.

**NAIGN (a gradient-map centre), 8–9 Oct.** No average difference from the current method (−0.006 [−0.042, +0.029])
and unstable training. Set aside.

**Seed variance (weekend, 30 seeds × 48 Tuning series).** Mean 0.521, SD 0.0095. The reported seed 0 (0.533) is the
second-highest of 30. Against the Hankel detectors: +0.014 (local) and +0.007 (global), neither distinguishable from 0.

**Evaluation check, "are we cheating?" (12 Oct).** Our sliding windows match TSB-AD's on 48/48 Tuning and 350/350 Eva
series, and TSB-AD's own metric code reproduces our VUS-PR to 6 decimals on the 6 Tuning series checked. Eva (seed 0) against the leaderboard leader
(TimeRCD_MAFT, quoted): +0.008 [−0.031, +0.046].

**Five baselines run ourselves through the same pipeline (Tuning, 12 Oct).**
*Why:* to separate what the learned geometry contributes from what the pipeline contributes. Every baseline uses exactly
our pipeline:
- window length $L$ from the period rule on the training prefix;
- series z-normalised with prefix statistics, windows divided by $\sqrt L$;
- fitted on the training-prefix windows only;
- the same Gaussian window-to-point mapping and the same VUS-PR.

Only the per-window score changes. Settings were fixed before running (`DG_AD/experiments/baselines/run_baselines.py`).

| baseline | what it scores for a window $y$ | mean VUS-PR |
|---|---|---|
| **B2 denoiser residual** | $\lVert y-m(y)\rVert$ with our trained denoiser (seed 0) at $\varepsilon_{eval}$. Our score *without the projector*. | 0.515 |
| **B4 kernel version of our score** | Our score with both networks replaced by their exact formulas, computed from the 8192 nearest training windows: Gaussian-weighted barycentre $m(y)$, then the weighted covariance at $m$ divided by $\varepsilon$ as $2\Gamma$. *Learning replaced by direct computation.* | 0.514 |
| **B1 nearest-neighbour distance** | Distance from $y$ to the closest training window. A training window may not match windows less than $L$ steps from itself, otherwise it would match its own overlapping neighbours. | 0.505 |
| **B3 PCA residual** | Distance from $y$ to the principal subspace of the training windows. The dimension $p$ = number of eigenvalues above the noise edge $\mathrm{median}(\lambda)(1+\sqrt{L/n})^2$ ($n$ = number of training windows). *A single flat subspace instead of a curved set.* | 0.494 |
| **B5 forecast residual** | A linear autoregression on the previous $L$ values (ridge-regularised), fitted on the prefix; score = absolute one-step prediction error at each time point. *A forecasting method, for contrast.* | 0.385 |

Our method: 0.521 (30-seed mean). All except B5 are not distinguishable from it (paired bootstrap CIs include 0). So on
Tuning the learned geometry does not beat a plain distance computed the same way; most of our competitiveness comes from
the shared pipeline.

**Why TSB-AD's own nearest-neighbour and PCA detectors score lower (12 Oct).** The largest single factor is how window
scores become time-point scores: we use a Gaussian average over the windows covering each point, TSB-AD the window
centre. This adds +0.05 and +0.07. The paper must say so; leaderboard comparisons are not like for like in this respect.

**Window length (12 Oct).** The best window length follows the anomaly length (Spearman +0.64 [+0.37, +0.83]) for our
method and the simple detectors alike. The period rule is the best fixed choice; a label-chosen length per series would
add +0.13 (a ceiling).

**Point-by-point readout (12 Oct).**
*Why:* the benchmark scores time points, but our score gives one number per window, and each time point lies in $L$
overlapping windows.

*Current readout ("Gaussian").* Each window gets one number $s_w=\lVert q_w\rVert$. The score of time $t$ is a weighted
mean of $s_w$ over the $L$ windows containing $t$, with Gaussian weights centred on each window's middle (s.d. $L/4$). A
one-point spike makes every window containing it score high, so the spike is smeared over about $L/2$ points.

*Point-by-point readout.* The score is built from the vector, not its length. For window $w$ starting at time $w$, the
orthogonal residual $q_w=(I-2\Gamma(m))(y_w-m(y_w))$ has $L$ entries, and entry $i$ belongs to time $w+i$. The score of
time $t$ is
$$\text{score}(t)=\sqrt{\operatorname{mean}_{w:\,w\le t\le w+L-1}\; q_w[t-w]^2},$$
the average squared residual *at that time point*, over all windows that contain it. A spike then appears only at its
own time step.

- *Implementation:* `DG_AD/experiments/metric_diag/point_readout.py: overlap`. For each position $j$ in the window, add
  column $j$ of the $(n_{\rm windows}\times L)$ matrix $q^2$ to the time points $j\dots j+n_{\rm windows}-1$, then divide
  by how many windows covered each point.
- *Variant tested:* dividing each position by its healthy spread first ("scaled"). It made no difference.

*Results (Tuning, by anomaly length relative to the period $T$):*
- spike-type series (anomalies < 0.1 T, 11 series): 0.29 → **0.52**; e.g. 560_YAHOO 0.006 → 1.000;
- anomalies 0.1–2 T: 0.62 → 0.38;
- longer than 2 T: 0.58 → 0.45.

So no single readout suits all anomaly lengths. Under the point readout the projector's contribution over the plain
residual doubles, +0.034 [+0.004, +0.069] against +0.018. The time-resolved residual keeps *where in the window* the
departure is, which the window's length throws away.

**What holds the projector back (12–13 Oct).**
- On average only about 7% of the residual is tangent, the same for healthy and anomalous windows, so there is little
  for a projector to remove.
- Larger noise scales hurt both the projector score and the plain residual equally. The hypothesis that the projector
  pays off at larger scales was **not supported**.
- Separate scales for the two networks, and a per-window scale: no gain (within ±0.012).
- Directions are adequate: on dense series RMM's own direction error is 0.25, comparable to the original paper near
  $10^5$ samples. Eigenvalue *sizes* are off by about 0.29.
- Clipping eigenvalues to [0, 1] or snapping to {0, 1} makes things worse. The soft sizes carry information, and
  eigenvalues above 1 are signal on some series and harmful on others.

**Does VUS-PR reward smoothing? (13 Oct, tested).** Mostly no. The Gaussian readout's advantage on longer anomalies
holds with no buffer at all (point-wise AUC-PR). The buffer only matters for 1–3-point anomalies.

**Noise scale against curvature (13 Oct, derived and checked).** The projector's orthogonal eigenvalue is about
$\sigma^2/(\sigma^2+\varepsilon)+\varepsilon/2R^2$, with $\sigma$ the noise s.d. and $R$ the curvature radius. The best
scale is $\varepsilon^*\approx\sqrt2\,\sigma R$, which matches exact computation on circles to within the grid spacing.

**Making the denoiser project onto the healthy set (13–14 Oct, toys).**

*The problem.* The score is meant to measure the distance from $y$ to its nearest healthy point $p^*$. The denoiser does
not return $p^*$. It returns a Gaussian-weighted average (barycentre) of the healthy windows near $y$, which is off $p^*$
in two ways:
- it slides *along* the set towards frequently visited regions (density pull);
- on curved parts it sits *inside* the curve (curvature sag, about $(\varepsilon/2)\lvert H\rvert$).

Both errors enter the error bound: the denoiser term directly, and the projector term because $2\Gamma$ is evaluated at
$m$ instead of $p^*$. If the denoiser returned $p^*$ exactly, $y-m$ would be purely orthogonal and the plain distance
$\lVert y-m\rVert$ would equal the true distance, with no projector needed.

*Step A: find $p^*$ by iterating the two networks (inference only; no training).* Start at $x=m(y)$ and repeat 20 times:
1. **Tangent step:** $x\leftarrow x+2\Gamma(x)\,(y-x)$. $2\Gamma(x)$ keeps only the part of $(y-x)$ that runs *along* the
   set, so $x$ slides along the set towards the point level with $y$. On a straight line one step lands exactly at the
   foot of $y$; on a curve it moves along the tangent and leaves the set slightly.
2. **Pull back:** $x\leftarrow m(x,\varepsilon_{pull})$. The denoiser pulls $x$ back onto the set. A small
   $\varepsilon_{pull}$ (we tried $\varepsilon$ and $\varepsilon/4$) means a small curvature sag.

The iteration stops moving when the tangent step is zero, i.e. $y-x$ has no part along the set, *and* $x$ is on the set.
Those are exactly the conditions defining the nearest point $p^*$. So step A converges towards $p^*$, up to the pull-back's
own small sag. Cost: about 40 network evaluations per window instead of 1.

*In the language of optimisation on manifolds* (the framing we will use; not "drifting"). Finding $p^*$ means minimising
$f(x)=\tfrac12\lVert x-y\rVert^2$ over the healthy set.
- The tangent step is a **Riemannian gradient step**, exactly RMM §4.4: the learned $2\Gamma$ turns the ordinary gradient
  $x-y$ into its part along the set.
- The pull-back is a **retraction** back onto the set, which RMM's §4.4 does not have; ours is the denoiser.
- So step A is Riemannian gradient descent with a learned retraction. It stops where the Riemannian gradient
  $2\Gamma(x)(x-y)$ is zero, which is where our score equals the true distance.
- Step B (below) is **amortised optimisation**: training a network to output the optimiser's answer in one pass.

*Step B: teach the denoiser to output that point directly (training).* Step A is slow at inference, and its quality
depends on the networks. Step B trains the denoiser so that its single output already is the step-A answer. Each round:
1. **Denoiser update** (500 epochs, learning rate $3\times10^{-5}$). For each training pair (healthy window $x$, noise
   level $\varepsilon$, noisy copy $y=x+\sqrt\varepsilon\,\xi$):
   - run step A from $y$ *with the current networks* to get a target $x^*(y)$, with no gradient flowing through it (a
     "stop-gradient": the target is treated as a fixed number);
   - loss $=\lVert m(y)-x^*(y)\rVert^2/\varepsilon\;+\;\lVert m(x)-x\rVert^2/\varepsilon$.
   - The first term says: output where the iteration would have ended. The second, the **anchor**, says: a healthy
     window is already on the set, so leave it where it is.
2. **Metric-network refit:** retrain $2\Gamma$ with the usual centred loss, now centred on the new denoiser.

3 rounds; each round's targets come from the previous round's networks.

*Why the stop-gradient and the anchor.* Without them the easiest way to make the loss small is cheating:
- the denoiser returns its input ($m(y)=y$), or the projector becomes the identity;
- then every score is zero.

The idea of the frozen (stop-gradient) target comes from Turan et al.'s analysis of drifting models (their Theorem 5.6):
without it, such training can reach a stationary point by collapse instead of by moving towards the answer. The anchor ties the denoiser to the real data.

*Results (noise-free circle, ellipse, sphere, torus; 3 seeds; anomaly at distance 0.4 from the set):*

| | distance of the point used to $p^*$ | score error $\lvert s-d\rvert$ | healthy floor (score at $d=0$) |
|---|---|---|---|
| current denoiser $m(y)$ | 0.036 | 0.031 | 0.0078 |
| step A (iterate, no training) | 0.011 | 0.005 | 0.0051 |
| step B (3 rounds), plain $m(y)$ | 0.0075 | 0.007 | 0.0012 |

- No collapse: the scores still track the true distance.
- After step B the plain distance $\lVert y-m(y)\rVert$ is as accurate as the projected score. The projector's role moves
  from correcting the score to steering the denoiser.
- **On the noisy sine, step B diverges** (non-finite within 1–2 rounds), while step A alone still helps (score error
  about 0.014 → 0.003–0.005 for $d$ = 0.05–0.2).
- Untested candidate causes:
  - projector eigenvalues above 1 make the tangent step overshoot;
  - the anchor forces *noisy* windows to be fixed points;
  - targets are built at large noise levels.

Tuning not tried.

**Ideas and reading (13–14 Oct).**
- Geometric drift operators (`DG_AD/docs/note_geometric_drift_2026-10-13.md`), including a density-free drift with
  orthogonal attraction and tangent repulsion. Not run.
- Paper reviews: diffusion maps (Nadler et al.), RMM §§4.3–4.4, "Generative Drifting is Secretly Score Matching",
  and "When Scores Learn Geometry" (geometry dominates the score by $\sigma^{-2}$; Tempered-Score Langevin samples
  uniformly on the set).
- Coifman–Lafon density normalisation of the denoiser's weights: running.

**Writing (ICML draft, `DG_AD/docs/paper/icml/`).**
- Introduction (one page);
- Background trimmed;
- Method (about 1.2 pages, algorithm box);
- Experiments protocol and baselines (half a page);
- compiles at 7 pages.

---

## 📊 Results & Insights
- **Headline result (Eva, seed 0, one shot):** 0.594 mean VUS-PR, level with the leaderboard leader (+0.008
  [−0.031, +0.046]), a foundation model. The single seed carries an unquantified uncertainty of about ±0.01.
- **But on Tuning the learned geometry is not distinguishable from simple distances run through the same pipeline.**
  The pipeline choices (window length, prefix normalisation, Gaussian readout) carry most of the result.
- **The theory holds where its assumptions hold** (toys), but real windows rarely meet them: noisy, finitely sampled,
  often sparse at our noise scale.
- **The projector's average contribution is small (+0.018)** because little of the residual is tangent. Its effects are
  large per series, in both directions. The most promising route found is to use $\Gamma$ to make the denoiser
  project (step A/B), not to correct the score afterwards.
- Window length and readout are the biggest levers, and both depend on anomaly length.

---

## ⏭️ Next Steps
- **Decisions for the meeting:**
  1. Eva seeds or a caveat;
  2. paper framing: an honest "geometric distance with a known error bound, level with the leader" rather than "learned
     geometry beats baselines";
  3. which of step B / readout / density normalisation to pursue before the deadline.
- Diagnose step B's divergence on the noisy sine; step A on Tuning (inference only) meanwhile.
- Coifman–Lafon normalisation results (running).
- A readout that serves both short and long anomalies without knowing the anomaly length.
- Results and theory sections of the paper; an ablation table.
