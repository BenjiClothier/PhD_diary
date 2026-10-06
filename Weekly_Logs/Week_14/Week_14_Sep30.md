# Weekly Log: [30 Sept - 7 Oct]

## 🎯 Focus of the Week
Act on the outcome of the 30 Sept supervisor meeting. First close out the
open question from Week 13 (how to combine the local and global scorers
without using future data), then turn to the meeting's to-do list, with
writing Hankel_AD up as a conference paper as the main thread. From 2 Oct
the work moved increasingly to DG_AD (a learned version of the same idea).

Source of truth for numbers: `Hankel_AD/docs/PROGRESS.md` and
`DG_AD/docs/PROGRESS.md`. Protocol for every VUS-PR figure unless stated
otherwise: TSB-AD-U, Tuning split (48 datasets) for development, Eva split
(350) for single final runs, `sliding_window` derived per dataset,
z-normalisation from the training prefix only (after the 1 Oct audit).

---

## 📝 To-Do List

**Carried over from Week 13:**
- [x] Try the "simple step" for combining scorers: only trust the local
  scorer's alarms where the global scorer is at least mildly suspicious.
  It failed (see below), so the whole combination question is written up
  in `Hankel_AD/docs/local_global_combination_report.md` and set aside.

**From the 30 Sept meeting:**
- [~] Write up Hankel Anomaly Detection as a paper ready to submit to a
  high-level conference. *Internal draft done
  (`Hankel_AD/docs/paper/two_beliefs_delay_embedding_ad.pdf`), deliberately
  not for submission: novelty is modest. Needs a decision with supervisors
  (see Next Steps).*
- [x] Can Hankel AD be used for volcano anomaly detection? *Tried on
  Whakaari, 5–6 Oct: no. A plain tremor-level alarm does better. Report and
  deck below.*
- [x] Why does the Hankel method work? (see Progress)
- [x] Explain in detail, with examples, why local works for OPPORTUNITY
  and not for others. (see Progress)
- [~] Can we pre-normalise the data so we don't need MAD? *Per-window
  centring tested 5 Oct: fixes drift, breaks OPPORTUNITY. A label-free
  "centre only when the training data show one level" rule is not yet run.*
- [~] Why is curvature not working? *Largely answered (1–2 Oct); a new
  network-based estimator works on shapes and on delay windows with known
  curvature (6 Oct), not yet on real series.*
- [ ] Hankel offers low complexity: is there an application in distributed
  anomaly detection? *Design note only
  (`Hankel_AD/docs/design_distributed_anomaly_detection.md`), not run.*

---

## 🔬 Progress & Experiments

**Gated combination ("simple step"), 30 Sept:**
- **Objective:** combine the local and global scorers without future
  data by trusting local's alarms only where global's robust z-score
  is above 1.
- **Setup:** training-only normalisation, gate fixed at 1 in advance,
  Tuning-48. Pass criteria stated before running: no significant loss
  vs. global; both Tuning OPPORTUNITY datasets ≥ 0.9; mean ≥ 0.5177.
- **Result:** mean 0.5012 vs. global 0.5295 (Wilcoxon p = 0.73);
  OPPORTUNITY 0.430 / 0.172 — fails, as predicted in advance (global is
  inverted on OPPORTUNITY, so it is rarely suspicious where the anomalies
  are). It did fix one false-alarm case (`386_UCR_id_84` 0.092 → 0.914).
- **Report:** `Hankel_AD/docs/local_global_combination_report.md` covers
  all five combination variants tried (four normalisations plus gating),
  each failing a different pre-stated criterion, and why: drift and
  anomalies look the same to any scaling rule, and `max()` lets one
  scorer's false alarms through.

**Why the Hankel method works, and when local wins, 30 Sept** (numbers
refreshed 5 Oct with the audit fixes):
- **Objective:** answer the two meeting questions with evidence, not
  assertion.
- **Setup:** a controlled synthetic test of four mechanisms (known ground
  truth, fixed L and d), an Eva family breakdown, and three real examples
  with figures.
- **Result:** write-up in `Hankel_AD/docs/why_global_and_local_work.md`.
  Global generalises within the one linear structure it learns (healthy
  windows at an unseen amplitude barely change its score, ×0.99 vs. ×2.46
  for local; first reported as ×1.02 vs. ×2.51 before the audit), and
  fails when the healthy data is several regimes: in a synthetic
  two-regime case an anomaly mixing the regimes lies inside the fitted
  subspace, VUS-PR 0.08 global vs. 1.00 local, the OPPORTUNITY pattern.
  Local remembers every training window, so it fails on any healthy data
  it hasn't seen: drift (`560_YAHOO`), chaotic dynamics. One prediction was
  refuted: chaos does not favour local (Lorenz: global 0.971 vs. local
  0.397 after the audit; 0.529 before). `142_MSL` no longer separates the
  scorers once normalisation uses the training prefix only (both 0.075).

**Reflection: is this novel, and what is the narrative? (30 Sept)**

*The honest assessment.* The Hankel_AD work so far is largely piecemeal
engineering. The detector itself is not novel: a subspace residual on
delay vectors, close to singular spectrum analysis and to Sub-PCA on the
TSB-AD leaderboard. The (L, d) selection is a stack of heuristics. The
last two weeks were reactive — ensemble, then five normalisations, then
gating — each patching the previous failure. That sequence is a
well-documented negative result, not a contribution. It will probably
make a thesis chapter rather than a stand-alone paper.

*What is not piecemeal.* Read past the engineering and the analysis keeps
pointing at one idea. Takens' theorem guarantees that healthy delay
vectors lie on a manifold; each detector is a different approximation of
that manifold, and where an anomaly lies relative to it decides who can
see it:

| where the anomaly lies | global (linear span) | local (training samples) | example |
| :--- | :---: | :---: | :--- |
| outside the span | ✓ | ✓ | spikes; synthetic shape break |
| inside the span, off the manifold | ✗ | ✓ | OPPORTUNITY; MGAB; synthetic two regimes, Lorenz parameter change |
| healthy but outside training coverage (false alarm) | ✓ tolerates | ✗ flags | drift (`560_YAHOO`); synthetic novel normal |

The failure of every normalisation fix is the same fact from the scaling
side: from scores alone, "healthy but not yet covered" and "anomalous"
look identical.

*Candidate narrative.* "What a linear delay-embedding detector can and
cannot see, and how evaluation choices hide it." In the thesis, this
chapter becomes the linear approximation and exactly where it breaks;
the following chapters (local tangent geometry, Hankel_CDC_AD; learned
geometry, DG_AD) model the manifold between the two extremes.

*What could be genuinely novel (as listed on 30 Sept, before testing):*
1. **One scale parameter linking global and local.** *Tested 30 Sept:
   false as stated* — the noise scale moves the empirical belief between
   "nearest neighbour" and "distance from the mean window", never to the
   global detector. The belief and the noise level are separate choices.
2. **The geometric taxonomy above,** validated by predicting each
   dataset's winner in advance and then checking. *Done: passes (below).*
3. **Evaluation pitfalls, quantified across methods.** *Measured for our
   detectors; not yet for other leaderboard methods.*

**Literature check on novelty (30 Sept):** full write-up in
`Hankel_AD/docs/literature_check_novelty.md`; key claims re-checked
against the sources.
- **The detector is already known.** Moskvina & Zhigljavsky (2003) is
  essentially its core (subspace of the uncentred trajectory-matrix
  covariance, distance to it), used for change points. It must be cited
  as the direct antecedent. TSB-AD's Sub-PCA is a different, weaker
  method, so beating it doesn't show novelty.
- **The scale idea is partly known.** Hoffmann (2007), kernel PCA for
  novelty detection: small kernel width behaves like a Parzen density,
  large width approaches standard PCA. DTE (ICLR 2024) links small noise
  to kNN. It becomes a bridge to DG_AD, not a headline.
- **The geometric taxonomy (span / manifold / coverage) was not found**
  in the main time-series anomaly taxonomies.
- **Evaluation pitfalls are mixed.** Contaminated training prefixes are
  already known (a TSB-AD GitHub issue). Whole-series normalisation is a
  stated convention whose effect is unmeasured. Whole-series score
  standardisation looks new.
- **Correction:** MGAB anomalies are splices (a matching segment removed
  and the ends joined), not changes of dynamics.

**Criticism of the taxonomy (30 Sept): the embedding isn't guaranteed to
be good.** Takens' embedding is sensitive to L and τ; τ is fixed at 1 to
avoid a parameter, and v4_3 picks L for detection power, not embedding
quality. The taxonomy's categories remain well-defined for the point
cloud actually built; the narrative should say "geometry of the delay
vectors as built", with Takens as motivation.

**Pre-registered blind test of the geometric taxonomy (30 Sept; recomputed
1 Oct with the audit fixes): passes.** Rule, set and criteria fixed in
advance (`Hankel_AD/docs/preregistration_geometry_predicts_winner.md`);
472 datasets never scored by the local scorer. The rule compares, on the
training prefix only, how far new healthy windows are from the fitted
subspace vs. from the nearest earlier window (Q = ratio of the two
medians), and predicts the detector with the smaller gap.
- Spearman with (global − local) VUS-PR: **0.572** after the audit fixes
  (0.596 before; needed ≥ 0.37).
- Accuracy on 301 clear-winner datasets: **0.837** vs. 0.784 for always
  predicting global (p = 0.013; 0.855 on 296 before the fixes).
- **Robust in L** (2 Oct): passes at half and double the selected L
  (Spearman 0.500 and 0.614).
- Caveat: within families the ordering mostly holds, but the fixed cutoff
  doesn't beat each family's own majority winner — part of the result is
  between-family.
- **Used as a switch (7 Oct, described, not pre-registered):** taking the
  local detector where the rule says so raises the blind-set mean from
  0.420 to 0.435 (+0.015 [+0.002, +0.029]); on Tuning-48, 0.514 → 0.541
  (+0.026 [−0.003, +0.063]). About a third of the way to the label-using
  best-of-two.

**Maths guide: the same detector with two beliefs (30 Sept–1 Oct).**
`Hankel_AD/docs/two_beliefs_maths_guide.md`, then the unified
`Hankel_AD/docs/mathematical_foundations.md`. Starting from Tweedie's
formula, both scorers are "guess the healthy window, measure the gap"
(a denoising residual). The global scorer is exactly this under the
belief "healthy data is one Gaussian" (at noise γ²); the local scorer is
the limit under "healthy data is exactly the training windows", at small
noise. Changing the noise level never turns one into the other.

**Ideas that did not work (1 Oct):**
- **Toeplitz deviation as a winner predictor:** Spearman 0.044, adds
  nothing to Q.
- **Choosing the noise level by likelihood cross-validation:** drives it
  to the local scorer (memorisation), no gain (0.488 vs. local 0.492 on
  43 valid datasets).
- **Richer taxonomy, step 1 (synthetic sequence anomalies):** main
  hypothesis falsified; most sequence anomalies collapse into the static
  categories once L spans the event (phase jump is the exception).

**Code and theory audit (1 Oct).** `Hankel_AD/docs/audit_2026-10-01.md`.
Core identities confirmed; five code problems found and fixed (training
windows matched to themselves in the local scorer; whole-series
normalisation by default; no floor on γ²; AUC-PR computation; missing
Eva scores); the (L, d) selection's derivation does not support its
formula and its periodicity test labels white noise periodic. Protocol-
clean Eva result for the global detector: mean 0.5729, median 0.6720.

**Curvature (1–2 Oct, 6 Oct).** The kernel estimate resolves curvature on
only 1 of 27 real series. A new estimator from DG_AD's networks works on
shapes with known curvature (circle 0.057, sphere 0.076 relative error)
and on delay windows of a sine and a torus (errors 9–25%), but its
automatic noise-level rule failed, and curvature does not explain the
learned scorer's high healthy scores.

**Paper (2 Oct).** Internal draft written up for supervisors
(`Hankel_AD/docs/paper/two_beliefs_delay_embedding_ad.pdf`), later
humanised; not for submission.

**Per-window centring of the input (5 Oct, Tuning-48):** local becomes
level with global (0.516 vs. 0.514, p = 0.68), but OPPORTUNITY breaks
(0.10 / 0.08). The window's level is noise for drift but signal for
multiple regimes.

**Whakaari / White Island volcano (5–6 Oct).** Tremor data 2011–2020,
five eruptions (2019: 9 Dec, 01:11 UTC). Eight experiments
(`Hankel_AD/docs/report_whakaari_2026-10-06.md`). A plain RSAM-level alarm
catches 4 of 5 eruptions at 1.1 false alarms a year; the best Hankel
detector needs 8.1 for the same. The detectors respond (a rise about 16 h
before 2019), but the level alarm already warns 16.2 h ahead. Choosing a
detector held out: about 80 candidates catch 3 of 5, about 400 catch
0 of 5 (selection overfitting). Why
(`Hankel_AD/docs/note_why_the_subspace_is_not_meaningful.md`): the
healthy-window subspace of these red-noise-like features is generic (its
first pattern is the constant, |cos| ≥ 0.996), so the global detector
measures roughness, and the precursors are smooth. Tested on the
benchmark: neither subspace genericness nor anomaly sharpness predicts
performance in general (sharpness explains the SMD cases).

**DG_AD: a learned version of the scorer (2–7 Oct).** Two networks
trained on healthy windows: a denoiser m(y) and a metric network whose
2Γ acts as the tangent projector; score = size of the residual's
orthogonal part.
- **Learned tangents (2 Oct):** faithful Riemannian Metric Matching
  estimates the dimension correctly (100% of points) but is not more
  accurate than tuned local PCA on simple shapes.
- **Synthetic cases (2 Oct):** the learned score matches local on
  regime mixtures (0.992) and beats both on the Lorenz parameter change
  (0.567), but lags global on Lorenz fast oscillation (0.629).
- **Tuning-48, first run (2 Oct):** 0.465, below the local detector at
  the same window (0.507, p = 0.030), losses concentrated on long windows.
- **Projector at the denoised point (5 Oct, your suggestion):** fixes
  Lorenz (0.629 → 0.947) at a cost on two regimes (0.992 → 0.897).
- **The dead spot (5–6 Oct):** windows between two regimes are averaged,
  not projected. Repeated denoising and a flow-based version fix it but
  cost Lorenz; the cause traced to the denoiser being inaccurate at small
  noise levels (an "error floor") with the old training noise range.
- **Lorenz false alarms (6 Oct):** healthy windows from rare stretches of
  the attractor that the training data never covered (5× further from any
  training window). Not noise, not curvature, not a learning failure.
- **Tuning-48 rerun with retrained networks (6 Oct):** **0.533**, above
  the 2 Oct run (+0.068 [+0.016, +0.128]) and level with local (0.507)
  and the published global (0.514); differences not distinguishable from
  zero. The gain goes with centring the training noise on the evaluation
  level, not with scaling it to window length. No (L, d) estimator used.
- **Cross-fitting (6–7 Oct):** a two-half version had a bug (networks
  trained on one window); 5-fold works (0.542) but moves single datasets a
  lot. On 429_UCR it exposed the metric network giving 2Γ eigenvalues up
  to about 9 (should be ≤ 1) and so amplifying residuals: fix = clip
  eigenvalues to [0, 1] (being tested).
- **Where global still wins:** mostly window length (7 of 10 datasets),
  drift on 2. A fixed longer window (2–4 periods) is worse on average.
- **Stretch scan (7 Oct):** score each window's residual in short and long
  stretches, calibrated on healthy windows, and give the score to the
  stretch that stands out. Removes the long-window loss on short synthetic
  anomalies (local detector 0.91–0.96 at every window length vs. 0.33
  whole-window at the longest); costs on Lorenz. Tuning-48 run in progress.
- **Nearest-point maps (6 Oct):** on 2-D shapes, an idempotent network
  built as the gradient of a scalar lands 2–5× closer to the true nearest
  point than the denoiser.

**Presentations and reports (7 Oct):** two-beliefs evidence report and
deck, proofs deck, Whakaari "why it failed" deck
(`Hankel_AD/docs/presentations/`).

**Working-practice changes (6–7 Oct, in all three CLAUDE.md files):** no
numeric pass/fail thresholds (state the question, measure, conclude from
results); diagnose shortfalls before concluding; report every experiment
from scratch; Hankel's dimension values not used as evidence in DG_AD.

---

## 📊 Results & Insights
- The two Hankel detectors are one quantity (a denoising residual) under
  two beliefs about healthy data, proved and checked to machine
  precision; the beliefs predict where each fails, and a label-free
  geometry rule predicts the winner blind (ρ = 0.572).
- The learned DG_AD scorer is now level with the Hankel detectors on
  Tuning-48 (0.533–0.542 vs. 0.514 global, 0.541 for the Hankel switch),
  without the (L, d) estimator. Not yet better.
- The biggest remaining lever is window length, and per-dataset choices
  matter in both directions; the stretch scan is the current attempt to
  make the choice matter less.
- Whakaari was a clear negative: the subspace is generic for these
  features, so the detector sees roughness, not smooth precursors.

---

## ⏭️ Next Steps
- **Decisions for the supervisor meeting:** what to do with the Hankel
  paper (as is, or a short benchmark-practice paper on evaluation
  pitfalls); which method to freeze for the one-shot Eva test.
- Finish the Tuning-48 stretch scan, including the eigenvalue clipping.
- A label-free, per-dataset choice of window length and noise level.
- Drift: treat the constant direction as always tangent (560_YAHOO,
  182_SMD).
- Retrain the synthetic networks with low-centred training noise and
  retry the dead-spot fixes.
- Nearest-point maps: diagnose the distance network; stage 2 on delay
  windows.
