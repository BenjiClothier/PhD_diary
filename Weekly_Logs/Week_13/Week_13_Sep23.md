# Weekly Log: [23 - 30 Sept]

## 🎯 Focus of the Week
Two sibling projects, run in parallel:
- **Hankel_AD** (closed-form global-subspace detector, `v4_3`): find out
  where the remaining gap to the leaderboard actually comes from, and fix
  it without adding more tuned constants.
- **Hankel_CDC_AD** (new this week): a *local* scorer built on diffusion
  geometry — the Carré du Champ (CDC) tangent plane at each point's
  nearest healthy neighbour — to test whether local geometry handles
  non-stationary data better than one global subspace.

Source of truth for every number below: `Hankel_AD/docs/PROGRESS.md`
(entries 2026-09-23 to 2026-09-29) and `Hankel_CDC_AD/docs/PROGRESS.md`
(entries 2026-09-22 to 2026-09-29). All VUS-PR figures use TSB-AD's
official protocol: `sliding_window` derived per dataset via
`find_length_rank`, never a fixed value. "Tuning" = the 48-dataset
TSB-AD-U Tuning split (development only); "Eva" = the official
350-dataset evaluation split. TimeRCD-MAFT figures are leaderboard
reference figures, not reproduced here.

---

## 📝 To-Do List
- [x] Citation and documentation integrity: fabricated SSA citation
  (found at the end of Week 12) traced and removed from the slides; stale
  `sliding_window=100` claim removed from the benchmark-rules doc.
- [x] Fix a `max_d` CLI bug that invalidated the "unclamped" ablation.
- [x] Oracle `(L, d)` search (label-informed upper bound) on Tuning and Eva.
- [x] Three new `(L, d)` estimator designs (`v5`, `v6`, `v7`) and a
  noise-pre-whitening test.
- [x] Per-dataset breakdown of the gap to TimeRCD-MAFT; root-cause the
  OPPORTUNITY family failure.
- [x] Build a native local scorer; test it; search for a rule that
  predicts when local beats global; build and test a local+global
  ensemble on Tuning, then once on Eva.
- [x] Hankel_CDC_AD: port the CDC scorer, validate it on synthetic
  manifolds with known ground truth, then compare against plain
  nearest-neighbour distance on real data.
- [x] Hankel_CDC_AD engineering: GPU port of the CDC gradient step, fix
  three memory bottlenecks (one caused a real out-of-memory crash).
- [x] Supervisor slides: [Slides/presentation.pdf](Slides/presentation.pdf)
  (18 slides; full protocol detail and citations in the speaker notes —
  uncomment `show notes on second screen` in the `.tex` to see them).
- [ ] Blind re-validation of the ensemble (see Results caveats).
- [ ] Cap-free CDC reruns on GPU.

---

## 🔬 Progress & Experiments

### Hankel_AD

**Corrections first.**
- The "unclamped" `max_d` rows in last week's ablation were in fact
  run with `max_d=14`, because omitting the flag fell back to the class
  default. Rerun genuinely unclamped (Tuning): mean 0.4995 / median 0.4793
  vs. 0.5295 / 0.5338 with the cap. **Week 12's statement that the cap
  "ties unclamped exactly" was wrong.** The cap does help (+0.03 mean);
  the conclusion to keep `max_d=14` survives. Fixed with an explicit
  `--max_d 0` sentinel.
- The Golyandina & Zhigljavsky (2011) citation is confirmed fabricated
  (the similar real paper, Hassani et al. 2011, derives a different
  window rule). The integer-multiple-of-period rule for `L` is now
  treated as an unsourced hypothesis. Several docs still to be cleaned.

**Oracle `(L, d)` search** — how far is `v4_3` from the best `(L, d)`
per dataset, chosen *with* labels? (A diagnostic upper bound, not a
method.) Coarse grid plus 40-trial Optuna search, run on the ECS grid.
- Eva (345–350 datasets): ours mean 0.5673 / median 0.6647; oracle
  0.7523 / 0.9313 (coarse), 0.7502 / 0.9352 (Optuna).
- The oracle picks much larger `d` (mean 49.7 vs. our 10.7 on Eva) and
  somewhat smaller `L`.

**Four attempts to close that gap by redesigning `(L, d)` selection all
failed** (Tuning, mean VUS-PR vs. `v4_3`'s 0.5295):
- The period diagnostic: the oracle's implied number of cycles
  (`oracle_L / T_dom`) ranges over three orders of magnitude (0.02–25.6),
  so "`L` = cycles × period" is not what the oracle is tracking.
- `v5` (broad search scored by held-out reconstruction whiteness):
  0.4624, with 11/48 datasets collapsing to VUS-PR < 0.05 (the cost
  function rewarded negative autocorrelation).
- `v6` (that loophole closed): 0.4667. `v7` (broad `L` + existing `d`
  rule): 0.4382. The same five datasets collapse under all three.
- Noise pre-whitening before choosing `d` (motivated by a real finding
  that correlated noise erodes the spectral gap): 0.5293 — no change.

**The gap to SOTA is one family, and it is not an `(L, d)` problem.**
Per-dataset comparison against TimeRCD-MAFT (348 datasets): 165 wins,
169 losses, 14 ties, Wilcoxon p = 0.531. The 27 OPPORTUNITY
(HumanActivity) datasets: ours 0.078 vs. 0.729, all 27 lose. Excluding
them, mean difference goes from −0.0196 to +0.0335. `L` ranges 28–256
across the family, all failing alike. On the worst case, anomalies score
*lower* than normal points (median percentile 0.193 vs. 0.503). Working
explanation: the "healthy" training data switches between regimes
(activities), which one global subspace cannot represent.

**Locality fixes that family, but a straight swap breaks the rest.**
Native 1-nearest-neighbour distance scorer (`src/local_scorer.py`), same
`(L, d)`:
- OPPORTUNITY-27: 0.0783 → 0.7558.
- Tuning-48: 0.5295 → 0.4688 (13 wins, 22 losses) — a straight
  replacement is falsified.
- Searched for a label-free rule predicting which scorer wins (spectral
  flatness, GMM cluster counts, a persistent-cohomology diagnostic,
  temporal subspace stability): the best combined signal was Spearman
  ρ = 0.37 — real but far from a usable cut.

**Ensemble: per window, take the larger of the two scorers'
robust z-scores.** No new tuned constants. Two normalisation bugs were
found and fixed during smoke tests before any grid was trusted
(percentile-rank saturation; near-zero MAD on near-duplicate training
windows).
- Tuning-48: 0.5329 vs. global 0.5295 (p = 0.93) — no regression — while
  recovering 26/27 OPPORTUNITY wins (0.7301).
- One variant tried and rejected on its own pre-stated criterion: a
  per-query local-subspace scorer, which failed the case it was built
  for and was 15–20× slower.

### Hankel_CDC_AD

- **Synthetic validation with known ground truth.** Tangent-plane
  reconstruction is essentially exact (flat subspace and sphere,
  principal-angle error ≤ 0.0015). The tangent-plane anomaly score gets
  AUC 1.0 on synthetic off-manifold anomalies and degrades gracefully
  with noise. **Curvature from the same package is badly wrong** (sphere,
  true curvature 1: estimates 4.2 → 24.0 → 59.1 as n grows, root cause
  not found) and was dropped.
- **Real data is much harder.** Tangent frames on real delay vectors are
  inconsistent between neighbouring points (64–99% of the worst case);
  the tangent component is a small part of the residual on the dataset
  studied in detail.
- **CDC vs. plain nearest-neighbour distance, Tuning split** (30/48 clean
  runs): 14 wins, 14 losses, 2 ties, Wilcoxon p = 0.927 — no evidence CDC
  beats plain distance. One exception, root-caused: `386_UCR_id_84`
  (+0.827), where the projection removes a rank inversion.
- **OPPORTUNITY-27:** CDC 0.7496 / plain distance 0.7557 / global 0.078.
  Distance wins 17 of 27. **Conclusion: the fix is locality, not the
  tangent-plane projection.**
- **Engineering:** the package cannot score new points (worked around);
  a GPU port of its gradient step matches the library to ~1e-11 and is
  8.5× faster on a large-`L` dataset; three memory bottlenecks fixed
  (one crashed the workstation); results found to be non-deterministic
  between sessions (cause not identified, double-fit protocol adopted);
  and the compute caps needed on CPU were shown to change VUS-PR by up
  to 0.33 on one dataset, so the capped 48-dataset aggregates are
  provisional.

---

## 📊 Results & Insights

**Headline result** (Hankel_AD ensemble, official Eva split, 350/350
datasets evaluated, `(L, d)` from `v4_3`,
`utils/ensemble_scorer_grid.py --split eva`):

| | mean VUS-PR | median VUS-PR |
| :--- | :---: | :---: |
| Global (`v4_3`) | 0.5671 | 0.6647 |
| Local (1-NN distance) | 0.5447 | 0.5854 |
| **Ensemble** | **0.6266** | **0.7159** |
| TimeRCD-MAFT (reference figure) | 0.5856 | 0.6511 |

Ensemble vs. global: 138 wins, 74 losses, 138 ties; Wilcoxon
p = 5.4×10⁻⁵.

**Caveats that must go with that number:**
- **Not a blind test.** The OPPORTUNITY family used to build and validate
  the ensemble is inside the Eva split.
- **Most of the gain is that family.** Excluding it (323 datasets):
  +0.0100 mean, Wilcoxon p = 0.051 — borderline.
- **Not an apples-to-apples SOTA comparison.** TimeRCD-MAFT is a
  leaderboard figure, not rerun here.
- **The z-score scale uses unlabelled test-period scores.** The MAD is
  taken over all windows, train and test. No labels are used, so this is
  not label leakage, but it is transductive: the test period's unlabelled
  score distribution sets the relative weight of the two scorers. Found
  30 Sept that the underlying reason a training-only MAD failed was a
  self-match (every training window's local score is exactly 0), not
  near-duplicate windows as first logged. A training-only rerun is in
  progress; on the one dataset checked so far (`560_YAHOO_id_10`) it
  brings back the original failure (ensemble 1.000 → 0.006), so the
  aggregate gain may partly depend on the transductive step.
- 7 Tuning datasets still lose badly; the mechanism is understood
  (`max()` cannot suppress one scorer's false positives) but not fixed.
- 4 OPPORTUNITY datasets stay catastrophic under every local method.

**Insights:**
- The remaining gap was never an `(L, d)` problem. Four redesigns
  confirmed this by failing.
- Locality is what matters on multi-regime data. Two independent
  projects reached this separately.
- The CDC tangent-plane machinery did not earn its cost over plain
  distance. This motivates next week's change of direction.
- Several of this week's findings were corrections of earlier claims
  (the `max_d` tie, the citation, the "1.8 points behind SOTA" framing).
  They are logged as findings, not quietly edited.

---

## ⏭️ Next Steps
- Blind re-validation of the ensemble, keeping OPPORTUNITY out of both
  building and evaluation.
- Decide whether a guarded combination rule for the 7 failure cases is
  worth the added complexity.
- Finish the citation clean-up in the Hankel_AD docs.
- Hankel_CDC_AD: cap-free GPU reruns before its Tuning aggregates are
  quoted anywhere.
- A learned-geometry follow-up (new project, DG_AD) is under way. It is
  deliberately not reported here until it is more solid.
