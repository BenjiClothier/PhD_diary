# Weekly Log: [30 Sept - 7 Oct]

## 🎯 Focus of the Week
Act on the outcome of the 30 Sept supervisor meeting. First close out the
open question from Week 13 (how to combine the local and global scorers
without using future data), then turn to the meeting's to-do list, with
writing Hankel_AD up as a conference paper as the main thread.

Source of truth for numbers: `Hankel_AD/docs/PROGRESS.md`. Protocol for
every VUS-PR figure unless stated otherwise: TSB-AD-U, Tuning split (48
datasets) for development, Eva split (350) for single final runs,
`sliding_window` derived per dataset.

---

## 📝 To-Do List

**Carried over from Week 13:**
- [x] Try the "simple step" for combining scorers: only trust the local
  scorer's alarms where the global scorer is at least mildly suspicious.
  It failed (see below), so the whole combination question is written up
  in `Hankel_AD/docs/local_global_combination_report.md` and set aside.

**From the 30 Sept meeting:**
- [ ] Write up Hankel Anomaly Detection as a paper ready to submit to a
  high-level conference.
- [ ] Can Hankel AD be used for volcano anomaly detection?
- [ ] Why does the Hankel method work?
- [ ] Explain in detail, with examples, why local works for OPPORTUNITY
  and not for others.
- [ ] Can we pre-normalise the data so we don't need MAD?
- [ ] Why is curvature not working?
- [ ] Hankel offers low complexity: is there an application in distributed
  anomaly detection?

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

---

## 📊 Results & Insights

---

## ⏭️ Next Steps
