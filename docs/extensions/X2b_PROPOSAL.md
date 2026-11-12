# X2-b proposal and the owner decisions it needs

Written on 2026-11-12, after the X2-a headroom pilot (D-124; `reports/session15/x2/pilot/`) and
before any X2-b training. Estimates are computed from the pilot's measured times.

## Owner decisions needed before any X2-b training
1. **Go or stop.** The reading is headroom, by one family. Options: proceed with X2-b on Can
   MH as below; or go back to D-116's alternatives (Square PH or MH, new infrastructure).
2. **The pilot runs.** D-124 says X2-b may count them as "tuning trial 1". In the Level 1
   search space (D-055, D-061) they are the seed-0 runs of trial 2 (cosine: diffusion, BC-RNN,
   BC-MLP) and trial 4 (weight decay 0.1: BC-MLP-chunk), not trial 1 (D-124 outcome note).
   Choose: count each as the seed-0 run of the trial whose configuration it has (its other 15
   checkpoints then evaluated like any trial's; saves 6.1 training-hours and 4.6 h of wall
   time), or retrain all 4 trials per family under a new prefix.
3. **Concurrency** (KI-035). The pilot's concurrency (4 trainings, 2 evaluation containers
   while diffusion trains, 4 after, a third-party container) exceeded the swap guard once.
   Choose one or more: a guard that samples while jobs run and holds launches; at most 1
   evaluation container beside 4 trainings; no third-party containers during X2-b.
4. **Disk** (KI-008). 24 GiB free with a 4 GiB floor. X2-b needs about 17.5 GiB (estimate,
   biased low). A reclaim is a precondition: the uv cache (36 GB; `uv cache prune`) and/or the
   Docker build cache (2.4 GB reclaimable). Frozen runs are never pruned or moved.
5. Still open from Session 14: whether to publish X1 and X3 (v1.1, D-105); the optional
   diffusion side of KI-052. Nothing was pushed, tagged or deployed.

## Proposed X2-b design

For the X2-b preregistration; decided before any X2-b training.
- **Search space:** Level 1's, unchanged (D-055, D-061). Per family: trial 1 is the base
  configuration, trial 2 the cosine schedule, trial 3 lr 3e-4 and trial 4 AdamW weight decay
  0.1. Architecture, horizons, parameter counts, batch and budget are fixed. It was chosen
  before the pilot outcome.
- **Budget:** the frozen one, equal for all families: 100,000 steps, batch 100, checkpoints
  every 5,000 (K = 20), and every checkpoint on all 50 `x2_validation` starts. Then:
  - §7 selection per run (`roboforge.eval.selection.select_checkpoint`, unchanged);
  - §6.3 choice per family (`roboforge.eval.tuning.choose_configuration`, unchanged; tuning
    seed 0, KI-062);
  - the chosen configuration then trains seeds 1 and 2.
- **New code (KI-064):**
  - an X2 suite driver, which runs trial and replicate training inside the binding, the
    validation sweep with the X2 evaluator, §7 and §6.3 by the frozen pure functions, and
    host guards that sample while jobs run; it is resumable;
  - an X2 latency benchmark (X2_harder_task §4).

  Both go in new files only. A research-integrity test checks that no X2-b code path reaches
  `x2_heldout`.
- **Cost (estimates from the pilot's measured times, local CPU, $0; with the pilot runs
  counted, 20 new runs and 460 new evaluations):**
  - **Training:** 30.5 training-hours summed. Diffusion is 5 × 4.62 = 23.1, BC-RNN 5.6,
    BC-MLP-chunk 1.0 and BC-MLP 0.9.
  - **Evaluations:** 64 container-hours at the measured mean of 503 s. This is biased low:
    the 5k–15k checkpoints were not measured and failing episodes run 500 steps (up to
    819 s), and the 4-container rate was not measured.
  - **Wall time:** about 21 h at the pilot's concurrency. The tuning phase takes 12.8 h:
    three diffusion trials at once set the pace, with 2 evaluation containers while they train
    and 4 after. The replicate phase takes 7.9 h. Fewer containers (item 3) make it longer.
  - **Disk:** about 17.5 GiB: 5.6 GiB of runs and 11.9 GiB of evaluation records at
    26.5 MiB each.
  - **If retrained:** 24 runs and 480 evaluations, adding 6.1 training-hours and
    2.8 container-hours.
- **Labels:** development numbers on validation; X2-b steps no X2 held-out start. X2-c
  (`docs/extensions/X2_FREEZE.md`, the held-out suite once, analysis once) is a later stage.
