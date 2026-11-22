# RoboForge — Technical Specification

This is the technical specification of RoboForge (version 2.0, prepared 2026-11-04; repository
`github.com/zackholt05-tech/roboforge-observatory`; author Zackary Holt). It states requirements and
does not itself report results. Older documents, code comments and evidence files cite it as
"MASTER_SPEC §N"; section and item numbers are unchanged, so those citations resolve here (numbers
that do not appear, such as 0.1, 0.3 and 12, are unused). "Session N" in this repository names
project stage N (§10), whose evidence is under `logs/sessionN/` and `reports/sessionN/`.

**Mission.** Train and compare imitation-learning policies that control an articulated robot arm in
a real 3D physics simulation: a controlled comparison of deterministic/recurrent behavior cloning
against a conditional diffusion action policy on two manipulation tasks, released with actual
rollout artifacts, an interactive 3D observatory, a reproducible research report and this
repository. robosuite + MuJoCo + robomimic replace the earlier 2D PushT benchmark entirely
(contingent on §2.3); no PushT code, state/action dimensions, simulator assets, dataset URLs or
claims are retained.

# 0. Authorship and methodology rules

## 0.2 Authorship, commit identity and credit
1. The sole named project author is **Zackary Holt**: project byline, report title page, README
   authorship section, website footer/about text, release metadata, repository project fields and
   original project copyright fields.
2. One Git author and committer identity: repository-local `user.name` `Zackary Holt` and a verified
   GitHub-associated `user.email` (a verified GitHub noreply address qualifies), never guessed;
   global Git config is untouched. Checked before committing with `git config --local user.name`,
   `git config --local user.email`, `git var GIT_AUTHOR_IDENT`, `git var GIT_COMMITTER_IDENT`, plus
   repository hooks/templates.
3. No other party is named as author, co-author, committer, credited contributor, release signatory
   or website byline. Commit and tag messages carry no `Co-authored-by:` trailers, bot badges or
   generated acknowledgments; verified with `git log --format=fuller -n 5` and
   `git show -s --format='%an <%ae> | %cn <%ce>%n%B' HEAD`. Published history is rewritten only with
   a recorded approval.
4. Third-party work is credited: dataset creators, the robomimic/robosuite/MuJoCo authors, the
   Diffusion Policy and DP3 papers, reused code/assets/models and other relevant publications are
   cited; licenses and copyright notices are kept in `THIRD_PARTY_NOTICES.md` or their required
   locations. Sole project authorship does not mean sole creation of external algorithms or
   datasets.
5. Statements about the project comply with applicable university, venue or publication disclosure
   rules.
6. Pushes, repository creation, public deployment, paid compute and destructive actions require a
   recorded approval: a decision entry in `docs/DECISIONS.md` (the meaning of "recorded approval"
   throughout). The repository is independent of other projects and imports none of their secrets,
   data, code, customer information or branding.

## 0.4 Methodology and safety rules
- **3D** means articulated-robot control in a 3D physics engine; mesh playback or 3D-looking renders
  of a 2D environment do not qualify.
- Training is offline supervised imitation learning from demonstration data; evaluation is by actual
  closed-loop simulator rollouts. Offline action MSE does not establish manipulation success.
- One simulator/dataset/controller/action convention is fixed before modeling. No action dimension
  (7D or other) is assumed until read from metadata and inspected; Cartesian end-effector delta,
  orientation, joint and gripper actuator conventions are task-specific, not interchangeable.
- Splits are by entire demonstration trajectories, preserving creator/operator partitions when
  available, with no subsequence overlap; normalization is fitted on the training partition only.
- Comparisons equalize data, observation privilege, parameter budgets where practical, training
  selection budget, evaluation states, task horizons, controller settings and safety/clipping rules.
  Exact simulator object poses are privileged state relative to RGB-only policies; observation
  modalities are distinguished explicitly and an unequal-modality comparison is not called causal.
- Validation initial states serve development/model selection; unseen test initial states serve the
  final result. Models and analysis protocol are frozen before the final test is accessed;
  deviations are recorded as exploratory.
- Success is measured over actual closed-loop trials with uncertainty, failures and compute/cost;
  negative results are preserved; paired test initial conditions are mandatory.
- Paid cloud actions need a recorded approval, estimated cost cap, per-job timeout, limited
  concurrency, retained checkpoints and a kill path, after a local test and a tiny remote smoke run.
  Observed costs are reported; the nominal GPU hourly rate is not the complete bill.
- Raw demonstration datasets, pretrained weights, simulator assets and third-party meshes are not
  redistributed unless rights explicitly allow; data terms are verified in Session 0; download
  scripts/manifests are stored instead.
- Recorded physical truth (MuJoCo), predicted action plans, actual intermediate denoising tensors
  and illustrative explanations are kept separate and never visually conflated on the website.
- All outputs and claims come from versioned machine-readable artifacts: no hand-entered statistics,
  no fabricated animation labelled as a trained rollout.

# 1. Project objective, scope and product story

## 1.1 Central scientific question
**Can a conditional diffusion action-sequence policy match or outperform deterministic and recurrent
behavior-cloning policies on closed-loop three-dimensional robotic manipulation, particularly under
variation in object placement and initial robot state?** Also studied: why each model succeeds or
fails, whether action multimodality matters, and how observation/action horizons and
guidance/sampling settings influence performance and latency. The question is empirical; diffusion
is not assumed to win.

## 1.2 The two research levels (order is mandatory)
**Level 1 — required core: 3D robot manipulation.** Policies use a fixed, verified set of
low-dimensional simulator observations (robot proprioception and, if the dataset provides it,
object/goal state): real 3D articulated control, but possibly privileged simulator information
rather than perception. Robust repeated-seed results on a primary and a secondary task; a controlled
RGB+proprioception experiment when technically and economically feasible, reported separately with
observation differences clearly stated. A valid state-based Level 1 paper is not delayed for point
clouds.

**Level 2 — advanced, separately gated: explicit 3D perception.** Only after the Level 1 final
benchmark is frozen: a camera-calibrated RGB-D/point-cloud pipeline from consistent simulator state
and a compact point-cloud-conditioned diffusion policy (DP3-inspired, not automatically an exact
reproduction), compared with a matching RGB baseline under identical
observations/training/evaluation states, studying spatial generalization. Marked
experimental/optional if dataset reconstruction, sensors, licenses or budget fail the feasibility
gate; not finishing it does not invalidate a completed Level 1 release.

## 1.3 Tasks, prioritized and accurately named
| Priority | Environment/dataset | Real task | Role | Requirement |
|---|---|---|---|---|
| P0 | `Lift` (robosuite / robomimic PH) | Articulated robot grasps and lifts a cube | Primary controlled benchmark | **Mandatory** |
| P1 | `Can` (robosuite / robomimic PH) | Grasp and place a can into a designated bin | Transfer of benchmark method to harder placement | **Mandatory if feasibility gate passes; otherwise explicitly scoped core result + blocked extension** |
| P2 | `Square` (PH), or compatible `StackCube` from **ManiSkill** only as an explicitly separate dataset/framework | Precision insertion or stacking, respectively | Stretch difficulty | Optional; Square is not called "Stack" |
| P3 | RGB-based policy for `Lift`, then `Can` if feasible | Same physics tasks with camera observations | Level 1 visual-policy extension | Conditional on observation extraction proof + budget |
| P4 | Depth/point clouds for a matched manipulation task | Explicit geometry-aware control | Level 2 | Advanced optional gate |

robomimic/robosuite datasets are never mixed with ManiSkill physics, controllers, rewards or action
conventions. ManiSkill is a fallback only if the primary stack fails Session 0; replacing the stack
requires a recorded approval of a protocol revision, with separately validated replay and
evaluation. If a compatible `Can` cannot be obtained, no pick-and-place dataset is invented and
`Lift` is not passed off as `Can`.

## 1.4 Minimum finished product
(a) reproducible data ingestion and simulation; (b) ≥2 learned baselines and an actual trained
conditional diffusion policy; (c) repeated-seed closed-loop comparisons on Lift, with a second task
if verified; (d) paired held-out evaluation and transparent CIs/failures; (e) budget-guarded remote
GPU training and checkpointing; (f) at least one authentic full-3D interactive rollout viewer with
verified kinematic fidelity; (g) real denoising-stage visualizations; (h) a paper-style research
report; (i) clean CI, documentation and downloadable or reproducible visual samples; (j) correct
sole project authorship and scientific attribution. The project is entirely simulated: no
physical-robot deployment is implied, 3D physics simulation is distinguished from real hardware, and
state conditioning is distinguished from inferring arbitrary 3D geometry from pixels.

## 1.5 Explicit exclusions
Not a general-purpose robotics foundation model, language-conditioned robot control, multi-robot
planning, real-world autonomous actuation, reinforcement learning from scratch, an LLM wrapper, a
full simulator built in WebGL, or an unsupervised robotics benchmark. No authentication, payments,
databases, model-serving endpoints or agent dashboards in the showcase unless essential and
approved. A static hosted site built from compressed verified artifacts is preferred to a live
inference backend.

# 2. Reference stack, version control and dataset proof

## 2.1 Preferred stack (candidate, not an assumption of compatibility)
| Component | Requirement |
|---|---|
| MuJoCo | Official DeepMind Python bindings: articulated physics, contacts, kinematic state, offscreen frames |
| robosuite | Robot/environment/tasks at a frozen compatible revision; robomimic's dataset docs recommend the `v1.5.1` branch with compatible dataset conversions. The selected robomimic version's actual recommendations apply, not outdated `mujoco-py` tutorials |
| robomimic (0.5-series candidate) | HDF5 demonstration conventions, BC/BC-RNN templates, diffusion-policy reference, action handling, checkpointing, environment wrappers; pinned to an exact tested release/commit, never floating `main` |
| PyTorch | `torch`, CUDA, NumPy, `h5py`, `gymnasium` and other dependencies pinned after a real smoke test |
| Modal | Pay-as-you-go remote PyTorch training (CPU where sufficient, GPU when justified); isolated training resources; persistent checkpoint storage |
| Experiment tracking | Immutable JSON manifests + Parquet/CSV + optional local MLflow/TensorBoard; no paid SaaS account required |
| Frontend | React + TypeScript + Vite + React Three Fiber / three.js; explanatory narrative, accessible interactions |
| Paper | LaTeX + BibTeX; tables and figures generated from frozen results; PDF built if a LaTeX toolchain is installed |

Exact Python/robotics versions are not pinned here: Session 0 assembles a verified compatible
lockfile/container. Headless EGL/OSMesa renderer settings and GPU drivers are documented separately
for laptop/macOS and Linux/Modal; a working macOS simulator does not imply a working Linux cloud EGL
setup.

## 2.2 Official dataset families and specific starting plan
The data are robomimic's **v0.1 dataset family as repackaged for newer robosuite versions** (not
necessarily the original 2021 simulator binaries), starting with **proficient-human (PH),
low-dimensional** `Lift` and `Can`. Candidate downloader commands, run from a pinned robomimic
checkout:

```bash
python robomimic/scripts/download_datasets.py \
  --tasks lift can --dataset_types ph --hdf5_types low_dim --dry_run
python robomimic/scripts/download_datasets.py \
  --tasks lift can --dataset_types ph --hdf5_types low_dim \
  --download_dir "$ROBOFORGE_DATA_ROOT"
```

These are documentation forms, not evidence of a download. Verified in practice: syntax, download
links, HDF5 files, SHA-256, task names, episode counts, action shapes, observation keys, controller
config, robot model, simulation metadata, provenance, usage terms and expected environment build.
The repackaged data may not exactly reproduce the original CoRL 2021 results; the paper says so.

Raw datasets are separately available for extracting RGB, depth and additional camera views with
robomimic's state-to-observations tooling. Raw `states`, `model_file`, `env_args` and tools such as
`dataset_states_to_obs.py` are verified before image or point-cloud experiments are scheduled; a
low-dimensional file is not assumed sufficient. Camera observations are generated from the matching
saved simulator states.

## 2.3 Feasibility proof: exact acceptance test
Session 0 generates `reports/feasibility/STACK_REPORT.md` and command logs showing:

1. An actual environment install on at least one viable machine/container with frozen code
   revisions; robomimic, robosuite, MuJoCo Python bindings and Python/CUDA compatibility checked.
2. A real PH **Lift low-dimensional** HDF5 file downloaded: keys, dimensions, length distribution,
   useful metadata, hash, license/terms, source URL. **Can** availability probed and reported
   separately.
3. A simulator `reset` and `step` for Lift with 3D `qpos`/robot-body poses; no image-only fake
   wrapper.
4. **A short recorded demonstration replayed** from the saved model XML, initial state and dataset
   actions; terminal positions, task outcome and returned state compared; replay error/tolerances
   documented. A controller mismatch is a blocker.
5. A headless RGB render of the *same* replayed trajectory, with exact camera name and rendering
   backend; if unavailable locally, a compatible Linux image on approved infrastructure or a CPU
   backend. Training feasibility is distinguished from render feasibility.
6. A small data loader producing (observation history, future action chunk) with shapes and values
   checked; a trivial one-batch policy forward pass.
7. A small number of **closed-loop** random-policy steps and a meaningful success/termination
   signal.
8. Provenance and redistribution review: dataset, example images, meshes, robot assets, framework
   code, reference weights.
9. Estimates of disk usage, training memory, rollout throughput and rendering cost, and whether a
   tiny Modal smoke test is needed; no paid job without a recorded approval.

**NO-GO:** if data do not download, simulator state does not load, controller/action dimensions
conflict or demonstration replay is unreliable, expensive model training does not start; the stack
is repaired/version-pinned, or an evidence-based alternate stack is adopted by a recorded approval
of a protocol revision.

## 2.4 Source registry — authoritative starting references
- robomimic docs, dataset v0.1 family:
  <https://robomimic.github.io/docs/datasets/robomimic_v0.1.html>
- robomimic repo and MIT code license: <https://github.com/ARISE-Initiative/robomimic> ;
  <https://github.com/ARISE-Initiative/robomimic/blob/master/LICENSE>
- robomimic diffusion-policy tutorial:
  <https://robomimic.github.io/docs/tutorials/training_diffusion_policy.html>
- robomimic state-to-observations extraction script:
  <https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/dataset_states_to_obs.py>
- robosuite repo and license: <https://github.com/ARISE-Initiative/robosuite> ;
  <https://github.com/ARISE-Initiative/robosuite/blob/master/LICENSE>
- MuJoCo docs: <https://mujoco.readthedocs.io/>
- original Diffusion Policy research: <https://diffusion-policy.cs.columbia.edu/>
- 3D Diffusion Policy / DP3 research: <https://3d-diffusion-policy.github.io/>
- ManiSkill as *conditional fallback*, not a mixed benchmark:
  <https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html>
- Modal GPU docs, Volumes and pricing: <https://modal.com/docs/guide/gpu> ;
  <https://modal.com/docs/guide/volumes> ; <https://modal.com/pricing>

Scholarly citations are never fabricated: BibTeX is retrieved from papers or official repositories,
names and venues are verified, and DOI/arXiv identifiers are included as applicable.

# 3. Research design: hypotheses, fairness and precommitment

## 3.1 Prespecified hypotheses (not assumed true)
- **H1 — closed-loop competence:** trained diffusion and BC policies improve true success over a
  non-learned random-action sanity baseline.
- **H2 — method comparison:** diffusion's closed-loop success may exceed deterministic BC on task
  distributions with multiple valid action sequences; paired initial states and uncertainty
  intervals; a null or negative finding is allowed.
- **H3 — temporal conditioning:** a recurrent policy and a diffusion policy with observation history
  may outperform one-frame deterministic BC; costs are measured; *history benefit* is separated from
  *generative architecture benefit*.
- **H4 — robustness:** policies may differ in sensitivity to object placement, robot initial
  condition and visuomotor perturbation; claims apply only to perturbations implemented and tested.
- **H5 — optional point-cloud extension:** with identical episodes and held-out initial states,
  explicit 3D perception may improve positional generalization over RGB-conditioned policies;
  meaningful only with a matched visual-data protocol.

Core comparisons use the exact same state observation set and action control. RGB and point-cloud
results are labelled by modality, data and compute budget, and are never said to beat a
privileged-state policy in a "fair perception comparison" without acknowledging the state privilege.

## 3.2 Experimental hierarchy and freeze rules
1. **Engineering controls:** random action, demonstration replay, short-horizon overfit on a few
   demonstrations, validation curves, reward/success consistency; *not* research performance claims.
2. **Level 1 core development:** models/settings selected using training and validation data +
   validation rollout starts only; one canonical primary protocol per task, with success-based
   checkpoint selection where feasible.
3. **Level 1 final held-out test:** only after `docs/FINAL_TEST_FREEZE.md` is committed with dataset
   IDs, algorithm configurations, model selection rule, fixed test initial-state schedule, metrics,
   comparisons and analysis plan. The locked protocol runs once per designated checkpoint (repeated
   stochastic rollouts allowed if prescribed). Any revisit of the test is labelled exploratory or a
   separate replication; nothing is silently tuned to the test.
4. **Level 1 robustness:** independent, explicitly defined perturbation sets; a distinct validation
   regime selects perturbation magnitude if needed; ranges frozen before the main robustness
   results.
5. **Level 1 RGB extension:** an explicitly segregated observation-modality study, not a revision of
   core test numbers.
6. **Level 2 point-cloud:** own preregistration, budget sign-off and frozen test schedule; an early
   unsuccessful point-cloud experiment is never relabelled a successful "3D perception" result.

## 3.3 Dataset splitting and leakage prevention
- Per task, whole episode IDs (and operator identity, if available) are deterministically hashed to
  train/validation/optional held-out demonstration subsets with a seed locked in config; all IDs are
  documented. 80/10/10 is a typical exploratory division **if episode count and dataset conventions
  justify it**, not the original robomimic benchmark's required split.
- Windows never cross an episode boundary (observation histories and future action targets are built
  within one trajectory); boundary behavior is defined (repeat edge observation versus drop window)
  and padding is masked from losses.
- State/action normalization is fitted on training episodes only and saved as reversible transforms;
  inference uses the exact transform and correct clipping for the simulator's actual action space.
- If demonstrator identities exist, within-demonstrator and held-out-demonstrator claims are
  reported distinctly; operator metadata is never invented.
- Rollout evaluation seeds/initial states are defined and frozen separately from the demonstration
  split. Replaying ground-truth demonstration actions never counts as autonomous success.
- Derived RGB/depth/point-cloud representations stay aligned with the training dataset via immutable
  `task`, `episode_id`, `step_index`, original state/hash and camera metadata; no resimulation with
  different control physics.

## 3.4 Minimum statistically meaningful evaluation
**Default when compute permits:** per task, **3 independent model seeds per primary algorithm**,
each evaluated on the **same 50 held-out initial-state seeds**: 150 policy-seed × initial-state
trials per model/task. Rollout cost is benchmarked first; a smaller precommitted plan needs a
recorded approval. Statistical power is not claimed merely because 50 starts were used; repeats are
added when uncertainty is large and cost permits, never only favorable ones.

- **Primary metric:** fraction of episodes achieving the simulator's task success criterion before
  the defined horizon, with numerator/denominator, confidence interval, model-seed spread and paired
  difference against baseline.
- **Secondary:** steps to first success, success persistence if relevant, normalized
  task-progress/goal-distance only if the task defines it legitimately, collision/contact failure
  categories when observable, trajectory/path length, action smoothness/jerk (with time-step units),
  action saturation, control frequency, inference-time latency, training-time GPU-hours, approximate
  actual dollars.
- **Diffusion-specific:** samples per decision, denoising iterations, plan diversity under repeated
  stochastic samples, execution horizon, rejection/clipping frequency; actual noise/denoising
  tensors and conditional state are recorded.
- **Uncertainty:** bootstrap by **held-out initial state** with policy training seed as a separate
  variation source (e.g. hierarchical bootstrap, or per-seed summaries plus paired state bootstrap);
  fixed bootstrap seed; method disclosed. Time steps of one rollout are not independent
  observations.
- **Sampling unit:** repeated deterministic evaluations on identical initial conditions are
  duplicates and do not increase sample size; for diffusion, policy noise seeds are prescribed
  independently and the sampling unit is documented.
- **Denominators:** per-task and per-model denominators, all failed or timed-out trials, NaNs,
  exception counts and early simulator resets are always shown.

**Mandatory truthfulness:** statistical significance alone does not establish "better"; absolute
changes, uncertainty, practical relevance, speed/cost tradeoffs and limits on interpretation are
reported. Benchmark-paper results are never reported as this project's own.

## 3.5 Evaluation starts, domain shift and distributions
- A versioned `eval_states.jsonl` or equivalent holds stable initial simulator configurations and
  state identifiers; all models reset from the same conditions, avoiding unintended different
  randomization sequences.
- Two fresh simulator processes reloading one state produce compatible robot/object configurations
  and near-identical short deterministic rollouts under identical controls; maximum discrepancies
  are recorded.
- **In-distribution** starts come from the benchmark's standard reset distribution;
  **out-of-distribution** variants have explicit bounds (e.g. object initial XYZ position shifts,
  yaw differences, gripper start pose shifts), **only if physically realizable** and replayable; no
  unsupported camera changes that invalidate observation labels.
- Spatial OOD is not claimed if only the random seed changed within the training distribution;
  appearance/camera OOD is kept separate from geometric OOD.
- A small hand-audited set of failures and successes for visuals is chosen by **predefined selection
  criteria**, not by appearance.

# 4. Simulation, data and policy interfaces

## 4.1 Canonical environment wrapper contract
`RoboForgeEnv` is a typed, documented adapter over the pinned simulator, with methods analogous to:

```python
reset(initial_state_or_seed) -> Observation, EpisodeMetadata
step(action: np.ndarray) -> Observation, reward, terminated, truncated, info
serialize_initial_state() -> Mapping
get_robot_kinematics() -> JointStateAndTransforms
get_scene_kinematics() -> ObjectTransforms
render(camera_id: str, width: int, height: int) -> np.ndarray
get_success() -> bool
close() -> None
```

Adapters are built; site-packages are **not** modified. Deterministic resets use a state snapshot
with model XML, episode metadata, MuJoCo generalized positions/velocities, mocap/actuator state when
needed, control setup and task randomization, never a global random seed alone. Semantic constants
come from **actual metadata**: robot name, joint names, link hierarchy, gripper actuator name,
control frequency `Hz`, action min/max and dimension, control mode, camera
names/intrinsics/extrinsics, task success function, observation keys and privileged features. Shape
contracts are tested at every layer.

## 4.2 Observation dictionary and modality contracts
```text
observation:
  robot:         joint_pos [n_joints] from active robot; joint_vel (if provided and permitted);
                 end_effector_pose (3D position + documented rotation encoding);
                 gripper_state (actual simulator-provided dimensionality)
  object_state:  position / orientation / target_state / optional contacts
                 (state-based Level 1 only; privileged if exact simulator pose)
  rgb:           camera_id -> uint8 [H,W,3]                      (optional Level 1 image policy)
  depth:         camera_id -> calibrated metric or documented depth units         (Level 2 only)
  point_cloud:   xyz [N,3], optional rgb/segmentation, valid mask, coordinate frame (Level 2 only)
```

Keys are explicit and declared in `ObservationSpec` and the run manifest. **RGB-only** means robot
proprioception + camera image and no object ground-truth pose: `object_state` must not leak through
normalization, timestep metadata, labels or model input plumbing. A goal specification that is
objectively part of the task definition may be used and is documented. Per algorithm, input tensor
shapes, dynamic ranges, mask/padding, history ordering and frame transformations are logged; tests
check finite batches and frame/action alignment.

## 4.3 Action conventions, horizons and control loop
Actions may involve end-effector Cartesian deltas, orientation commands, joint controls and gripper
control. **A fixed 7D vector is not asserted** without dataset validation (3 positional + 3
rotational + 1 gripper would give 7D, but the actual controller can differ). One canonical action
representation per benchmark, with reversible mappings:

- `dataset_action -> normalized_model_action` for loss and diffusion sampling;
- `normalized_model_action -> clipped_env_action` for the simulator step;
- optional `env_action -> readable_world_motion` for visualization; Cartesian motion is never
  inferred by plotting raw joint commands.

Target defaults: **observation horizon** `T_o=2`, **prediction horizon** `T_p=16`, **execution
horizon** `T_a=8` (common Diffusion Policy configuration), each verified against demonstration
length, controller rate, robot response and memory budget, frozen on validation and never optimized
on test outcomes. The action chunk is recomputed in receding-horizon fashion after executing `T_a`
actions; a whole predicted plan is never played without sensing again. Guarded against: wrong axis
order, reward misalignment, off-by-one action targets, action-limit violation, gripper sign
inversion, quaternion antipodes, angle discontinuities, inconsistent frame definitions and
inconsistent task-specific action spaces.

## 4.4 Dataset schema and run provenance
Third-party HDF5 datasets live under ignored `data/raw/`. Immutable task manifests record
`dataset_id`, original download URL, acquisition timestamp, SHA-256, frame/control convention,
original episode IDs, per-episode length, operator/source if available, train/val split and
transformation version. Derived data live under ignored `data/processed/` with separate dataset
hashes and versioned converter code.

A shape-validating `SequenceDataset` outputs `obs_history[T_o,...]`, `future_action[T_p,A]`,
`valid_mask[T_p]`, `episode_id`, `start_t`, plus modality keys. Tests cover
first/middle/end-of-episode windows, zero-length/short demonstrations, consistent
padding/normalization, and future leakage (perturbing future frames leaves current observation
features unchanged). If the dataset includes `next_obs`, policy inputs come from `obs`, not future
labels.

## 4.5 Real simulator rollouts and artifact export
Each held-out rollout saves **numerical ground truth from the physics engine**, not just a video:

```text
run_id / model_id / checkpoint_sha256 / task / policy_seed / init_state_id
simulator_build / control_mode / control_dt / episode_limit
step_index / timestamp_seconds / reward / success / terminated / truncated
joint_names + joint_angles + joint_velocities
link_world_transforms (position + quaternion in documented frame)
gripper/link transforms and object_world_transforms
raw_policy_action / executed_env_action / action_clip_events
policy_decision_indices / plan_id / planning_latency_ms
selected_camera_frames or reference to compressed MP4/WebP
optional actual diffusion snapshots with scheduler timestep
```

Versioned gzipped JSON, Arrow/Parquet and compressed video with a machine-readable index are
preferred to large uncompressed JSON. The robot URDF/MJCF asset hash and joint hierarchy are
recorded so a web viewer can reconstruct joint rotations and object positions. Frame-conversion
tests cover MuJoCo coordinates to three.js; the 3D reconstruction is verified against an actual
MuJoCo reference render at 3+ timestamps with a stated allowable error. Artifact contract
(placeholders are schema documentation; produced artifacts are valid JSON with real values and
hashes):

```json
{"schema_version": "2.0", "project": "RoboForge", "task": "lift",
 "model_id": "diffusion_state_seed0",
 "checkpoint_sha256": "<actual sha256>", "dataset_sha256": "<actual sha256>",
 "environment": {"engine": "mujoco", "robosuite_revision": "<pin>", "dt": "<real>"},
 "initial_state_id": "lift_test_0042", "policy_seed": 0, "outcome": {"success": false, "steps": 0},
 "observations": "<path to verified observations>",
 "poses": "<path to per-frame robot/object transforms>",
 "actions": "<path to predicted and executed actions>",
 "denoising": "<path to real model intermediate arrays or null>",
 "video": "<path to actual simulator recording or null>"}
```

# 5. Learning system — required models, losses and controls

## 5.1 Shared training and inference framework
A common `Policy` interface provides `train_batch`, `act`, `reset_episode`, `save_checkpoint`,
`load_checkpoint` and `export_diagnostics`. Observation preprocessing, normalizers, action bounds,
episode buffering, checkpoint/manifest code, metrics and evaluator are shared. All models accept
`ObservationSpec` and `ActionSpec`; no hard-coded PushT dimensions.

Logged: train/validation losses, optimizer settings, gradient norms, seeds, parameter counts,
training samples/steps, effective batch sizes, inference latency, model selection. Checkpoints hold
full model/optimizer/scheduler state, normalization stats, config, dataset/split hash, global step,
RNG states if practical, code revision and training device; strict reload, resume and forward
equivalence are tested. Checkpoints are not committed to Git unless size, license and a recorded
approval allow; download instructions and SHA-256 are provided. The pipeline is first verified with
1 tiny CPU batch and a short deliberate overfit of 1–4 demonstration episodes: a **unit/integration
sanity test**, not benchmark success.

## 5.2 Baseline A: deterministic behavior cloning (mandatory)
- **State Level 1:** MLP with 2–4 residual or standard blocks, layer norm as useful, configurable
  width (e.g. 256), mapping current/short-history state to a normalized action.
- Optional lightweight RGB encoder for the image-extension study, with an *identical image backbone
  across compared policy families when practical*.
- Masked action MSE or Huber, chosen on validation, with each dimension's meaning reported. An
  action **chunk** may be predicted to match the diffusion horizon; a strictly matched one-step
  ablation is included if chunking is tested.
- Actual closed-loop success is measured; no-op/random sanity baselines are not primary learned
  comparators.

## 5.3 Baseline B: recurrent behavior cloning (mandatory)
- GRU/LSTM over observation histories; hidden state reset at episode boundaries; truncation/padding
  handled.
- Current action or action chunk, with a declared setup and a comparison controlling for output
  horizon.
- Training data and approximate parameter count matched as feasible; material capacity differences
  reported.
- Tests show rolling observations influence results and hidden state does not carry between
  evaluation episodes.

## 5.4 Baseline C: stochastic mixture policy (optional but desirable)
Gaussian mixture policy (mixture-density network) or robomimic's compatible GMM-BC variant,
parameterizing K component means, scales and weights, with correct negative log likelihood,
bounded/executed actions and no loss explosion from invalid scales. It isolates whether diffusion's
advantages reflect **stochastic action modeling** or other architecture features; included only if
budget permits a statistically credible comparison.

## 5.5 Main algorithm: conditional diffusion action policy (mandatory)
A **conditional denoising diffusion model for continuous action sequences**, implemented or
carefully adapted; recommended start: a **1D temporal U-Net** conditioned on a history of
robot/object observations (later RGB features). Scheduler and model semantics are tested and
documented against the official Diffusion Policy and the robomimic 0.5 reference implementation.
Minimum training sequence:

1. Encode `T_o` permitted observations into a conditioning representation `c`.
2. Take the normalized target action sequence `A_0 ∈ R^{T_p × action_dim}`.
3. Sample diffusion time `t` and Gaussian noise `ε`; form `A_t = sqrt(ᾱ_t) A_0 + sqrt(1−ᾱ_t) ε` for
   an appropriate DDPM-style schedule.
4. Predict `ε̂ = f_θ(A_t, t, c)` (or a documented alternative target such as `v`) with a masked
   loss; the exact prediction target is recorded in the config.
5. At inference, sample a noisy action chunk with the declared noise schedule, denoise for the
   specified inference steps, invert normalization, clamp to action constraints and execute only the
   next `T_a` commands before replanning.

A network outputting a single deterministic action after one forward pass **is not a diffusion
policy**. Tests cover noise-scheduler equations, timestep embedding, reverse denoising shapes,
condition leakage, EMA behavior if used, temporal padding masks, observed-state consistency, dynamic
action ranges and deterministic seeded sampling.

**Validation reference:** shapes and a synthetic training case are compared against robomimic's
official Diffusion Policy template; externally adapted components are marked and licensed; a wrapper
around off-the-shelf scripts is not presented as a new diffusion algorithm.

Initial hyperparameter search *bounds, not fixed values*: U-Net width `64–256`, 2–3 down/up stages,
schedule `DDPM/DDIM`, 10–100 inference denoising steps, history 1–3, prediction horizon 8–24,
execution horizon 1–8, AdamW, LR roughly `1e-4` with a logged scheduler, EMA optional. The
validation search is deliberately small and is judged on latency and closed-loop success, not
training action-loss alone.

**Real intermediate samples:** an instrumented sampler saves `A_{t_k}` at selected scheduler
timesteps **before any conversion into world-space trajectories**, with the encoded condition ID,
scheduler coefficients/timestep, noise seed, normalized action tensors and denormalized candidate
actions. Noisy action sequences are labelled *proposals*, never shown as a path the robot travelled;
path overlays integrate candidate commands through a clearly identified kinematic approximation or
short clone simulation and are labelled accordingly.

## 5.6 Optional advanced architecture: Diffusion Transformer (DiT-style)
A conditional Transformer denoiser over action tokens, with diffusion-timestep embeddings and
observation cross-attention/FiLM conditioning, is a **stretch**, not needed for the Level 1 release:
implemented only after core diffusion is competitive/reproducible and its budget is approved,
reporting parameter count, matched training budget and inference latency. Attention is not assumed
to beat a U-Net on small demonstration datasets. If omitted, the reason is documented and no DiT
evaluation is claimed.

## 5.7 Architecture parity and counterfactual checks
- All families run under the same train/val episode IDs, normalization, control horizon, reward and
  success criteria, simulator version and held-out initial states.
- Compute budgets and the number of hyperparameter trials are tracked; at minimum a tuned
  capacity-matched BC baseline is offered (a heavily tuned diffusion model against an untuned tiny
  MLP proves nothing).
- At least 2 diagnostic ablations follow a full core comparison, from: observation history 1 vs 2+,
  prediction/execution horizon, diffusion denoising steps, action normalization method, EMA or no
  EMA, state-proprioception vs privileged object state. **The core comparison is frozen first.**
- Input perturbation and sensitivity are tested: out-of-range observations, no-op action, maximum
  action clipping, NaN handling and robot joint-limit enforcement in the simulator.

# 6. Training, Modal GPU jobs, resource governance

## 6.1 Local-first / remote-compute workflow
Laptop (implementation, unit tests, tiny CPU smoke) → validated config + pinned dataset manifest +
code revision, under an approved budget, cost estimate and timeouts → Modal GPU (PyTorch training
only) or authorized cloud CPU, producing checkpoints, metrics and resource logs → persistent
versioned artifact storage (Modal Volume / approved bucket), retrieved as signed/hashed artifacts →
local analysis and closed-loop simulator evaluation (local CPU or authorized cloud hardware) →
freeze benchmark → export rollout truth → static 3D Observatory.

**GPU accelerates compatible PyTorch training, not MuJoCo CPU physics.** Throughput is benchmarked
for both; the training resource choice is separate from the rollout/rendering choice, for which
cloud CPU parallelization may be the better allocation.

## 6.2 Modal integration implementation
- `infra/modal_train.py` (or a canonical Python module): pinned container image, CUDA/PyTorch
  compatibility, exact mounts/secrets, specified GPU (initially T4/L4/A10 as availability and memory
  dictate), CPU/memory, timeout, concurrency, checkpoint volume.
- `modal run` or an equivalent verified entrypoint accepts `--config`, `--budget-approved`,
  `--dry-run` or equivalent explicit confirmation; no remote allocation during import, linting,
  tests or page build.
- **Reusable immutable datasets in an authorized persistent Volume/bucket** rather than
  re-downloading multi-gigabyte data per run; dataset hashes and storage cost recorded.
- Checkpoints saved periodically and at graceful interruption; volume synchronized; downloads
  validated by hash; a checkpoint verified to load locally.
- **Idempotent launch** keyed by run ID that rejects duplicate jobs; explicit cancellation command;
  container ID and start/stop times logged; no retry loop without max attempts and an explicit
  incremental budget.
- Dry-run cost estimator
  `python -m roboforge.compute.estimate --config ... --gpu ... --max-minutes ...` covering GPU,
  CPU/memory, storage and plausible idle/prep/render time; estimates are not hard billing caps.
- Modal may require a valid payment method even with monthly credits; posted prices and free-credit
  conditions are rechecked at launch (Starter advertised $30/month credits, which do **not**
  guarantee spending stays under $30). Each billable run requires a recorded approval.
- No API keys, GitHub tokens or cloud secrets in committed configs or web build assets; Modal
  Secrets or environment variables with least privilege; redacted logs.

## 6.3 Budget gates
Paid compute requires an approved spending cap recorded as a decision (`docs/DECISIONS.md`). The
gates are default **proposals**, not authorized spending:

- **Gate A (smoke):** CPU/local exercises; one optional ≤15-minute GPU smoke on one model, approved
  separately.
- **Gate B (pilot):** one capped training seed for BC and diffusion on Lift, collecting wall
  time/epoch, GPU utilization, storage and evaluation throughput.
- **Gate C (main):** recorded approval of the proposed seed count × GPU hours × resource pricing,
  with a reserve for failed runs and simulator evaluation; nothing executes before acceptance.
- **Gate D (extensions):** separate approvals for `Can`, RGB visual policies, DiT, or point-cloud
  Level 2.

Jobs have a timeout, concurrency of one by default, manual launch flags and spend warnings; a
platform account/workspace budget is configured under a recorded approval where supported.
Client-side budgets/timeouts are not perfect financial hard stops. All-in spend is compared to
budget, and no further jobs launch once it is consumed.

## 6.4 Experiment tracking and artifact discipline
Every run has a stable ID and a manifest:

```text
run_id, UTC start/end, git SHA and dirty status
framework versions + container-image digest + GPU/CPU details
source dataset sha256 + derived dataset sha256 + split manifest
TaskSpec + ObservationSpec + ActionSpec and simulator/controller versions
model config + parameters + normalization checkpoint hash
train seed + validation seed set + exact replay seed schedule
optimizer/scheduler/EMA + data augmentations + max steps
validation metrics + checkpoint-selection rule + chosen checkpoint hash
GPU training seconds + CPU/sim seconds + measured/billed cost where known
recorded errors/restarts/aborted runs, including reasons
```

Immutable `runs/<id>/` directories are not reused for conflicting configs. A human-readable
experiment ledger (`docs/EXPERIMENT_LEDGER.md`) records hypotheses, outcomes, failures and why the
next experiment was selected. Results summaries are generated from machine-readable runs, never
entered by hand. All paper figures report `source_manifest` and `generated_at`.

## 6.5 Required CLI contract (illustrative names; the real commands must work)
```bash
make setup                  # lockfile-based install or documented container
make test                   # unit tests, no paid GPU or dataset download by default
make smoke-sim              # actual robot reset/step and headless frame
make download TASK=lift     # verified official dataset fetch with provenance
make verify-data            # HDF5 manifest, replay and split contract
make train-smoke            # local CPU miniature overfit
make train MODEL=bc TASK=lift CONFIG=...
make train-modal MODEL=diffusion TASK=lift CONFIG=... # approval gate
make eval TASK=lift EVAL_SET=validation MODEL=...   # closed-loop
make freeze                 # validate + commit locked protocol
make eval-final             # explicit frozen-test access guard
make export-observatory     # real trajectories → immutable compressed web assets
make paper                  # tables/figures/report from actual outputs
make web-test
make release-check
```

`make` targets are never disconnected echo statements; where a platform command differs, a
documented wrapper is added rather than inventing CLI options.

# 7. Level 1 expansions and Level 2 point-cloud research

## 7.1 Secondary 3D manipulation task: Can
After **Lift** models and validation have passed, **Can** runs with compatible PH data and
environment, identical algorithm families, predeclared model budgets and independent test initial
states. Can is pick-and-place (grasp a can and put it in a bin), not a generic "place any shape
anywhere" benchmark, with its own success condition and possibly a different controller or action
representation, checked and configured per task. Saved demonstration frames are not rollouts. A
headline metric combining Lift and Can is permitted **only if the aggregation rule was frozen before
test evaluation**; task-by-task results are preferred. If Can is unavailable, or a model fails to
learn it despite proper training, the blocked task or honest zero/low success is reported, not
deleted.

## 7.2 Level 1 RGB-conditioned policy study
The RGB extension tests whether policies can act from **camera pixels plus robot proprioception**,
not exact object-pose privilege. Once raw simulator snapshots and cameras are reproducibly
replayable:

1. **Aligned RGB frames** are generated from the exact corresponding demonstration states at 84×84
   or 128×128 (chosen via a throughput/accuracy pilot); camera intrinsics/extrinsics, rendering
   backend, model XML, frame hashes and episode/time indices are saved.
2. Image/state alignment is verified by manual visual inspection of a fixed random sample and a code
   test comparing projected object position with segmentation/visible location where available.
3. Visual BC and visual diffusion share identical frozen RGB encoder initialization (e.g. a compact
   ResNet), augmentations, preprocessing and proprioceptive features; whether the encoder is trained
   from scratch, pretrained and frozen, or fine-tuned is recorded.
4. A **held-out RGB test initial-state suite** with identical simulator/controller/settings.
   Architectural claims compare visual-only with visual-only; state-based Level 1 remains an
   **oracle-style comparator** when it uses privileged object state.
5. View/lighting changes are added only if the renderer and training inputs make them meaningful; no
   camera-robustness claim if the model always saw the same camera.
6. If dataset extraction is too expensive or incompatible, an honest feasibility note is published;
   untrained RGB policy renders are never shown as learned visual control.

## 7.3 Level 2 gate: explicit 3D geometry, not just a prettier 3D viewer
**Level 2 begins only once Level 1 core results are finalized and traceable.**
`docs/LEVEL2_GO_NO_GO.md` contains: (a) the precise hypothesis and baseline; (b) the approved
compute/dataset budget; (c) reproducible depth/point-cloud extraction from corresponding
demonstration states; (d) verified camera projection and coordinate frames; (e) the comparison
protocol; (f) an explicit fallback if it fails. If any component is absent, it records
`NOT EXECUTED` and the project continues to completion.

A **DP3-inspired** point-cloud-conditioned policy is implemented **only if feasible**; official DP3
reproduction is claimed only if its published implementation/protocol is followed and its
constraints met. Possible architecture: RGB-D camera / simulator depth (calibrated projection, mask,
depth units, camera extrinsics) → 3D point cloud XYZ in an explicitly logged world or camera frame
(workspace crop, N points sampled reproducibly, optional semantic mask) → compact point encoder,
PointNet-style or published DP3-style (aggregate features + robot proprioception) → conditional
diffusion U-Net, same policy backbone where practical (action chunks, identical control/environment
as the RGB baseline) → closed-loop 3D manipulation.

- **Ground-truth leakage:** a policy using object segmentation masks or world-frame coordinates
  derived from simulator labels is a privileged-geometry experiment; a pure geometric-vision
  experiment obtains points from sensor-like depth and calibration alone. Explicit robot
  segmentation may be a separate, labelled ablation. Privileged GT object masks are never compared
  with raw RGB as a vision-method superiority finding.
- **Calibrated depth conversion:** image width/height, perspective versus orthographic projection,
  near/far plane, depth-buffer normalization and physical units, camera-pose-to-world transform,
  coordinate handedness, UV convention and invalid/background pixels are verified. A numerical test
  projects a known 3D simulator point into image pixels and back within tolerance; an overlay image
  verifies point-cloud alignment with actual robot geometry.
- **Experiment:** at minimum, point-cloud diffusion against an RGB diffusion policy with identical
  demonstrations, robot proprioception, held-out starts, action normalization, horizon,
  optimizer/training budget and observation sampling rate; in-distribution and precommitted
  object-placement changes are compared; camera-pose perturbations are optional, distinguishing
  image from world-coordinate point-cloud invariance; point-cloud sample size and throughput are
  logged.
- If visual extraction or GPU memory fails, Level 2 terminates without rewriting earlier results;
  the feasibility lessons, tests performed and next technically sound step are documented. The
  project is complete without Level 2.

## 7.4 Other stretch experiments, only after the main result
- **DiT denoiser** versus U-Net on a single task at comparable capacity/compute.
- **Action mode visualization:** multiple diffusion chunks sampled from the *same* fixed condition,
  with spread quantified and resulting paths shown in state space; noise alone is not multimodality.
- **Dataset-quality study:** PH vs properly labelled MH demonstrations on one task, with no model
  unfairly receiving more/cleaner data.
- **More precise assembly:** `Square` (precision nut insertion), only if the compatible dataset and
  simulator pass the same Session 0-style contract.
- **ManiSkill StackCube:** an independent task and simulator with separate protocol and datasets; no
  pooled success table with robosuite.

# 8. RoboForge Observatory — visual storytelling requirements

## 8.1 Visual identity and communication objective
The Observatory is **a research-grade 3D robotics laboratory presented with editorial quality**, not
a generic dark dashboard or a static video wall. **30-second test:** within 30 seconds a visitor
understands (1) what task the arm is doing, (2) which policy controls it, (3) whether it succeeds,
and (4) what the research found. Aesthetic: distinct and restrained (graphite/charcoal surfaces,
warm ivory typography, one vivid accent color for learned policies, one distinct error color, crisp
industrial linework, restrained cinematography), with accessible text contrast, reduced-motion
options, keyboard navigation and mobile fallbacks. Visuals are **tethered to actual experimental
evidence**: a stylized 3D robot that is not a real rollout is labelled *illustrative*, and no
decorative animation is presented as a trained policy's achievement.

## 8.2 Site map and narrative arc
- **A. Hero — "Learning to Move."** Cinematic actual recorded Lift rollout of a trained policy with
  a subtle end-effector trail, annotated
  `MuJoCo simulation | trained checkpoint: [hash] | task: Lift | outcome: success/failure`;
  immediate links `Open the Lab` and `Read the Research`.
- **B. Why Diffusion for Actions?** Deterministic BC's single prediction versus diffusion's
  distribution over future action chunks; explanatory schematic distinguished from recorded
  denoising tensors; noise→action animation uses **real stored intermediate samples** when
  available.
- **C. The Robotic Lab.** Primary immersive interactive 3D viewer: task selector (`Lift`, `Can`;
  optional `Square`), policy selector, synchronized playback, scrubber, scene labels, optional
  model-controlled data channels, camera orbit/zoom/reset presets.
- **D. Head-to-head Comparison.** Two synchronized, seed-matched robot simulations, same task and
  initial physical state, driven by **two separately recorded closed-loop rollouts**; outcome,
  steps, control commands, trajectories and sampled confidence bands where scientifically
  appropriate.
- **E. Inside the Policy.** Interactive conditional action denoising: scheduler step slider, actual
  normalized action chunks and denormalized control values, conditional observation context,
  time-series chart of each control dimension. A candidate end-effector path only if generated by an
  explicitly verified integration/re-simulation method, never by treating raw action vectors as XYZ.
- **F. Research Findings.** Real success rates with denominator and confidence interval, paired
  differences, model-seed scatter, latency versus success, failure examples, ablations;
  zero/negative results presented honestly; persistent links to the actual experiment manifests.
- **G. Beyond Pixels (only if completed).** RGB and point-cloud panels with calibrated point-cloud
  reconstruction and real Level 2 results. If Level 2 was not executed, no results widget; a clearly
  marked *future research* explanation without fake point clouds is allowed.
- **H. Methodology & Credits.** Concise experimental design, dataset links, simulator versions,
  code/paper links, genuine citations; sole RoboForge author credit **Zackary Holt**, with separate
  acknowledgment of third-party researchers/resources.

## 8.3 3D viewer: technical acceptance criteria
React Three Fiber / three.js with **verified articulated kinematic playback**: GLTF robot models
(where licensing permits) or an accurate link hierarchy built from the robot's MJCF geometry, never
a single sliding mesh impersonating a 7-DoF robot. Supported: per-joint angles/poses, gripper
aperture, object transforms, camera viewpoints, trajectory path trails, controlled render loop,
stable time interpolation. If exact licensed assets cannot be distributed, the fallback is **the
original simulator video plus a clearly approximate 3D schematic**. **Coordinate correctness:**
MuJoCo/world axes and units, quaternion convention (scalar-first versus scalar-last), handedness and
transforms to three.js are declared; object contact/grasp is verified against simulator reference
frames; tests and a review montage from multiple timesteps catch wrong joint transforms. Viewer
controls:

- Task and trained-checkpoint selector, with source manifest reference.
- `Play / Pause / Frame-step / Restart / Speed ×0.25–×2 / Timeline seek`.
- Orbit camera, zoom, orthographic/isometric preset, top/side view and auto-follow toggle.
- Optional joint/link labels and Cartesian axes, object target overlay, gripper/end-effector trail.
- Model A/B comparison on the **same initial state** with synchronized world simulation time and
  clearly marked independent outcomes.
- Camera image inset from the **original simulation**, not a fabricated framebuffer; date/seed/build
  metadata.
- Tooltips distinguishing **recorded physical state** from **inference proposals**; visible
  `source: run_id` per clip.
- Performance: smooth interaction on a modern laptop at a reasonable asset size; graceful fallback
  to pre-rendered video on low-end/mobile/reduced-motion devices; no universal frame rate promised.

## 8.4 Diffusion Explorer — real denoising, not decorative confetti
Each exported denoising example includes the conditioning state, fixed policy/noise seed, scheduler
timesteps and the actual action tensor at those timesteps. Visitors scrub from Gaussian noise to the
final proposed action sequence and inspect per-control-dimension time plots, with the current
observation, action-chunk execution horizon and differences between candidate plans shown. A
"diversity" view samples the same context repeatedly and shows multiple final action chunks,
identifying genuinely distinct modes versus small variations. 3D paths come from forward-simulating
*candidate controls from a cloned initial simulator state* **or** a validated, labelled kinematics
approximation; the true executed path is a separate overlay; a warning states that not all proposals
were executed or are dynamically feasible.

## 8.5 Export format, integrity and performance
`python -m roboforge.viz.export --manifest ...` reads *frozen evaluation artifacts* and outputs
deterministic, compressed web data plus a `public/data/index.json` registry. Named examples follow
transparent selection rules (e.g. first success and first failure by sorted seed after predeclared
conditions); one borderline/interesting failure may be curated **only if explicitly labelled a case
study rather than representative**. Asset generation fails if model/checkpoint hash, run provenance
or simulator build metadata are missing.

- Compressed, lazily loaded web assets with video fallback; not 100s of MB of JSON by default.
- No redistribution of the raw dataset or unauthorized third-party meshes; no full trained weights
  in the browser unless specifically authorized and license-compliant.
- Video thumbnails/screenshots come from actual simulator rollouts; `docs/VISUAL_PROVENANCE.md`
  lists output, model run, seed, simulator source and transformation applied.
- Unit tests parse all exported JSON against TypeScript runtime schemas (e.g. Zod) and verify frame
  counts, monotonic timestamps, joint naming and no missing transforms.
- The web reads frozen summary JSON from the same pipeline as the LaTeX tables; values are never
  duplicated by hand.

## 8.6 Design acceptance check
Desktop and mobile views are tested: no clipping, scroll-trapping, broken keyboard controls,
unreadable labels, unresponsive sliders or unbounded GPU/WebGL consumption; `prefers-reduced-motion`
respected; generated descriptive alternative text for factual figures. A short high-quality
repository preview GIF/MP4 is built from real rollouts and checked to identify model and task
accurately; fake physics clips are never publication hero material.

# 9. Repository structure and software engineering conventions

The repository may be reorganized if justified in `docs/DECISIONS.md`, preserving the functional
boundaries and explicit tests.

```text
roboforge/
├── README.md, Makefile, .gitignore / .gitattributes, LICENSE, THIRD_PARTY_NOTICES.md
├── pyproject.toml / uv.lock                 # tested, pinned Python env
├── CITATION.cff                             # author Zackary Holt, external refs in paper
├── .env.example                             # never real secrets
├── .github/workflows/{python-tests,web-build,research-integrity}.yml
├── configs/simulator/{lift,can}.yaml, configs/data/{lift,can}.yaml
├── configs/models/{bc,bc_rnn,bc_gmm,diffusion,dit}.yaml
├── configs/experiments/{pilot,level1_core,level1_rgb,level2_pointcloud}.yaml
├── configs/eval/{validation,final_test,ood}.yaml
├── src/roboforge/
│   ├── data/{fetch,inspect,manifest,split,windows,normalization,rgb_extract,pcd_extract}.py
│   ├── sim/{env_adapter,state_replay,controller,kinematics,renderer,eval_states}.py
│   ├── models/{policy,encoders,bc,bc_rnn,bc_gmm,diffusion_unet,diffusion_transformer,
│   │           point_encoder}.py
│   ├── train/{train,loop,checkpoint,ema,metrics,seed,config}.py
│   ├── eval/{rollout,protocol,aggregation,statistics,robustness,freeze}.py
│   ├── viz/{export,artifact_schema,projection,video,figures}.py
│   └── compute/{estimate,cost_log}.py
├── infra/modal_train.py, infra/containers/  # pinned CPU and GPU environment definitions
├── scripts/                                 # stable reproducible CLIs
├── tests/{unit,integration,simulator,research_integrity,web_contract}/
├── data/{raw,processed}/                    # gitignored
├── data/splits/, data/manifests/            # small versioned manifests; source metadata, hashes
├── runs/                                    # heavy artifacts gitignored; summaries versioned
├── results/{protocol,frozen,tables,figures,selected_rollouts,provenance}/
├── website/package.json / lockfile, website/src/{app,components,scenes,hooks,lib,styles}/
├── website/public/data/                     # bounded, verified rollout bundles
├── website/public/media/                    # license-checked/actual render previews
├── paper/{main.tex,references.bib,sections/}
├── paper/generated/, paper/main.pdf         # generated with provenance; PDF when reproducible
├── reports/feasibility/
└── docs/{SPECIFICATION,STATUS,DECISIONS,KNOWN_ISSUES,EXPERIMENT_LEDGER,RESEARCH_PROTOCOL,
          FINAL_TEST_FREEZE,DATA_PROVENANCE,SIMULATOR_CONTRACT,DATA_DICTIONARY,REPRODUCIBILITY,
          LEVEL2_GO_NO_GO,VISUAL_PROVENANCE,RESULTS}.md
```

**Packaging/testing rules:** Python typing, ruff formatter/linter, pytest, lightweight configs via
Pydantic/typed dataclasses, stable seeds, error messages for missing dataset and credentials, no
hidden global mutable state, lockfile/dependency recording. CI runs small deterministic headless
tests (GPU/full-simulator slow tests marked separately) and **never** downloads huge datasets,
allocates paid compute, publishes research numbers or pushes code as a bot. No Git LFS for required
artifacts unless publicly reproducible and approved; checksum-verified releases or public object
storage instead.

**Test categories:**

1. Dataset HDF5 schema, episodes, split disjointness, sample windows, normalization inversion and no
   future leakage.
2. Simulator startup, controller dimension, action clipping, success detection, reset determinism
   and demonstration replay.
3. Model forward/backward, finite gradients, diffusion noise/scheduler equations, fixed-seed sample
   reproducibility, masked loss, strict checkpoint resume.
4. Evaluation fairness: paired state set, equal time horizons, no accidental privileged state, seed
   isolation, honest denominator, no silent failures.
5. Stats: paired bootstrap on known synthetic controls, reproducible confidence intervals,
   non-duplicated trials, result-table provenance.
6. Web artifacts: schema consistency, kinematic transforms, quaternion and coordinate conversion,
   camera projection, denoising tensor dimensions and source integrity.
7. Frontend builds, interactions, reduced-motion fallback; the 3D viewer fails gracefully when
   artifacts are absent.
8. Documentation/paper: every headline metric maps to a real run+checkpoint+evaluation manifest;
   external citations resolve and required license notices exist.

# 10. Project stages: deliverables and acceptance criteria

Stages run in order, Session 0 to Session 12. Each table gives the stage objective, its numbered
steps (cited elsewhere as "Session N step K"), its acceptance criteria and its end-of-stage
condition (cited elsewhere as "Session N STOP"). Acceptance is shown by executed commands with
explicit pass/fail results and paths to evidence, recorded in `docs/STATUS.md`, `docs/DECISIONS.md`,
`docs/KNOWN_ISSUES.md` and `docs/EXPERIMENT_LEDGER.md` with exact package and dataset revisions,
hashes, configurations, failures, costs and reproduction commands. A GO/NO-GO gate is passed only
when its stated evidence exists and any required approval is recorded; a later stage cannot
retroactively validate a missing earlier gate. A blocked stage is recorded as blocked, with a
recovery plan. Optional stages (8, 9) still record a GO/NO-GO and any deliberate skip.

### Session 0 — Evidence-based feasibility, licenses and environment lock
| Step | Requirement |
|---|---|
| Objective | Prove that real 3D benchmark training and evaluation are possible before large implementation. |
| 1 | Exact mutually compatible robomimic/robosuite commits, Python, MuJoCo bindings and container setup chosen from the official current release docs, with rationale, and pinned reproducibly; no `mujoco-py` dependency unless explicitly justified and compatible. |
| 2 | PH Lift and PH Can dataset URLs, terms, size, HDF5 structure, controller and task metadata verified. **Lift actually downloaded**; Can attempted where reasonable; SHA-256 and manifest computed; other data never silently substituted. |
| 3 | Tiny simulator script with `reset`, `step`, `success`, offscreen render and real 3D robot state; at least one actual demonstration replayed and final state/success compared; tolerances documented; 3D screenshots inspected. |
| 4 | Raw-data availability for later RGB and depth extraction inspected **without downloading large datasets unless approved**; feasibility measured/predicted realistically. |
| 5 | All third-party licenses and required citations inventoried; GitHub repository identity and configured user email verified (creation/push only when authorized). |
| 6 | Minimal skeleton; `make smoke-sim` and `make test` run; `reports/feasibility/STACK_REPORT.md` with logs, artifact hashes, version matrix, downloads, known hazards and the GO/NO-GO decision. |
| Acceptance | One real PH Lift episode file is parsed; a matching 3D sim actually runs; a short replay and offscreen render are observed and compared; controller/action info is inspected; the downstream data path is credible. Otherwise **NO-GO** with a recovery plan, not a training scaffold pretending feasibility. |
| End-of-stage condition | Exact validated revisions, environment commands, dataset SHA, controller/action dimension and success signal recorded; Session 1 follows **only on GO**. |

### Session 1 — Canonical datasets, episode alignment and train/val splits
| Step | Requirement |
|---|---|
| Objective | An audited data pipeline that every policy uses identically. |
| 1 | Dataset fetcher and manifest generator for Lift and Can; resumable downloads and integrity checks without re-downloading unchanged files. |
| 2 | Typed `ObservationSpec`, `ActionSpec`, `TaskSpec`; parser for the actual HDF5 action/proprioception/object keys and initial states; erroneous/missing episodes sanitized without silently deleting data. |
| 3 | Deterministic **episode-level** split manifest; window sampler with observation/action horizons, padding mask and train-only normalization; transformed-data checksums. |
| 4 | Dataset visual explorer and summaries: episode count/duration, action ranges, timestep offsets, state distributions, success flags (only where defined), dataset versions. |
| 5 | No-future-state-leakage tests, window-boundary tests, invertible action-normalizer tests and a one-batch forward-pass smoke test. |
| Acceptance | `make download`, `make verify-data`, selected dataset inspection scripts and pytest tests pass on actual Lift data (Can processed if verified); re-running with the same source hash produces identical splits/hashes; outputs `docs/DATA_DICTIONARY.md`, `docs/DATA_PROVENANCE.md`, `reports/data_audit.json`. |
| End-of-stage condition | Actual keys, tensor shapes, training IDs and normalization hash recorded. |

### Session 2 — 3D simulator adapter, faithful replay, closed-loop evaluator
| Step | Requirement |
|---|---|
| Objective | The research measurement apparatus exists before models are trained. |
| 1 | Stable simulator wrapper: reset from an exact initial snapshot, action stepping, rewards, success, truncation, camera rendering, robot and object world transforms. |
| 2 | Data replay comparison on at least 10 appropriately sampled demos: state/trajectory divergence summary, video inspection, seed reproducibility, controller bounds. |
| 3 | Evaluation initial-state generator split into **validation** and **final test**, with tested episode limits and task success definition; final test states saved hidden/locked; scripts block inadvertent final evaluation until `FINAL_TEST_FREEZE.md` exists. |
| 4 | Random/no-op policy and demonstration-replay instrumentation (clearly labelled non-autonomous); full rollout recorder with frame-indexed kinematics and recorded frames. |
| 5 | End-to-end export of one real simulation clip for a very small web diagnostic; coordinate conversion to three.js verified with reference frames before the frontend is built. |
| Acceptance | Two fresh processes replay the same controlled seed to a matching trajectory within declared tolerances; the environment receives a valid dynamic-dimension action; a random policy produces correctly recorded *closed-loop* outcomes; test metadata identifies train/validation/test partitions; `make smoke-sim` and simulator tests pass. |
| End-of-stage condition | Exact MuJoCo physics/control/dt, reset mechanism and file/transform schema recorded. |

### Session 3 — Research protocol and deterministic/recurrent baselines
| Step | Requirement |
|---|---|
| Objective | Rigorous, functioning baseline policies on the same 3D control problem. |
| 1 | `docs/RESEARCH_PROTOCOL.md` committed *before* reported final test runs: hypotheses, tasks, observation privilege, data splits, paired eval schedule, success definitions, model selection, planned uncertainty analysis. |
| 2 | Shared policy/trainer/checkpoints plus `BC-MLP` and `BC-GRU` or `BC-LSTM`; optionally a GMM policy, only after core checks. |
| 3 | Tiny-batch and tiny-episode overfit tests, full forward/backward/gradient-norm tests, CPU train/eval smoke, checkpoint-reload equivalence. |
| 4 | Validation-only training runs of limited length; initial **closed-loop** Lift behavior gathered; action clipping/sign/coordinate problems diagnosed. |
| 5 | True parameter counts and matched-capacity configurations computed; baseline cost/latency measured. |
| Acceptance | Both learned baselines consume **actual demonstration data**, decrease loss on a small controlled overfit test, load from saved checkpoints and perform real Lift rollouts in the pinned simulator; validation results logged without test contamination. |
| End-of-stage condition | Baseline configs, run IDs, known action/controller quirks and open tuning decisions recorded. |

### Session 4 — Conditional diffusion action policy and truthful denoising instrumentation
| Step | Requirement |
|---|---|
| Objective | A real, testable conditional diffusion policy, not a renamed deterministic head. |
| 1 | Observation encoder, temporal U-Net, timestep conditioning, DDPM/DDIM sampler, normalized action-chunk training, horizon logic, optional EMA, step-inference control loop and episode reset, implemented and tested. |
| 2 | Sampler compared with **robomimic's reference** on fixed synthetic arrays, schedule equations, tensor shapes, seed reproducibility and known reconstruction/noise behaviors. |
| 3 | Tiny-batch overfit with real Lift demonstration windows; training-loss reduction, numerical stability and finite gradients tracked. |
| 4 | `export_denoising` with intermediate **actual** per-timestep action tensors, conditioned observations, normalization, noise seed and scheduler semantics. |
| 5 | A small CPU or authorized GPU validation rollout; action normalization and replanning correctness debugged before any mass training. |
| Acceptance | The network actually performs iterative denoising of action sequences; passes synthetic/math unit tests, checkpoint reload and tiny offline overfit; controls the 3D Lift simulator without invalid actions; all denoising visualization artifacts trace to a real sampler call. |
| End-of-stage condition | Exact architecture, parameter count, scheduler, horizons and validated smoke-run artifacts recorded. |

### Session 5 — Modal + costs + pilot training/evaluation benchmark
| Step | Requirement |
|---|---|
| Objective | Cloud training is reliable and economically bounded; full research costs are estimated. |
| 1 | Modal CLI/SDK entrypoint, pinned CUDA image, Volume for data/checkpoints, safe secrets, machine-readable cost/time log, dry-run and approved launch path. |
| 2 | Resumable/interrupt-safe checkpoints and hash-validated artifact retrieval; training output proven reusable outside Modal. |
| 3 | One short training pilot profiled on the selected GPU and one CPU/parallel rollout pilot; whether rendering is the CPU/GPU bottleneck checked. |
| 4 | Actual projected cost for **3 training seeds × 3 algorithms × two tasks**, plus validation/test rollouts, with storage/rendering/failed-run reserve; alternative budgets if the total is too high. |
| 5 | **Gate C** (§6.3) approved and recorded before any full suite; automatic retry and multi-GPU scale-up prevented. |
| Acceptance | A paid smoke test occurs **only with an explicit recorded approval**; otherwise the working integration and local dry-run are present and cloud execution is marked pending authorization. If approved: real remote training, checkpoint persistence, resume, logs and accurate cost accounting verified. |
| End-of-stage condition | Actual GPU choice, billed/tested runtime, cost proposal and approved limit recorded; Session 6 follows **only after compute authorization or a clearly budget-reduced local path**. |

### Session 6 — Level 1 Lift primary benchmark and checkpoint selection
| Step | Requirement |
|---|---|
| Objective | The primary 3D manipulation comparison is completed on genuinely trained models. |
| 1 | BC-MLP, BC-RNN and diffusion trained on Lift PH with preapproved seeds/budgets under common split/observation/action semantics; failures tracked too. |
| 2 | Checkpoints selected by validation metrics/rollouts and the predeclared rule; tuning only on allowed validation starts, with a bounded equalized search. |
| 3 | Initial model plots: training/validation curves, seed spread, action histograms, validation success and latency, clearly labelled validation, not final test. |
| 4 | Real successes and failures inspected; if all policies perform poorly, data alignment, controller semantics and architecture are diagnosed before the final freeze. A poor but correctly measured result remains reportable; a predetermined success is not chased. |
| 5 | The **Lift** final test configuration frozen only when methodology is stable; model/checkpoint hashes and initial-state schedule recorded. |
| Acceptance | Each core model has at least one real converged training record, independent seeds as approved, actual held-out *validation* rollouts, versioned selected-checkpoint hashes and an all-in cost record; no final-test cherry-picking. |
| End-of-stage condition | Performance reported as validation-only. |

### Session 7 — Level 1 Can, paired held-out testing, robustness, scientific freeze
| Step | Requirement |
|---|---|
| Objective | The Level 1 core research is finished *before* optional perception work. |
| 1 | If Can was verified compatible, fair BC-MLP / BC-RNN / diffusion training and validation repeated on Can; if not, the precise dataset/simulator blocker recorded and the honest Lift result stands. |
| 2 | Reasonable OOD initial-state perturbations, failure categories and at least 2 controlled ablations implemented and preregistered on **validation**; any budget-forced narrowing of scope documented. |
| 3 | Protocol **frozen** in `docs/FINAL_TEST_FREEZE.md`: selected configs, checkpoint hashes, held-out initial states, rollout counts, significance/CI method, task metric aggregations and visual example-selection rule; locked with a Git commit. |
| 4 | Final paired test executed **once**, under the frozen protocol, for all core models and tasks that passed feasibility; no tuning on the final test. |
| 5 | Immutable frozen machine-readable results, CI/pairwise-difference tables, failures and full cost/latency results; engineering-incomplete cases, task failure and statistical uncertainty explicitly distinguished. |
| 6 | The **Level 1 research freeze**, `docs/LEVEL1_COMPLETE.md`, with actual executed results, omissions and reasons; new exploratory follow-ups recorded separately. |
| Acceptance | A real paired held-out closed-loop benchmark with defensible success metrics and traceable model/dataset/simulator hashes; the uncertainty method handles paired initial states and seed replication; source manifests back every number. **Level 1 is then a complete publishable scientific unit** even if no advanced experiments run. |
| End-of-stage condition (Session 7 STOP) | **Hard scientific boundary:** core test selection, labels and findings are never retroactively modified because later RGB/point-cloud results look better. |

### Session 8 — Level 1 RGB-conditioned policy (separate feasibility/budget gate)
| Step | Requirement |
|---|---|
| Objective | Test visual control using camera observations without contaminating core conclusions. |
| 1 | Raw state snapshots, compatible renderer, RGB extraction, exact time-step alignment, camera metadata and storage verified; a small demonstrably aligned clip with ground-truth actions created. |
| 2 | Budget approval recorded, with an **RGB experiment freeze** specifying a fair visual BC vs visual diffusion comparison on Lift; optionally Can. |
| 3 | Image-encoder policies trained using a consistent vision backbone, frozen input keys, camera settings and identical evaluation states. |
| 4 | Actual RGB-conditioned closed-loop evaluation results and representative failure videos generated; if unsuccessful, that fact is published. |
| 5 | If raw data or rendering fails, the RGB study is marked `NOT EXECUTED` with details and the project continues to Session 9; no misleading "vision AI" claim. |
| Acceptance | Either (A) camera-aligned data verified and actual visual policies trained/evaluated with traces, or (B) a documented explicit no-go with reproducibility evidence and no fake results. |
| End-of-stage condition | Core Level 1 metrics remain frozen and unchanged. |

### Session 9 — Level 2 explicit 3D perception, DP3-inspired (optional conditional experiment)
| Step | Requirement |
|---|---|
| Objective | Explore geometric perception *only if* Level 1 results, dataset access, camera calibration and budget support it. |
| 1 | `docs/LEVEL2_GO_NO_GO.md` written, enumerating technical/ethical/licensing/resource blockers; Level 2 needs its own recorded approval. On no-go nothing is trained, the skip is documented and the project moves to Session 10. |
| 2 | Depth/point-cloud observations generated aligned to training episodes and simulator state; camera calibration and 3D point reprojection validated numerically and visually. |
| 3 | Compact point encoder + conditional diffusion implemented and compared with a matched RGB-conditioned diffusion reference on identical physical tasks and held-out initial states. |
| 4 | Spatial-shift evaluation included only if physically meaningful and fully implemented; whether segmentation, camera coordinates or true object poses confer privileged information is logged. |
| 5 | Actual success metrics, point-cloud schema, compute costs, failures and any negative result recorded; labelled **DP3-inspired** unless the exact official method was reproduced. |
| Acceptance | The `GO` path requires a real trained point-cloud policy and matched closed-loop evaluation, plus calibrated point-cloud overlays; the `NO-GO` path requires a justified skip document. **Publication of RoboForge is never gated on Level 2 success.** |
| End-of-stage condition | The GO or NO-GO outcome recorded. |

### Session 10 — RoboForge Observatory: interactive 3D scientific storytelling
| Step | Requirement |
|---|---|
| Objective | A polished, accurate visual explanation driven by *actual frozen experiment artifacts*. |
| 1 | Selected authentic policy rollouts exported for Lift and Can where available, with model/seed/state/physics provenance, real link and object transformations and actual diffusion denoising traces. |
| 2 | 3D kinematic robot viewer with orbit camera, real timeline, highlights, policy selector, replay and actual outcome annotations; verified against 3+ simulator reference frames; visual approximations reported. |
| 3 | Synchronized paired comparison, denoising explorer, model metrics/failures, explanatory scientific storytelling, methodology/credits and responsive navigation. |
| 4 | Optional RGB and point-cloud panels **only if** the corresponding studies were actually run; otherwise only honest future-research context without fake results. |
| 5 | Web asset budgets, mobile layouts, loading states, accessibility, keyboard and reduced motion optimized; optional hosting only when authorized. |
| Acceptance | `npm run build` (or equivalent) passes; E2E/UI sanity checks pass; no frontend fixture data pretends to be measured results; source provenance is visible; a reviewer can confirm animated robot joint motion matches the saved MuJoCo transforms; screenshots/videos/GIFs are genuine. |
| End-of-stage condition | A deployed/staging URL recorded only if it exists, with links to actual assets and any representation limitations. |

### Session 11 — Research paper, reproducibility and independent critical audit
| Step | Requirement |
|---|---|
| Objective | A serious report that survives technical scrutiny without overstating novelty. |
| 1 | Tables/figures generated directly from frozen run manifests: core performance, paired differences + CIs, model/seed dispersion, OOD results, action horizon/denoising ablations, real hardware/cost/latency, failure taxonomy, optional RGB/point-cloud panels if run. |
| 2 | LaTeX paper following the structure in §11; all figures tied to source manifests; all claims supported. |
| 3 | Independent audit of simulator/controller alignment, evaluation seed pairing, split leakage, privilege mismatch, invalid confidence intervals, dataset reuse, GPU expenditure, missing null results, incorrectly claimed novelty or statistical significance. |
| 4 | Review findings require **engineering correction or transparent limitations**; a material post-freeze test-protocol correction is labelled a corrected/replication analysis with full provenance, never silently swapped into the original result. |
| 5 | `paper/main.pdf` generated if the toolchain is available; otherwise exact reproducible compilation commands documented and no PDF claimed. |
| Acceptance | Real methods and results, no placeholders or fabricated numbers, verified third-party bibliography, reproducibility appendix, quantitative limitations and the correct single RoboForge project-author name; all generated charts have run provenance. |
| End-of-stage condition | Independent audit findings, outstanding known issues and paper build proof recorded. |

### Session 12 — Fresh-clone reproduction, final quality assurance, public release
| Step | Requirement |
|---|---|
| Objective | A clean, credible, discoverable repository that someone else can understand and partially reproduce. |
| 1 | Fresh clone in a separate clean environment/container: pinned deps installed, tests run, small dataset/simulator smoke, one checkpoint loaded when distribution permits, at least one reported figure/table reproduced from frozen artifact manifests, website and paper built. |
| 2 | All data download links and exact commands verified; no dead placeholders, private paths, personal secrets, copyrighted dataset copies or broken citations. |
| 3 | All Git commit author/committer names, absence of co-author trailers, GitHub identity and `CITATION.cff` validated; complete commit history reviewed if feasible; published history not rewritten without a recorded approval. |
| 4 | Full test matrix: Python, simulator, research-integrity, website, type checks, lints, security scans, links and licenses; any skipped test documented with precise justification. |
| 5 | README above-the-fold preview generated from real rollout artifacts, with clear key findings, benchmark methods, reproducibility commands, dataset/license requirements, limitations and links to the paper and optional deployed site. |
| 6 | Hosting/security and asset integrity verified; the public repository/release is created and pushed only with authorization and available credentials; a remote push is never claimed unless it happened. |
| 7 | `docs/FINAL_RELEASE_AUDIT.md` published with pass/fail, actual frozen results, executed commands, release hashes and outstanding limitations. |
| Acceptance | The project can be freshly installed/tested and at least a representative analysis result reproduced; public materials match the actual experiments; README/website/paper are internally consistent; the named project author is Zackary Holt; external work is properly credited; remote publication verified if authorized. |
| End-of-stage condition | Project complete: real deliverables, actual findings, limitations and final artifact links summarized; no further phase starts without a new recorded decision. |

# 11. Report, README, reproducibility and scientific language

## 11.1 Paper: required structure
**Working paper title:** *Learning 3D Robotic Manipulation from Demonstrations: A Controlled
Comparison of Behavior Cloning and Diffusion Policies*. **Author:** **Zackary Holt** (and no other
RoboForge author name). **Scope:** simulated 3D control; state-conditioned versus visual-conditioned
perception is labelled explicitly.

| # | Section | Required content |
|---|---|---|
| 1 | Abstract | Research question, benchmark tasks, compared algorithms, dataset and simulator, primary *actual* result and uncertainty, one limitation; written only after experiments |
| 2 | Introduction | Why learning sequential robot control from demonstrations matters; multimodality, compounding error, spatial uncertainty; specific contributions without claiming new diffusion science |
| 3 | Related Work | Behavior cloning, recurrent imitation learning, Diffusion Policy, robomimic/robosuite, optional DP3/geometry-aware control; accurate citations |
| 4 | Benchmark Setup and Data | Exact dataset version/PH source, size, train/val split, action/controller, task rewards/success, simulator revision differences, observed state features and any privileged information |
| 5 | Methods | BC-MLP, BC-RNN, diffusion U-Net, action horizons, normalization, denoising loss and sampler, model/seed/budget fairness, optional model families clearly delineated |
| 6 | Experimental Protocol | Held-out initial-state selection, validation-based checkpoint selection, final test freeze, rollout limits, seeds, paired comparisons, CIs and OOD perturbations |
| 7 | Results | Main success with denominators and uncertainties; seed variation; compute/latency; task-specific and comparative conclusions; null results included |
| 8 | Robustness and Failure Analysis | Real category counts, objective taxonomy, illustrative but honestly selected cases, differences between task/observation regimes |
| 9 | Visual Policy / Point-Cloud Extension | Included **only if run**, with matched modality comparisons; otherwise a future research avenue, with the feasibility gate described transparently |
| 10 | Limitations / Threats to Validity | Sim-to-real gap, privileged state, limited demonstrations, small number of tasks, model tuning budget, simulator-version mismatch with old papers, stochastic policies, distribution shift, reproducibility and cost constraints |
| 11 | Conclusion | Answers supported by actual evidence, not promised improvements |
| 12 | Reproducibility Appendix | Exact git SHA, dependency/container versions, hashes, downloads/license, train/validation IDs, eval state manifests, command lines, timeouts/budget, checkpoint availability, seeded scripts and figure builders |

Charts are made from files in `results/frozen/`, with source manifest/checkpoint IDs in figure
metadata or an index. Result tables are generated by scripts and injected; **headline metrics are
never typed manually into LaTeX**. Success numerator and denominator are defined; a "final" PDF
contains no `TBD` values; the bibliography uses verified sources and author names. Where a data
license disallows redistribution, a repeatable downloader or official external references are
provided; unavailable checkpoints or private credentials are stated as limitations; "fully
reproducible with one command" is not claimed if reproduction needs large paid GPU or a private
file.

## 11.2 README structure, optimized for a GitHub reviewer
Above the fold:

- **ROBOFORGE** in a short, compelling line: "Can a diffusion model teach a robot arm to grasp and
  place objects?"
- One authentic, compact side-by-side actual-simulator demo preview with task/model/source label.
- One precise paragraph on the research and a small table of actual key findings (or
  `Results pending` until available).
- Two clear links: `Explore the Observatory` (only after deployed) and `Read the Paper` (only if a
  real file exists).

Then: research questions, architecture diagram, tasks, comparison protocol, what makes the results
trustworthy, training and evaluation commands, resource requirements, demo selection criteria,
optional RGB/point-cloud status, limitations, credits to external researchers, licenses, and sole
project author Zackary Holt; no long unverified marketing language. `CITATION.cff` references
**Zackary Holt** as project author, the exact release version, and a DOI only if one is issued.
External papers and datasets are cited separately; a GitHub project is not implied to be a
peer-reviewed publication.

## 11.3 Public claims allowed versus forbidden
**Allowed when supported:** "We trained three families of policies and evaluated them on [N] matched
simulator initial states"; "The diffusion policy achieved [X/Y] success on held-out Lift starts";
"Adding point clouds [did/did not] improve performance under [precise condition]."

**Forbidden:** "RoboForge achieves human-level robotics," "This robot learned real-world grasping,"
"First 3D diffusion model," "Proved diffusion superior," "Model understands objects" without
operational definition, "3D perception" when only simulator object-state vectors were used,
"published research paper" when only a local unreviewed PDF exists, or a cherry-picked success clip
that conceals poor measured success.

# 13. Final release acceptance

## 13.2 Final release acceptance matrix
| Requirement | Must be true to claim completion |
|---|---|
| True 3D control | Articulated robot, world-state changes and contacts occur in a validated 3D physics simulator |
| Data provenance | Original official dataset hash, metadata, compatible controller and licenses documented |
| Replay | Real demonstrations replay with tested state/behavior alignment |
| Baselines | Real trained BC and recurrent BC policies; honest tracked checkpoints |
| Diffusion | Real iterative action denoising, conditioned on permitted observations, trained and evaluated |
| Main benchmark | Closed-loop held-out paired initial-state trials and success denominators; uncertainty methodology |
| Two tasks | Lift mandatory; Can performed if compatible or precise documented limitation |
| Honest modalities | Privileged simulation state, RGB-conditioned policies and point clouds clearly distinguished |
| Level 1 freeze | Frozen report and test manifest exist before optional Level 2 development |
| Level 2 | If run, calibrated real depth/point clouds + matched evaluations; otherwise prominently marked optional/not run |
| Compute governance | Approved spend (recorded decision), actual runtime/spend logs, checkpointing, bounded jobs |
| 3D showcase | Authentic kinematic playback or transparently labeled fallback, with verifiable source manifests |
| Denoising | Real saved intermediate sampler tensors, not purely illustrative animation |
| Research report | Real results, statistical uncertainty, limitations, reproducibility steps and verified sources |
| GitHub authorship | Zackary Holt sole named RoboForge author, verified Git author/committer, no co-author trailers |
| External credit | External algorithms/data/frameworks/assets properly cited and licensed |
| Release | Tests/build/reproducibility/links passed or accurately disclosed; public upload only when authorized |
