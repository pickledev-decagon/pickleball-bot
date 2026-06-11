# Build Log — dPickleBall Competition Bot

## Phase 0: Environment Setup (May 4, 2026)

### Step 0.1 — Conda Environment
```
conda create -n dpickleball pip python=3.10.12
```
Installed dpickleball-ml-agents fork (Unity ML-Agents 1.1.0 modified):
```bash
git clone https://github.com/dPickleball/dpickleball-ml-agents.git
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
pip install stable-baselines3 opencv-python==4.7.0.72 \
            pynput==1.8.1 numpy matplotlib torch torchvision
```

**Issues encountered:**
- `grpcio==1.48.2` has no pre-built wheel for macOS arm64 → installed `grpcio==1.80.0` (API compatible)
- `numpy==2.2.6` incompatible with `opencv-python==4.7.0.72` (ABI break) → downgraded to `numpy==1.26.4`
- `setuptools==82.0.1` removed `pkg_resources`, breaking grpcio build → downgraded to `setuptools==69.5.1`
- SB3 2.8.0 installs `gymnasium` — old `gym` still present for ml-agents backward compatibility

### Step 0.2 — Unity Build
| Detail | Value |
|---|---|
| Source | Google Drive `dPickleball BuildFiless.zip` |
| Size | 395 MB |
| Training build | `Training/Mac/Mac.app/Contents/MacOS/dp` |
| Competition build | `Competition/Mac/Mac.app/Contents/MacOS/dp` |

**Fix**: Unity executable lacked execute permissions — `chmod -R 755`.

### Step 0.3 — Smoke Test
- Observation shape: `(3, 84, 168)` — RGB, 84px height, 168px width
- Agent names: `PAgent1?team=0?agent_id=0` (left), `PAgent2?team=0?agent_id=2` (right)
- env.step() latency: **20-29ms** (includes Unity physics + rendering + gRPC)
- MPS available: `True` (PyTorch 2.11.0)

### Step 0.4 — Key Finding: CPU = GPU for this env
The 20ms env.step() overhead dwarfs model inference (~1-2ms). Switching from MPS to CPU has negligible impact on training throughput. This means we can train on any laptop, including the M1 Air 8GB.

---

## Phase 1: Ball Tracking Preprocessing (May 4, 2026)

### Step 1.1 — Frame Capture
Captured 200 frames from the live Unity environment for analysis.

### Step 1.2 — BallTracker (HSV)
- Default HSV range: `H=[20-35], S=[80-255], V=[80-255]` (standard yellow)
- **Detection rate: 98-100%** on live env frames
- Bug: Unity observations are float32 [0,1], OpenCV HSV expects uint8 [0,255] → converted `(frame*255).astype(np.uint8)`

### Step 1.3 — EnhancedUnityWrapper
| Parameter | Value |
|---|---|
| Frame stack | 4 grayscale frames |
| Observation shape | `(4, 84, 168)` |
| Ball coordinates | Passed via `info["ball_pos"]` dict |
| Frame differencing | Module ready (not enabled by default) |
| Paddle detection | Module ready (not enabled by default) |

---

## Phase 2: Self-Play Training Pipeline (May 4, 2026)

### Version 1 — Single-Env Self-Play (archived)

### Version 2 — Parallel Envs + Curriculum (current)

**Motivation (from lecturer and champion feedback):**

| Feedback | Source | Action Taken |
|---|---|---|
| CPU = GPU, no meaningful difference | Lecturer | Force `device="cpu"` |
| 8 parallel envs is the sweet spot | Champion (last year's winner) | Rebuilt with `DummyVecEnv` × 8 |
| Try curriculum learning | Lecturer | 3-stage curriculum |
| No extra packages in competition | Lecturer | Already compliant — verified |
| Sparse reward problem | Lecturer | Dense reward + paddle-hit detection |

**Status**: In progress.

---

## Phase 3: Model Architecture (planned)
- Hyperparameter tuning
- Dueling network head

## Phase 4: Competition Submission (planned)
- Async inference pattern
- teamX.py / teamY.py
- Stress test

---

## Build 2026-05-20 — Strategic Gap Fixes (Post-Competition-Repo Review)

**Context**: Fetched and analyzed the official `dPickleBallEnv` competition repository (https://github.com/dPickleball/dPickleBallEnv). Identified 6 critical gaps between the training pipeline and actual competition requirements. All fixes applied in this build. **No training was run** — code-only changes.

**Files created:** `agent/eval_harness.py`, `agent/competition_export.py`, `scripts/validate_tracker.py`
**Files modified:** `env/wrapper.py`, `agent/wrapper_vec.py`, `agent/reward_shaping.py`, `env/preprocessing.py`, `agent/experiment_configs.py`, `agent/train_selfplay_v2.py`
**Verification**: `python3 -m py_compile` passed for all 9 files.

---

### Fix 1 — Left-Side Training via Observation Symmetry (🔴 CRITICAL)

**Gap**: Competition requires controlling BOTH left and right paddles. Code hardcoded `possible_agents[1]` (right side only).

**Fix**: Added `play_as_left` parameter to `EnhancedUnityWrapper`. When `True`:
- Agent assigned to `possible_agents[0]` (left)
- Raw RGB frame **horizontally flipped** via `cv2.flip(obs, 1)` before preprocessing
- Action's horizontal component **mirrored back** (`h = 2 - h`) before sending to Unity
- `create_vec_env` automatically splits 8 envs as **4 right + 4 left**

**Rationale**: The court is perfectly symmetric. One model learns both sides without doubling compute. At inference, the same checkpoint loads for `TeamRight` and `TeamLeft` (with flip/mirror wrapper).

**Files**: `env/wrapper.py` (lines 10–22, 65–68, 156–162), `agent/wrapper_vec.py` (lines 10, 39, 47, 61)

---

### Fix 2 — Async Inference Policy for 0.01s Deadline (🔴 CRITICAL)

**Gap**: Competition `Competition.py` runs policy inference in background threads with 0.01s response deadline. Previous action reused if thread not finished. Current `evaluate.py` did synchronous `model.predict()` which would miss the deadline.

**Fix**: Created `agent/competition_export.py` with `AsyncPolicy` class:
- `policy(obs, reward)` returns **previous action immediately** (< 0.01s guaranteed)
- Background thread started (if not running) to compute **next action** from current obs
- On inference error, safely retains previous action
- `TeamRight` / `TeamLeft` subclasses handle competition export
- Left-side subclass applies obs flip + action mirror transparently

**Rationale**: Mirrors the official competition threading model exactly. The script always receives an action on time, with the tradeoff of 1-step action delay (standard in real-time RL robotics).

**File**: `agent/competition_export.py` (187 lines)

---

### Fix 3 — Deeper Frame Stack for Temporal Context (🟡 HIGH)

**Gap**: Default 4-frame stack = ~100ms history. Ball trajectory after wall bounces requires ~500ms+ to predict. Official example uses 64 frames.

**Fix**:
- Default frame stack raised **4 → 16** across all creation paths
- Added experiment configs `stack_16` and `stack_32`
- Memory footprint checked: 16 frames × 84 × 168 × 8 envs = ~7 MB (negligible on 16GB Mac)

**Rationale**: At ~40 FPS, 16 frames = 400ms. Enough to see ball approach wall, hit, and rebound. Official example's 64 frames may be overkill; 16 is a compute-efficient starting point.

**Files**: `env/wrapper.py` (line 11), `agent/wrapper_vec.py` (lines 10, 47), `agent/train_selfplay_v2.py` (line 16), `agent/experiment_configs.py` (lines 108–122)

---

### Fix 4 — Win-Rate Evaluation Harness (🟡 HIGH)

**Gap**: Training evaluated via `explained_variance` and `value_loss` — training diagnostics, not competition metrics. No way to measure "can this bot actually win matches?"

**Fix**: Created `agent/eval_harness.py` with `MatchEvaluator`:
- Plays **full matches to 21 points** (respects actual competition scoring)
- Opponent types: `no_op`, `random`, `square`, `self`
- Reports: `win_rate`, `wins/losses/draws`, `avg_match_len`, `avg_point_diff`
- `benchmark_all()` runs the full suite in one call

**Pass criteria established**:
| Opponent | Minimum Win Rate |
|---|---|
| no_op | > 95% |
| random | > 90% |
| square | > 85% |
| self (previous checkpoint) | > 55% and improving |

**Rationale**: Win rate is the only metric that matters for competition success. This harness replaces training diagnostics as the north star for model selection.

**File**: `agent/eval_harness.py` (155 lines)

---

### Fix 5 — Shot Placement / Rotation Reward (🟡 MEDIUM)

**Gap**: Reward shaping only rewarded proximity, hitting, and aggression. Ignored **ball placement quality** (corners are harder to return) and the **rotation action** (which controls spin/angle).

**Fix**: Added to `agent/reward_shaping.py`:
- `get_ball_land_zone(ball_x, ball_y)` — categorizes landing zone (`corner`, `far_back`, `near_net`, etc.)
- `placement_reward()` — rewards corner/far-back shots most (0.5), near-net least (0.1)
- `shaped_reward()` now accepts `placement_bonus` parameter
- Added `placement_bonus` experiment config (default 0.3)
- **Rotation action is no longer hardcoded** — wrapper passes all 3 action components through

**Rationale**: A bot that hits the ball straight back is easy to return. Winning requires placement. The 5-second competition rule means weak returns lose. Rotation enables angled shots that use walls.

**Files**: `agent/reward_shaping.py` (lines 20–50, 84–99), `agent/experiment_configs.py` (lines 124–128)

---

### Fix 6 — Ball Tracker Occlusion Handling + Validation (🟡 MEDIUM)

**Gap**: `BallTracker` returns `(-1, -1)` on miss with no recovery. Occlusion by paddle, motion blur, or lighting changes would break the heatmap observation.

**Fix**:
- Added `SmoothBallTracker` class to `env/preprocessing.py`:
  - Maintains velocity estimate via exponential smoothing (`alpha=0.7`)
  - On miss, predicts next position from velocity
  - Bounds predictions to `[0, 1]` court range
- Created `scripts/validate_tracker.py`:
  - Runs tracker on **actual Unity frames** for N steps
  - Reports: detection rate, miss rate, max consecutive misses, quick recoveries
  - Warns if miss rate > 10% or max consecutive misses > 20

**Rationale**: Heatmap observation is the team's biggest edge (explained variance 0.329 vs 0.009 baseline). But it's worthless if the tracker fails during a rally. Validation on real frames is mandatory before trusting it in competition.

**Files**: `env/preprocessing.py` (lines 94–129), `scripts/validate_tracker.py` (123 lines)

---

### Compilation Verification

```bash
python3 -m py_compile \
  env/wrapper.py \
  agent/wrapper_vec.py \
  agent/reward_shaping.py \
  env/preprocessing.py \
  agent/experiment_configs.py \
  agent/train_selfplay_v2.py \
  agent/eval_harness.py \
  agent/competition_export.py \
  scripts/validate_tracker.py
# Result: ALL FILES COMPILED SUCCESSFULLY
```

---

### Known Limitations / Not Yet Addressed

1. **Self-play buffer gap**: Stage 3 opponent buffer is empty on a fresh run until final checkpoint is saved. Intermediate checkpoints need to be added to buffer during training (requires win-rate evaluation of each checkpoint).
2. **Unity build path hardcoded**: All scripts assume Mac build at `~/Downloads/dPickleball BuildFiles/Training/Mac/Mac.app`. Cross-platform (Windows/Linux) paths not yet parameterized.
3. **Evaluate.py transpose bug risk**: `agent/evaluate.py` still has the potential channel-order bug identified in prior audit. Superseded by `agent/competition_export.py` for competition use.
4. **No MPS utilization**: `device="cpu"` everywhere. Could experiment with `device="mps"` for model inference once env.step() latency is confirmed as the bottleneck.

---

### Next Steps (Post-Build)

| Priority | Action | File / Script |
|---|---|---|
| 1 | Run tracker validation on actual Unity frames | `python scripts/validate_tracker.py --steps 1000` |
| 2 | 10k-step smoke test with 4-left/4-right envs | `python run_experiment.py --config stack_16 --steps 10000 --envs 2` |
| 3 | Add intermediate checkpoint → buffer logic in `train_selfplay_v2.py` | `agent/train_selfplay_v2.py` |
| 4 | Run full win-rate eval after first meaningful checkpoint | `agent/eval_harness.py` |
| 5 | Test competition export end-to-end with dummy obs | `agent/competition_export.py` |

---

## Build 2026-05-20b — A2 Bug Fix (Competition Export Preprocessor)

**Context**: During A2 validation (competition export dry-run), discovered that `AsyncPolicy` received raw `(3, 84, 168)` RGB from the competition script but passed it directly to the model, which expected `(16, 84, 168)` stacked grayscale. This caused silent inference failures — every step fell back to `[1,1,1]` neutral action.

**Severity**: 🔴 CRITICAL — would cause the bot to do nothing in competition.

**Fix**: Rewrote `agent/competition_export.py`:
- Added `CompetitionPreprocessor` class that replicates `EnhancedUnityWrapper._preprocess()`:
  1. Transpose `(3, H, W)` → `(H, W, 3)` for OpenCV
  2. `COLOR_RGB2GRAY`
  3. `cv2.resize` to `(168, 84)` → output shape `(84, 168)`
  4. Normalize to `[0, 1]` float32
  5. Expand dims → `(1, 84, 168)`
  6. Maintain `deque` frame stack across `policy()` calls
  7. First call fills buffer with N copies (matches wrapper `reset()` behavior)
  8. Concatenate → `(stack, 84, 168)`
- `AsyncPolicy` now instantiates `CompetitionPreprocessor(frame_stack=16)`
- `policy()` calls `preprocessor.process(raw_obs)` before starting background inference
- `TeamRight` / `TeamLeft` pass `frame_stack=16` to constructor

**Files modified**: `agent/competition_export.py`, `scripts/test_competition_export.py`

**Validation**:
```bash
python scripts/test_competition_export.py
# Result: 🎉 All competition export tests passed!
#   TeamRight: <0.01s response, actions in {0,1,2}
#   TeamLeft: <0.01s response, actions in {0,1,2}
#   Preprocessor: (3,84,168) -> (16,84,168) ✓
```

**Known limitation**: Preprocessor doesn't auto-detect match boundaries (new match starts). Buffer self-corrects within 16 frames (~0.4s). If competition recreates `TeamX` instances per match, this is a non-issue.

---

## Build 2026-05-20c — C1: Mid-Training Checkpoint → Opponent Buffer

**Problem**: `OpponentBuffer` was empty during Stage 3 self-play (60–100% of training). Only the final checkpoint was added after training completed. This meant 40% of training timesteps were wasted against no-op opponents instead of progressively stronger past versions of the agent.

**Fix**: Created `agent/selfplay_callback.py` — a custom SB3 `BaseCallback` subclass that replaces `CheckpointCallback`:

1. **Standard checkpoint save** to `checkpoints_selfplay_v2/` (same as before)
2. **Auto-copy** to `opponent_buffer_v2/` with a `*_mid_{timesteps}.zip` name
3. **Auto-add** to `OpponentBuffer` with a heuristic `win_rate = current_step / total_steps`
   - Later checkpoints get higher heuristic scores (assumed stronger)
   - Buffer sorts by win_rate and evicts weakest beyond `max_size=5`
4. **Stage 3 curriculum** now samples from a living pool of past checkpoints instead of falling back to no-op

**Files created**: `agent/selfplay_callback.py`
**Files modified**: `agent/train_selfplay_v2.py` (replaced `CheckpointCallback` with `SelfPlayCheckpointCallback`)

**Checkpoint timeline example** (300k total steps, save every 30k):
| Step | Event | Buffer state |
|------|-------|--------------|
| 0 | Training starts | Empty |
| 30k | Mid-checkpoint saved → buffer | 1 entry |
| 60k | Mid-checkpoint saved → buffer | 2 entries |
| 90k | Mid-checkpoint saved → buffer | 3 entries |
| 120k | Mid-checkpoint saved → buffer | 4 entries |
| 150k | Mid-checkpoint saved → buffer | 5 entries |
| 180k | **Stage 3 begins** + mid-checkpoint → buffer | 6 entries (oldest evicted) |
| 210k | Mid-checkpoint saved → buffer | 6 entries (rolling) |
| 240k | Mid-checkpoint saved → buffer | 6 entries (rolling) |
| 270k | Mid-checkpoint saved → buffer | 6 entries (rolling) |
| 300k | Final checkpoint saved → buffer | 6 entries |

**Known limitation**: Heuristic win_rate uses training progress, not actual win-rate evaluation. For accurate sorting, run `agent/eval_harness.py` offline and update `opponent_buffer_v2/manifest.json`.

**Compilation**: `python3 -m py_compile agent/selfplay_callback.py agent/train_selfplay_v2.py` → OK

---

## Build 2026-05-20d — C2+C3+C4: Tech Debt Cleanup

### C2: Parameterized Unity Build Path

**Problem**: Build path hardcoded to `~/Downloads/dPickleball BuildFiles/Training/Mac/Mac.app` in 12 files across the codebase.

**Fix**: Created `config.py` — single source of truth for all paths and defaults:
- `BUILD_PATH` — auto-detected from `DPICKLEBALL_BUILD_PATH` env var → `.env` file → Mac default
- `NUM_ENVS`, `FRAME_STACK`, `STAGE*_FRAC`, `TIMESTEPS_*`, `BUFFER_MAX_SIZE` — migrated from scattered constants
- `CHECKPOINT_DIR`, `LOG_DIR`, `BUFFER_DIR`, `RESULTS_DIR` — canonical directory paths
- `PPO_DEFAULTS` — default PPO hyperparameters

**Files modified**: `config.py` (new), `agent/train_selfplay_v2.py`, `run_experiment.py`, `scripts/grid_search.py`, `scripts/validate_tracker.py`

To move the build: set `DPICKLEBALL_BUILD_PATH` env var or create `.env` in project root.

### C3: Fixed `agent/evaluate.py`

**Problem**: Old file had transpose bugs, no frame stacking, no async inference, no left-side support. Would silently fail in competition.

**Fix**: Replaced with a thin re-export from `agent/competition_export.py`. Old `create_agent_from_checkpoint()` signature preserved for backward compatibility but internally uses `AsyncPolicy`.

### C4: Updated `phase_4_competition_export.md`

**Problem**: Documentation described a TorchScript-based approach that was never implemented and doesn't match the actual `competition_export.py` architecture.

**Fix**: Rewrote with:
- Actual architecture diagram (CompetitionPreprocessor → AsyncPolicy → model.predict)
- Working code examples for `teamX.py` / `teamY.py`
- Performance benchmarks from real A2 test
- Known limitations and next steps

### Compilation Verification

```bash
python3 -m py_compile config.py agent/train_selfplay_v2.py run_experiment.py \
  scripts/grid_search.py scripts/validate_tracker.py agent/evaluate.py
# Result: ALL 6 COMPILED OK
```
