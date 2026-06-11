# Phase 4: Competition Export

**Status: ✅ IMPLEMENTED** — see `agent/competition_export.py` and `scripts/test_competition_export.py`

## Goal

Export trained model(s) as competition-ready agents that meet the official API contract:
- `policy(observation, reward)` → `np.array([v, h, r], dtype=np.int32)`
- Response time < 0.01s per call
- Both left and right sides trained and playable

---

## Architecture

```
Competition.py  ──calls──▶  TeamRight.policy(raw_rgb, reward)
                                │
                                ▼
                     CompetitionPreprocessor
                         (3, 84, 168) RGB → (16, 84, 168) stacked grayscale
                                │
                                ▼
                     AsyncPolicy._infer_thread (background)
                                │
                                ▼
                         model.predict()  (SB3 PPO)

         Returns PREVIOUS action immediately while background thread
         computes NEXT action.
```

**Key insight**: We don't need TorchScript. SB3's `model.predict()` is fast enough (~0.1ms on CPU) once the preprocessor converts raw RGB to stacked grayscale.

---

## Files

| File | Purpose |
|------|---------|
| `agent/competition_export.py` | `AsyncPolicy`, `TeamRight`, `TeamLeft`, `CompetitionPreprocessor` |
| `scripts/test_competition_export.py` | End-to-end test with dummy model |
| `agent/evaluate.py` | Legacy — re-exports from `competition_export.py` |

---

## Usage

### Competition Submission

```python
# teamX.py (right-side agent)
import numpy as np
from agent.competition_export import CompetitionPreprocessor, AsyncPolicy

class Team:
    def __init__(self):
        self.agent = AsyncPolicy(
            model_path="checkpoint_right.zip",
            device="cpu",
            play_as_left=False,   # right side
            frame_stack=16,        # must match training config
        )

    def policy(self, observation, reward):
        return self.agent.policy(observation, reward)
```

```python
# teamY.py (left-side agent — same checkpoint, different wrapper)
import numpy as np
from agent.competition_export import CompetitionPreprocessor, AsyncPolicy

class Team:
    def __init__(self):
        self.agent = AsyncPolicy(
            model_path="checkpoint_right.zip",  # SAME model
            device="cpu",
            play_as_left=True,    # left side — applies flip + mirror
            frame_stack=16,
        )

    def policy(self, observation, reward):
        return self.agent.policy(observation, reward)
```

### Quick Test

```bash
# Dry-run with dummy model (no Unity needed)
python scripts/test_competition_export.py
```

---

## How It Works

### 1. CompetitionPreprocessor

Replicates the training wrapper's preprocessing pipeline at inference time:

1. Receive raw RGB `(3, 84, 168)` float32 [0,1] from competition
2. Transpose to `(H, W, C)` for OpenCV
3. `cv2.COLOR_RGB2GRAY`
4. `cv2.resize` to `(168, 84)` → output `(84, 168)`
5. Normalize to [0,1]
6. Expand dims → `(1, 84, 168)`
7. Maintain frame stack deque (fills with first frame on init)
8. Concatenate → `(16, 84, 168)`

### 2. AsyncPolicy

Mirrors the official `Competition.py` threading pattern:

- `policy()` returns previous action immediately (<0.01s)
- If no background thread is running, starts one with current obs
- Background thread: preprocess → predict → store next action
- On error, safely falls back to previous action `[1,1,1]` (neutral)

### 3. Left-Side Support

The same model can play both sides:

- **Right side** (`play_as_left=False`): pass through unchanged
- **Left side** (`play_as_left=True`):
  1. Horizontal flip observation before inference
  2. Mirror horizontal action after inference (`h = 2 - h`)

This works because the pickleball court is perfectly symmetric. Training with 4 right + 4 left envs teaches the model both perspectives.

---

## Performance Benchmarks

From `scripts/test_competition_export.py` (M1 Air, CPU):

| Metric | Value |
|--------|-------|
| Cold start response | 0.5–3.5 ms |
| Warm response | 0.1–0.3 ms |
| Competition deadline | 10 ms |
| **Safety margin** | **20–100×** |

---

## Known Limitations

1. **Frame stack persistence**: The preprocessor doesn't auto-detect match boundaries (new match starts). Buffer self-corrects within 16 frames (~0.4s). If competition recreates `Team` instances per match, this is a non-issue.

2. **No TorchScript**: Currently uses SB3 `model.predict()`. TorchScript export was in the original plan but is unnecessary given the 0.1ms inference speed. Could be added for extra safety margin if needed.

3. **`observation` shape assumption**: Preprocessor auto-detects channel order (first dim ≤ 4 = channel-first). Competition sends `(3, 84, 168)` which is correctly handled.

---

## Pass/Fail Criteria

| Metric | Target | Current |
|--------|--------|---------|
| Response < 10ms | 100% of calls | ✅ 0.1-3.5ms |
| Action shape (3,) | All calls | ✅ |
| Action values {0,1,2} | All calls | ✅ |
| dtype int32 | All calls | ✅ |
| Left/right both working | Both | ✅ |
| No crashes on error | Fallback to neutral | ✅ |

---

## Next Steps

1. Train a model → get first meaningful checkpoint
2. Replace dummy checkpoint path with real one in `teamX.py` / `teamY.py`
3. Run `scripts/test_competition_export.py` with real checkpoint
4. Run stress test: 1000 consecutive `policy()` calls, verify 0 crashes
5. Package for competition submission
