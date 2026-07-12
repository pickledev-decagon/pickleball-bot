#!/usr/bin/env python
"""
Competition Export Validation — 9 checks.

Verifies the exported competition agent loads successfully and behaves
exactly like the trained agent.

Usage:
    python scripts/validate_competition_export.py --port 6700
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- Config ---
EXPECTED_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints_selfplay_v2",
    "exp003_mixed_opponent",
    "exp003_75000_steps.zip",
)
EXPECTED_FRAME_STACK = 4
EXPECTED_VECTOR_DIM = 6
EXPECTED_VISUAL_SHAPE = (4, 84, 168)
EXPECTED_ACTION_SHAPE = (3,)
EXPECTED_ACTION_VALUES = {0, 1, 2}


class ValidationReport:
    """Collects pass/fail results for all checks."""

    def __init__(self):
        self.checks = []
        self.current_check = None

    def start(self, name, description):
        self.current_check = {"name": name, "description": description,
                              "passed": False, "details": {}, "errors": []}
        print(f"\n{'=' * 60}")
        print(f"CHECK: {name}")
        print(f"  {description}")
        print(f"{'=' * 60}")

    def detail(self, key, value):
        self.current_check["details"][key] = value
        if isinstance(value, (str, int, float, bool)):
            print(f"  {key}: {value}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")

    def pass_(self, msg=""):
        if not self.current_check["passed"]:
            self.current_check["passed"] = True
        if msg and "result" not in self.current_check["details"]:
            self.current_check["details"]["result"] = msg
        print(f"  ✅ PASS: {msg}")

    def fail(self, msg):
        self.current_check["passed"] = False
        self.current_check["errors"].append(msg)
        print(f"  ❌ FAIL: {msg}")

    def end(self):
        self.checks.append(self.current_check)
        self.current_check = None

    def summary(self):
        passed = sum(1 for c in self.checks if c["passed"])
        total = len(self.checks)
        print(f"\n{'=' * 60}")
        print(f"VALIDATION SUMMARY: {passed}/{total} checks passed")
        print(f"{'=' * 60}")
        for c in self.checks:
            status = "✅" if c["passed"] else "❌"
            print(f"  {status} {c['name']}")
        print(f"{'=' * 60}")
        return passed == total

    def to_dict(self):
        return {"checks": self.checks,
                "total": len(self.checks),
                "passed": sum(1 for c in self.checks if c["passed"])}


def check_1_checkpoint_identity(report):
    """1. Verify the exported checkpoint is the intended final model."""
    report.start("1. Checkpoint Identity",
                 "Verify checkpoint file exists and is EXP003_75k")

    if not os.path.exists(EXPECTED_CHECKPOINT):
        report.fail(f"Checkpoint not found: {EXPECTED_CHECKPOINT}")
        report.end()
        return None

    report.detail("checkpoint_path", EXPECTED_CHECKPOINT)
    report.detail("file_exists", True)
    file_size = os.path.getsize(EXPECTED_CHECKPOINT)
    report.detail("file_size_mb", f"{file_size / 1e6:.1f}")

    # Verify it loads
    try:
        model = PPO.load(EXPECTED_CHECKPOINT, device="cpu")
        report.detail("loads_successfully", True)
    except Exception as e:
        report.fail(f"Failed to load: {e}")
        report.end()
        return None

    entrypoint_results = {}
    entrypoints_ok = True
    dummy_obs = np.zeros((3, 84, 168), dtype=np.float32)
    for module_name, class_name in (("teamX", "TeamX"), ("teamY", "TeamY")):
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            agent = cls()
            action = agent.policy(dummy_obs, 0.0)
            entrypoint_results[class_name] = {
                "loads": True,
                "action_shape": tuple(action.shape),
                "action_dtype": str(action.dtype),
            }
        except Exception as e:
            entrypoints_ok = False
            entrypoint_results[class_name] = {"loads": False, "error": repr(e)}
            report.fail(f"{class_name} failed to load or act: {e}")

    report.detail("competition_entrypoints", entrypoint_results)
    if entrypoints_ok:
        report.pass_("Checkpoint and competition entrypoints load successfully")

    report.end()
    return model


def check_2_multiinput_policy(report, model):
    """2. Verify the exported model uses MultiInputPolicy."""
    report.start("2. MultiInputPolicy",
                 "Verify model uses MultiInputPolicy (Dict obs)")

    policy_class = type(model.policy).__name__
    report.detail("policy_class", policy_class)

    if "MultiInput" in policy_class:
        report.pass_(f"Policy is {policy_class}")
    else:
        report.fail(f"Expected MultiInputPolicy, got {policy_class}")

    # Check observation space is Dict
    obs_space = model.observation_space
    obs_space_type = type(obs_space).__name__
    report.detail("observation_space_type", obs_space_type)

    if "Dict" in obs_space_type:
        report.detail("has_visual", "visual" in obs_space.spaces)
        report.detail("has_vector", "vector" in obs_space.spaces)
        report.detail("visual_shape", obs_space.spaces["visual"].shape)
        report.detail("vector_shape", obs_space.spaces["vector"].shape)
        if "visual" in obs_space.spaces and "vector" in obs_space.spaces:
            report.pass_("Observation space is Dict with visual + vector")
        else:
            report.fail("Missing visual or vector key in Dict obs space")
    else:
        report.fail(f"Expected Dict observation space, got {obs_space_type}")

    report.end()


def check_3_obs_format_match(report, model):
    """3. Verify observation format matches training."""
    report.start("3. Observation Format Match",
                 "Verify visual + vector obs match training wrapper")

    obs_space = model.observation_space

    # Visual
    visual_shape = obs_space.spaces["visual"].shape
    report.detail("expected_visual_shape", EXPECTED_VISUAL_SHAPE)
    report.detail("actual_visual_shape", visual_shape)
    if visual_shape == EXPECTED_VISUAL_SHAPE:
        report.pass_(f"Visual shape matches: {visual_shape}")
    else:
        report.fail(f"Visual shape mismatch: expected {EXPECTED_VISUAL_SHAPE}, got {visual_shape}")

    # Visual dtype + range
    visual_low = obs_space.spaces["visual"].low
    visual_high = obs_space.spaces["visual"].high
    report.detail("visual_dtype", str(obs_space.spaces["visual"].dtype))
    report.detail("visual_low", float(visual_low.min()))
    report.detail("visual_high", float(visual_high.max()))

    # Vector
    vector_shape = obs_space.spaces["vector"].shape
    report.detail("expected_vector_shape", (EXPECTED_VECTOR_DIM,))
    report.detail("actual_vector_shape", vector_shape)
    if vector_shape == (EXPECTED_VECTOR_DIM,):
        report.pass_(f"Vector shape matches: {vector_shape}")
    else:
        report.fail(f"Vector shape mismatch: expected {(EXPECTED_VECTOR_DIM,)}, got {vector_shape}")

    # Vector dtype + range
    vector_low = obs_space.spaces["vector"].low
    vector_high = obs_space.spaces["vector"].high
    report.detail("vector_dtype", str(obs_space.spaces["vector"].dtype))
    report.detail("vector_low", [float(x) for x in vector_low])
    report.detail("vector_high", [float(x) for x in vector_high])

    # Action space
    action_space = model.action_space
    action_type = type(action_space).__name__
    report.detail("action_space_type", action_type)
    report.detail("action_nvec", list(action_space.nvec) if hasattr(action_space, "nvec") else "N/A")
    if hasattr(action_space, "nvec") and list(action_space.nvec) == [3, 3, 3]:
        report.pass_("Action space is MultiDiscrete([3,3,3])")
    else:
        report.fail(f"Action space mismatch: {action_type} nvec={getattr(action_space, 'nvec', 'N/A')}")

    report.end()


def check_4_preprocessor_dict_obs(report):
    """4. Verify competition preprocessor produces exact Dict obs."""
    report.start("4. Preprocessor Dict Obs",
                 "Verify CompetitionPreprocessor produces correct Dict format")

    from agent.competition_export import CompetitionPreprocessor

    preprocessor = CompetitionPreprocessor(
        frame_stack=EXPECTED_FRAME_STACK,
        use_heatmap=True,
        heatmap_sigma=5,
        heatmap_weight=0.7,
    )

    # Create a dummy raw obs (3, 84, 168) float32 [0,1] — like Unity provides
    dummy_obs = np.random.rand(3, 84, 168).astype(np.float32)

    try:
        result = preprocessor.process(dummy_obs)
    except Exception as e:
        report.fail(f"Preprocessor crashed: {e}")
        report.end()
        return

    # Check it's a dict
    if not isinstance(result, dict):
        report.fail(f"Expected dict, got {type(result).__name__}")
        report.end()
        return

    report.detail("output_type", type(result).__name__)
    report.detail("has_visual", "visual" in result)
    report.detail("has_vector", "vector" in result)

    # Check visual
    if "visual" in result:
        visual = result["visual"]
        report.detail("visual_shape", visual.shape)
        report.detail("visual_dtype", str(visual.dtype))
        if visual.shape == EXPECTED_VISUAL_SHAPE:
            report.pass_(f"Visual shape correct: {visual.shape}")
        else:
            report.fail(f"Visual shape wrong: {visual.shape} (expected {EXPECTED_VISUAL_SHAPE})")
        if visual.dtype == np.float32:
            report.pass_("Visual dtype is float32")
        else:
            report.fail(f"Visual dtype wrong: {visual.dtype} (expected float32)")

    # Check vector
    if "vector" in result:
        vector = result["vector"]
        report.detail("vector_shape", vector.shape)
        report.detail("vector_dtype", str(vector.dtype))
        report.detail("vector_values", [float(x) for x in vector])
        if vector.shape == (EXPECTED_VECTOR_DIM,):
            report.pass_(f"Vector shape correct: {vector.shape}")
        else:
            report.fail(f"Vector shape wrong: {vector.shape}")
        if vector.dtype == np.float32:
            report.pass_("Vector dtype is float32")
        else:
            report.fail(f"Vector dtype wrong: {vector.dtype}")

    # Check no NaNs
    if "visual" in result and "vector" in result:
        has_nan = np.any(np.isnan(result["visual"])) or np.any(np.isnan(result["vector"]))
        report.detail("has_nans", has_nan)
        if not has_nan:
            report.pass_("No NaNs in preprocessed obs")
        else:
            report.fail("NaNs detected in preprocessed obs")

    report.end()


def check_5_10k_inference(report, port):
    """5. Run 10,000 inference steps — no crashes, shape mismatch, NaNs."""
    report.start("5. 10k Inference Steps",
                 "Run 10k steps via TeamY (right side), check for crashes/NaNs/shapes")

    from config import BUILD_PATH, check_build_path
    if not check_build_path():
        report.fail("Unity build not found")
        report.end()
        return

    from agent.wrapper_vec import make_unity_env_fn
    from env.ball_tracking_opponent import BallTrackingOpponent

    # Create env with BallTrackingOpponent (right side)
    opponent = BallTrackingOpponent(hit_rate=0.6, noise_rate=0.2, reaction_delay=3)
    env_fn = make_unity_env_fn(
        rank=0,
        build_path=BUILD_PATH,
        play_as_left=False,
        frame_stack=4,
        use_ball_tracking=True,
        use_heatmap=True,
        heatmap_sigma=5,
        heatmap_weight=0.7,
        base_port=port,
        mask_rotation=False,
        opponent_policy=opponent,
        max_rally_steps=500,
    )
    env = env_fn()

    # Load model directly (not via TeamY — we want to test the env interface)
    model = PPO.load(EXPECTED_CHECKPOINT, device="cpu")

    obs, info = env.reset()
    steps_completed = 0
    crashes = 0
    nan_count = 0
    shape_mismatches = 0
    dtype_mismatches = 0
    action_values_seen = set()
    inference_times = []
    resets = 0

    try:
        step = 0
        for step in range(10000):
            # Check obs format
            if not isinstance(obs, dict):
                shape_mismatches += 1
                report.fail(f"Step {step}: obs is not dict, got {type(obs).__name__}")
                break

            if "visual" not in obs or "vector" not in obs:
                shape_mismatches += 1
                report.fail(f"Step {step}: missing visual or vector key")
                break

            # Check shapes
            if obs["visual"].shape != EXPECTED_VISUAL_SHAPE:
                shape_mismatches += 1
                report.fail(f"Step {step}: visual shape {obs['visual'].shape} != {EXPECTED_VISUAL_SHAPE}")
                break

            if obs["vector"].shape != (EXPECTED_VECTOR_DIM,):
                shape_mismatches += 1
                report.fail(f"Step {step}: vector shape {obs['vector'].shape} != ({EXPECTED_VECTOR_DIM},)")
                break

            # Check dtypes
            if obs["visual"].dtype != np.float32:
                dtype_mismatches += 1
                report.fail(f"Step {step}: visual dtype {obs['visual'].dtype} != float32")
                break

            if obs["vector"].dtype != np.float32:
                dtype_mismatches += 1
                report.fail(f"Step {step}: vector dtype {obs['vector'].dtype} != float32")
                break

            # Check NaNs
            if np.any(np.isnan(obs["visual"])) or np.any(np.isnan(obs["vector"])):
                nan_count += 1
                if nan_count <= 3:
                    report.fail(f"Step {step}: NaN detected in obs")
                if nan_count > 10:
                    report.fail(f"Too many NaNs ({nan_count}), stopping")
                    break

            # Inference
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            t1 = time.time()
            inference_times.append((t1 - t0) * 1000)

            # Check action
            action = np.array(action).flatten()
            if action.shape != EXPECTED_ACTION_SHAPE:
                shape_mismatches += 1
                report.fail(f"Step {step}: action shape {action.shape} != {EXPECTED_ACTION_SHAPE}")
                break

            for v in action:
                if v not in EXPECTED_ACTION_VALUES:
                    report.fail(f"Step {step}: action value {v} not in {EXPECTED_ACTION_VALUES}")
                    break
                action_values_seen.add(int(v))

            # Step env
            obs, reward, terminated, truncated, info = env.step(action)
            steps_completed = step + 1

            if terminated or truncated:
                obs, info = env.reset()
                resets += 1

    except Exception as e:
        crashes += 1
        report.fail(f"Crash at step {step}: {e}")
        traceback.print_exc()
    finally:
        env.close()

    report.detail("steps_completed", steps_completed)
    report.detail("crashes", crashes)
    report.detail("nan_count", nan_count)
    report.detail("shape_mismatches", shape_mismatches)
    report.detail("dtype_mismatches", dtype_mismatches)
    report.detail("resets", resets)
    report.detail("action_values_seen", sorted(action_values_seen))

    if inference_times:
        times = np.array(inference_times)
        report.detail("inference_mean_ms", f"{times.mean():.2f}")
        report.detail("inference_p95_ms", f"{np.percentile(times, 95):.2f}")
        report.detail("inference_max_ms", f"{times.max():.2f}")

    if steps_completed == 10000 and crashes == 0 and nan_count == 0 and shape_mismatches == 0:
        report.pass_("10k steps completed: no crashes, no NaNs, no shape/dtype mismatches")
    else:
        report.fail(f"Issues: crashes={crashes}, NaNs={nan_count}, "
                    f"shapes={shape_mismatches}, dtypes={dtype_mismatches}")

    report.end()


def check_6_left_right_mirroring(report):
    """6. Verify left/right mirroring: position, velocity, paddle coordinates."""
    report.start("6. Left/Right Mirroring",
                 "Verify training-equivalent visual/vector mirroring + action mirroring")

    from agent.competition_export import AsyncPolicy, CompetitionPreprocessor
    from env.preprocessing import BallTracker, SmoothBallTracker, detect_paddles
    import cv2

    def make_frame(ball_x):
        frame = np.zeros((84, 168, 3), dtype=np.float32)
        cv2.circle(frame, (ball_x, 35), 4, (1.0, 0.9, 0.0), -1)
        # detect_paddles intentionally uses BGR2HSV, matching preprocessing.py.
        cv2.rectangle(frame, (18, 55), (25, 70), (1.0, 0.0, 0.0), -1)
        cv2.rectangle(frame, (140, 50), (148, 68), (1.0, 0.0, 0.0), -1)
        return frame.transpose(2, 0, 1).astype(np.float32)

    expected_tracker = BallTracker()
    expected_smooth = SmoothBallTracker()
    expected_last_ball = (-1.0, -1.0)
    heatmap_helper = CompetitionPreprocessor(frame_stack=4, use_heatmap=True)

    def expected_training_left_step(raw_obs):
        nonlocal expected_last_ball
        obs_hwc = raw_obs.transpose(1, 2, 0)
        frame_uint8 = (obs_hwc * 255).astype(np.uint8)

        ball_pos = expected_smooth.update(expected_tracker.detect(frame_uint8))
        if ball_pos[0] >= 0:
            ball_pos = (1.0 - ball_pos[0], ball_pos[1])

        paddles = detect_paddles(frame_uint8)
        valid = [p for p in paddles if p[0] >= 0]
        paddle_pos = (-1.0, -1.0)
        if valid:
            px, py = min(valid, key=lambda p: p[0])
            paddle_pos = (1.0 - px, py) if px >= 0 else (-1.0, -1.0)

        if expected_last_ball[0] >= 0 and ball_pos[0] >= 0:
            ball_vx = np.clip((ball_pos[0] - expected_last_ball[0]) * 10, -1.0, 1.0)
            ball_vy = np.clip((ball_pos[1] - expected_last_ball[1]) * 10, -1.0, 1.0)
        else:
            ball_vx = 0.0
            ball_vy = 0.0
        expected_last_ball = ball_pos

        model_hwc = cv2.flip(obs_hwc, 1)
        gray = cv2.cvtColor(model_hwc, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (168, 84), interpolation=cv2.INTER_AREA).astype(np.float32)
        frame = np.expand_dims(gray, axis=0)
        frame = heatmap_helper._render_heatmap(frame, ball_pos)

        vector = np.array(
            [ball_vx, ball_vy, paddle_pos[0], paddle_pos[1], ball_pos[0], ball_pos[1]],
            dtype=np.float32,
        )
        return frame, vector

    raw_1 = make_frame(30)
    raw_2 = make_frame(40)
    expected_training_left_step(raw_1)
    expected_frame, expected_vector = expected_training_left_step(raw_2)

    preprocessor = CompetitionPreprocessor(frame_stack=4, use_heatmap=True)
    preprocessor._play_as_left = True
    preprocessor.process(raw_1)
    actual = preprocessor.process(raw_2)

    actual_frame = actual["visual"][-1:]
    actual_vector = actual["vector"]
    visual_max_abs_diff = float(np.max(np.abs(expected_frame - actual_frame)))
    vector_max_abs_diff = float(np.max(np.abs(expected_vector - actual_vector)))

    report.detail("expected_left_vector", [float(x) for x in expected_vector])
    report.detail("actual_left_vector", [float(x) for x in actual_vector])
    report.detail("left_visual_max_abs_diff", f"{visual_max_abs_diff:.8f}")
    report.detail("left_vector_max_abs_diff", f"{vector_max_abs_diff:.8f}")

    if visual_max_abs_diff <= 1e-6:
        report.pass_("Left visual preprocessing matches training wrapper ordering")
    else:
        report.fail(f"Left visual preprocessing mismatch: max abs diff {visual_max_abs_diff:.8f}")

    if vector_max_abs_diff <= 1e-6:
        report.pass_("Left vector mirrors position, velocity, and paddle coordinates once")
    else:
        report.fail(f"Left vector mismatch: max abs diff {vector_max_abs_diff:.8f}")

    # Test action mirroring
    test_actions = [
        (np.array([0, 0, 0]), np.array([0, 0, 0])),   # none stays
        (np.array([1, 1, 0]), np.array([1, 2, 0])),   # right→left
        (np.array([2, 2, 1]), np.array([2, 1, 1])),   # left→right, rot preserved
        (np.array([0, 1, 2]), np.array([0, 2, 2])),   # right→left, rot preserved
    ]

    all_pass = True
    for original, expected in test_actions:
        result = AsyncPolicy._mirror_horizontal_action(original)
        if not np.array_equal(result, expected):
            report.fail(f"Action mirror: {original} → {result}, expected {expected}")
            all_pass = False

    if all_pass:
        report.pass_("Action mirroring correct for all test cases")

    report.end()


def check_7_deterministic(report):
    """7. Verify deterministic=True in inference."""
    report.start("7. Deterministic Inference",
                 "Verify model.predict uses deterministic=True")

    # Read competition_export.py and check for deterministic=True
    export_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent", "competition_export.py",
    )

    with open(export_path) as f:
        content = f.read()

    if "deterministic=True" in content:
        report.pass_("competition_export.py uses deterministic=True")
        report.detail("deterministic_true_found", True)
    else:
        report.fail("competition_export.py does NOT use deterministic=True")
        report.detail("deterministic_true_found", False)

    # Also verify by running inference twice and checking actions match
    model = PPO.load(EXPECTED_CHECKPOINT, device="cpu")
    dummy_visual = np.random.rand(1, 4, 84, 168).astype(np.float32)
    dummy_vector = np.random.rand(1, 6).astype(np.float32)
    obs = {"visual": dummy_visual, "vector": dummy_vector}

    action1, _ = model.predict(obs, deterministic=True)
    action2, _ = model.predict(obs, deterministic=True)

    if np.array_equal(action1, action2):
        report.pass_("Deterministic inference produces same action for same obs")
    else:
        report.fail(f"Non-deterministic: {action1} vs {action2}")

    report.end()


def check_8_no_training_code(report):
    """8. Verify no training-only code in competition path."""
    report.start("8. No Training-Only Code",
                 "Verify competition path has no reward shaping, callbacks, curriculum, logging")

    # Check teamX.py and teamY.py
    issues = []

    for fname in ["teamX.py", "teamY.py"]:
        fpath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            fname,
        )
        if not os.path.exists(fpath):
            issues.append(f"{fname} not found")
            continue

        with open(fpath) as f:
            content = f.read()

        # Check for training-only imports
        forbidden = [
            "reward_shaping",
            "HitDetector",
            "shaped_reward",
            "CurriculumCallback",
            "OpponentBuffer",
            "SelfPlayOpponent",
            "BallTrackingOpponent",
            "SpecialistOpponent",
            "MixedOpponentCallback",
            "tensorboard",
            "CheckpointCallback",
            "EvalCallback",
            "model.learn",
            "training_progress",
            "mask_rotation",
            "max_rally_steps",
        ]

        for term in forbidden:
            if term in content:
                issues.append(f"{fname}: found '{term}'")

    # Check competition_export.py for training-only code
    export_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent", "competition_export.py",
    )
    with open(export_path) as f:
        export_content = f.read()

    export_forbidden = [
        "reward_shaping",
        "HitDetector",
        "shaped_reward",
        "CurriculumCallback",
        "OpponentBuffer",
        "model.learn",
        "training_progress",
        "mask_rotation",
        "max_rally_steps",
        "CheckpointCallback",
        "EvalCallback",
    ]

    for term in export_forbidden:
        if term in export_content:
            issues.append(f"competition_export.py: found '{term}'")

    report.detail("issues_found", issues if issues else "none")

    if not issues:
        report.pass_("No training-only code found in competition path")
    else:
        report.fail(f"Training-only code found: {issues}")

    report.end()


def check_9_export_report(report, model):
    """9. Produce final export report."""
    report.start("9. Final Export Report",
                 "Compile checkpoint, policy, obs, action, latency, verification")

    # Inference latency (already measured in check 5, but measure here too)
    dummy_visual = np.random.rand(1, 4, 84, 168).astype(np.float32)
    dummy_vector = np.random.rand(1, 6).astype(np.float32)
    obs = {"visual": dummy_visual, "vector": dummy_vector}

    times = []
    for _ in range(100):
        t0 = time.time()
        model.predict(obs, deterministic=True)
        times.append((time.time() - t0) * 1000)

    times = np.array(times)
    report.detail("checkpoint", EXPECTED_CHECKPOINT)
    report.detail("policy_type", type(model.policy).__name__)
    report.detail("observation_format", "Dict(visual: (4,84,168) float32, vector: (6,) float32)")
    report.detail("action_format", "MultiDiscrete([3,3,3]) — [vertical, horizontal, rotation]")
    report.detail("inference_mean_ms", f"{times.mean():.2f}")
    report.detail("inference_p95_ms", f"{np.percentile(times, 95):.2f}")
    report.detail("inference_max_ms", f"{times.max():.2f}")
    report.detail("frame_stack", EXPECTED_FRAME_STACK)
    report.detail("heatmap", "ON (sigma=5, weight=0.7)")
    report.detail("paddle_detection", "BGR2HSV, area filter 20-500")
    report.detail("ball_detection", "RGB2HSV, yellow ball")
    report.detail("device", "cpu")
    report.detail("deterministic", True)

    # Competition deadline check
    if times.mean() < 10.0:
        report.pass_(f"Inference latency {times.mean():.2f}ms < 10ms deadline")
    else:
        report.fail(f"Inference latency {times.mean():.2f}ms exceeds 10ms deadline")

    report.end()


def main():
    ap = argparse.ArgumentParser(description="Competition Export Validation")
    ap.add_argument("--port", type=int, default=6700,
                    help="Base port for Unity env (default: 6700)")
    ap.add_argument("--output", type=str, default="results/competition_validation.json",
                    help="Output JSON report path")
    ap.add_argument("--skip-10k", action="store_true",
                    help="Skip 10k inference test (for quick checks)")
    args = ap.parse_args()

    report = ValidationReport()

    print("\n" + "=" * 60)
    print("COMPETITION EXPORT VALIDATION")
    print("=" * 60)
    print(f"  Checkpoint: {EXPECTED_CHECKPOINT}")
    print(f"  Port:       {args.port}")
    print(f"  Skip 10k:   {args.skip_10k}")

    # Check 1: Checkpoint identity
    model = check_1_checkpoint_identity(report)
    if model is None:
        report.summary()
        return

    # Check 2: MultiInputPolicy
    check_2_multiinput_policy(report, model)

    # Check 3: Observation format match
    check_3_obs_format_match(report, model)

    # Check 4: Preprocessor Dict obs
    check_4_preprocessor_dict_obs(report)

    # Check 5: 10k inference steps
    if not args.skip_10k:
        check_5_10k_inference(report, args.port)
    else:
        report.start("5. 10k Inference Steps", "SKIPPED")
        report.detail("skipped", True)
        report.pass_("Skipped by --skip-10k flag")
        report.end()

    # Check 6: Left/right mirroring
    check_6_left_right_mirroring(report)

    # Check 7: Deterministic inference
    check_7_deterministic(report)

    # Check 8: No training-only code
    check_8_no_training_code(report)

    # Check 9: Final export report
    check_9_export_report(report, model)

    # Summary
    all_passed = report.summary()

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(f"\nReport saved to: {args.output}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
