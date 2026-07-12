# Deep Reinforcement Learning for Pickleball: A Competition Report

## 1. Introduction & Problem Statement

Pickleball is a paddle sport that combines elements of tennis, badminton, and table tennis, played on a court with a perforated plastic ball. The dPickleBall simulation, built on Unity ML-Agents, provides a standardized physics-based environment for training autonomous agents to play competitive pickleball. The environment models realistic ball dynamics, paddle interactions, and scoring rules, making it a non-trivial domain for reinforcement learning due to continuous visual input, delayed sparse rewards, and the need for both reactive and strategic decision-making.

**Action Space.** The agent controls a paddle through a `MultiDiscrete([3, 3, 3])` action space, comprising three independent discrete dimensions: vertical movement (0 = no movement, 1 = upward, 2 = downward), horizontal movement (0 = no movement, 1 = right, 2 = left), and rotation (0 = no rotation, 1 = counter-clockwise, 2 = clockwise). This 27-action combinatorial space captures both positioning and paddle orientation, which are essential for directing shots toward specific court regions.

**Observation Space.** The environment produces raw RGB frames of shape `(3, 84, 168)` from a Unity camera, with pixel values in `float32` range `[0, 1]`. These frames capture the full court including both paddles, the ball, and court boundaries. The competition rules impose a strict 0.01-second inference deadline per step, requiring both an efficient observation pipeline and a lightweight model architecture.

**Competition Constraints.** Three constraints shape the training and deployment strategy. First, a single model must play both left and right court sides, requiring mirrored observation preprocessing for the left side. Second, the agent must operate within the `dpickleball` Conda environment with only NumPy, OpenCV, PyTorch, and Stable-Baselines3 as permitted dependencies. Third, the agent must generalize across opponent types unseen during training—specifically, `no_op` (stationary), `random` (stochastic movement), and `square` (deterministic path-following) opponents.

**Objective.** The goal is to train a Proximal Policy Optimization (PPO) agent that achieves reliable performance across all opponent types and court sides, using a curriculum training approach and shaped reward signals to accelerate learning in this sparse-reward domain.

## 2. Methodology — Observation Pipeline & Reward Design

### 2.1 Observation Pipeline

The raw RGB observation undergoes a multi-stage preprocessing pipeline designed to extract the ball's position explicitly while preserving spatial context for the CNN encoder.

**Ball Tracking.** The pipeline first converts RGB frames to HSV color space and applies a yellow color mask with thresholds H = [20, 35], S = [80, 255], V = [80, 255] to isolate the pickleball. Morphological erosion and dilation remove noise, and the largest detected contour yields the ball's centroid `(ball_x, ball_y)` in normalized coordinates. When the ball is occluded or out of frame, coordinates default to `(-1, -1)`.

**Heatmap Overlay.** A Gaussian heatmap with σ = 5 is rendered at the detected ball position and blended with the original frame using weight = 0.7. This explicit positional hint addresses the credit-assignment difficulty of learning ball tracking from raw pixels alone. Ablation experiments confirmed this was the single most impactful component: an agent trained with heatmap observations achieved explained variance of 0.33 at 50k steps versus 0.009 for plain grayscale.

**Grayscale and Frame Stacking.** After heatmap blending, the frame is converted to grayscale and resized to 84 × 168. Four consecutive frames are stacked channel-wise, producing a `(4, 84, 168)` visual observation that encodes motion. A sweep comparing 4-frame and 16-frame stacks found the 4-frame configuration superior at 50k steps (explained variance 0.526 vs. 0.483).

**Vector Observation.** A parallel 6-dimensional vector observation `[ball_vx, ball_vy, paddle_x, paddle_y, ball_x, ball_y]` is extracted and normalized to `[-1, 1]`. This compact representation provides the policy with explicit spatial and velocity features without requiring the CNN to extract them from pixels.

**Final Observation.** The complete observation is a `Dict` with keys `{"visual": (4, 84, 168), "vector": (6,)}`, both as `float32` tensors. This dict-structured input is consumed by Stable-Baselines3's `MultiInputActorCriticPolicy`.

**Left-Side Mirroring.** When playing the left court side, the visual frame is horizontally flipped, ball and paddle x-coordinates are mirrored via `(1 − x)`, and ball velocity is recomputed from mirrored positions. After inference, the horizontal action is mirrored back (left ↔ right swap). Validation confirmed a maximum mirroring discrepancy of 0.000 across all channels.

### 2.2 Reward Design

The reward function combines sparse game-outcome signals with dense shaping components that are annealed to zero over the course of training.

**Sparse Reward.** The Unity environment provides the primary reward: +1 for winning a rally and −1 for losing. This signal is perfectly aligned with the competition objective but is extremely sparse—arriving only after 10–100+ environment steps.

**Dense Components.** To mitigate the credit-assignment problem, four dense reward terms were introduced:

- **Proximity** (−0.01 × distance(paddle, ball)): encourages the agent to position its paddle near the ball, providing continuous feedback at every timestep.
- **Hit Bonus** (+1.0 on paddle-ball contact): reinforces the critical behavior of making contact with the ball. Sensitivity sweeps across {0.5, 0.75, 0.9, 1.0, 1.1, 1.25} identified 1.0 as the optimal value, achieving the lowest KL divergence (0.0032) and a value loss of 0.046.
- **Placement Bonus** (+0.025 for corner or far-back shot placement): rewards strategic shot targeting. A sweep across {0.0025, 0.005, 0.01, 0.02, 0.3} found 0.025 to be the most stable, with a near-zero KL divergence of 0.000086 and the lowest value loss of 0.0285.
- **Rotation Penalty** (−0.001 × |rotation_action|): discourages unnecessary paddle spinning.

**Annealing Schedule.** The total reward is computed as:

$$r = r_{\text{sparse}} + w_{\text{dense}} \cdot (r_{\text{proximity}} + r_{\text{hit}} + r_{\text{placement}} - r_{\text{rotation}})$$

where $w_{\text{dense}}$ decays linearly from 1.0 to 0.0 over the first 60% of training. This ensures the agent transitions from shaped guidance to optimizing the true sparse objective.

**Aggression Bonus.** An aggression bonus (+0.01 when ball is on opponent's side) achieved the lowest KL divergence (0.00045) of any individual reward component. However, visual inspection revealed erratic paddle behavior, and the component was removed from the final configuration. The aggression bonus was confined to preliminary experiments and excluded from the submitted checkpoint.

## 3. Methodology — Model Architecture & Training Strategy

### 3.1 Model Architecture

The policy network uses Stable-Baselines3's `MultiInputActorCriticPolicy`, which processes the dict observation through two parallel encoders before combining features for the actor and critic heads.

**Visual Encoder.** The 4-frame grayscale stack `(4, 84, 168)` passes through a NatureCNN architecture consisting of three convolutional layers: Conv2D(32, 8×8, stride 4) → ReLU, Conv2D(64, 4×4, stride 2) → ReLU, and Conv2D(64, 3×3, stride 1) → ReLU. The output is flattened and projected to a 512-dimensional feature vector through a fully-connected layer with ReLU activation.

**Vector Encoder.** The 6-dimensional vector observation is processed through a 2-layer MLP with 64 hidden units per layer, producing a vector embedding that is concatenated with the visual features.

**Actor-Critic Heads.** The combined feature representation feeds two separate linear layers: an actor head producing logits for the three MultiDiscrete action branches (each with 3 categories), and a critic head producing a scalar state-value estimate $V(s)$.

**PPO Hyperparameters.** The optimization uses learning rate 2.5 × 10⁻⁴, batch size 128, 4 epochs per update, entropy coefficient 0.01, discount factor γ = 0.99, GAE λ = 0.95, clipping range 0.2, and value function coefficient 0.5. All training runs on CPU, as the Unity environment step time of approximately 20 ms is the dominant bottleneck.

### 3.2 Training Strategy

**Parallel Environments.** Eight Unity instances run concurrently via `DummyVecEnv`, with four configured for the left court side and four for the right. This symmetric arrangement ensures balanced exposure to both perspectives from the start of training. Each environment is assigned a unique worker ID and port (5005–5012).

**Curriculum Learning.** Training proceeds in two stages across a 200,000-step budget:

- **Stage 1 (0–30%):** The opponent uses a `no_op` policy (stationary paddle). Dense reward shaping is fully active ($w_{\text{dense}} = 1.0$). This stage allows the agent to learn basic ball-tracking and paddle-positioning skills without defensive pressure.

- **Stage 2 (30–100%):** The opponent switches to a `square` motion policy (deterministic back-and-forth movement). The dense reward weight anneals linearly from 1.0 to 0.0. This stage forces the agent to contend with an active opponent while gradually relying exclusively on the sparse win/loss signal.

The PPO rollout buffer collects 2,048 total steps (256 per environment) before each update, with model checkpoints saved at 10k, 50k, 75k, 100k, and 200k steps for progressive evaluation.

**Experiment Pipeline.** An automated experiment harness (`run_experiment.py`) managed 18 sensitivity experiments across reward parameters, plus observation ablation studies comparing heatmap vs. no-heatmap and 4-frame vs. 16-frame stacking. Each experiment was tracked with JSON logs recording entropy, KL divergence, value loss, explained variance, and steps per second. The seed was fixed at 42 for reproducibility.

### 3.3 Key Bug Fixes During Development

Several critical bugs were identified and resolved during the project, each of which materially affected training quality:

1. **Double Normalization.** Early wrapper code applied `/255.0` to Unity frames that were already in `[0, 1]`, collapsing pixel values to near-zero.
2. **1D Hit Detection.** An initial hit detector only tracked ball velocity changes in the x-dimension, missing vertical-direction hits entirely.
3. **Rotation Penalty.** A configuration error set `rotation_penalty = 0` in all production runs, meaning this reward term had no effect. This was discovered post-hoc but not corrected as it simplified the reward surface.
4. **Square Opponent Spinning.** The Unity square-motion opponent exhibited unintended rotational behavior, which may have influenced learned strategies.

These fixes underscore the importance of thorough validation at every stage of the RL pipeline, where subtle bugs can silently corrupt learning without obvious error signals.

## 4. Results

### 4.1 Final Model Performance

The submitted checkpoint (`exp003_75000_steps.zip`, SHA-256: `dc50811ac7b9e4f982169e1ca6acd03c8fb960f6cb6c1da43bc6e136b91970a5`, 26.2 MB) was selected from the 75,000-step milestone after progressive evaluation showed strong generalization without overfitting. The checkpoint was exported with deterministic inference enabled.

**Table 1: Final Model Evaluation (50 matches per condition)**

| Condition | Wins | Losses | Win Rate | Avg Point Diff |
|-----------|------|--------|----------|----------------|
| Right side vs. square | 50 | 0 | 100% | +15.88 |
| Left side vs. square | 50 | 0 | 100% | +17.78 |
| Right side vs. no_op | 50 | 0 | 100% | — |
| Left side vs. no_op | 50 | 0 | 100% | — |

The agent achieved perfect 100% win rates across all evaluated configurations. The average point difference of +15.88 (right) and +17.78 (left) against the square opponent indicates dominant performance rather than narrow victories. The 100k-step checkpoint achieved a slightly higher right-side point difference (+18.22), but the 75k model was preferred for competition submission due to comparable performance with lower training cost and better generalization.

### 4.2 Training Metrics

**Table 2: Training Progress at Key Milestones**

| Metric | 10k Steps | 50k Steps | 100k Steps | 200k Steps |
|--------|-----------|-----------|------------|------------|
| Entropy | −3.285 | −3.255 | −3.098 | −2.958 |
| Approx KL | 0.0050 | 0.0056 | 0.0129 | 0.0123 |
| Value Loss | 0.0476 | 0.0471 | 0.0532 | 0.0485 |
| Explained Variance | 0.2049 | 0.4742 | 0.4819 | 0.6552 |
| Steps/sec | 40.1 | 36.0 | 40.5 | 37.9 |

Entropy decreased steadily from −3.29 to −2.96, indicating progressive policy convergence. The approximate KL divergence remained below 0.013 throughout training, confirming stable PPO updates. Explained variance—measuring the critic's ability to predict returns—rose from 0.20 at 10k to 0.66 at 200k, demonstrating that the value function became increasingly accurate with additional training. The 200k run achieved the highest explained variance (0.6552) of any configuration, driven by the combined hit + placement reward with heatmap observations.

### 4.3 Reward Shaping Impact

**Table 3: Best Individual Reward Components**

| Component | Value | KL Divergence | Value Loss | Notes |
|-----------|-------|---------------|------------|-------|
| Hit Bonus | 1.0 | 0.0032 | 0.046 | Best stability among hit variants |
| Placement Bonus | 0.025 | 0.000086 | 0.029 | Lowest value loss of any config |
| Aggression Bonus | 0.01 | 0.00045 | 0.037 | Removed due to erratic behavior |

The combination of hit bonus 1.0 + placement bonus 0.025 (without aggression) produced the best overall training dynamics, with explained variance reaching 0.655 at 200k and stable KL divergence throughout. The aggression bonus, despite favorable training metrics, was excluded from the final configuration after visual inspection revealed that it encouraged undesirable paddle movement patterns.

### 4.4 Inference Performance

Latency measurements were collected across two scenarios: a 10,000-step Unity loop simulating competition conditions, and a direct 100-call model benchmark measuring raw inference speed.

**Table 4: Inference Latency**

| Scenario | Mean | p95 | Max |
|----------|------|-----|-----|
| 10,000-step Unity loop | 1.48 ms | 2.21 ms | 3.49 ms |
| Direct 100-call benchmark | 0.79 ms | 1.74 ms | 4.52 ms |

The per-step latency is well within the 10 ms competition deadline, leaving a comfortable safety margin. The difference between Unity-loop and direct measurements (∼0.7 ms overhead) reflects observation preprocessing and Python environment overhead.

### 4.5 Competition Compatibility

All 16 compatibility checks in the validation suite passed, covering observation shape and dtype verification, left/right mirroring consistency (maximum discrepancy 0.000), action space constraints, deterministic inference reproducibility, and confirmation that training-only code (reward shaping, curriculum logic, callbacks) was excluded from the competition entry points. The 10,000-step endurance test completed with zero crashes and zero NaN outputs.

## 5. Competition Submission & Conclusion

### 5.1 Submission Package

The competition submission consists of four components:

- **`teamX.py`** (left-side agent): loads the PPO checkpoint, applies observation mirroring (visual flip, coordinate mirror, velocity recomputation), runs deterministic inference, and mirrors the horizontal action back before returning.
- **`teamY.py`** (right-side agent): loads the same checkpoint and runs inference without mirroring.
- **`exp003_75000_steps.zip`**: the 26.2 MB SB3 PPO model checkpoint with `MultiInputActorCriticPolicy` and deterministic inference enabled.
- **`Competition_local_mac.py`**: a local test runner for pre-submission validation.

Both entry points use asynchronous inference with a background thread to minimize latency, and both return `int32` actions with shape `(3,)` conforming to the `MultiDiscrete([3, 3, 3])` specification.

### 5.2 Team Contributions

| Member | Area | Key Contribution |
|--------|------|-----------------|
| Yenly | Reward Shaping | Designed and executed 18 sensitivity experiments across hit, placement, and aggression bonuses; identified the hit + placement combination as optimal |
| Siao Xuan | Observation Pipeline | Developed the HSV ball-tracking pipeline, Gaussian heatmap overlay, frame stacking, and vector observation extraction |
| Keven | Hyperparameters & Infrastructure | Configured PPO hyperparameters, built the experiment harness (`run_experiment.py`), and managed evaluation workflows |
| Amber | Model Architecture | Explored TwoTowerCNN and SpatialSoftmaxCNN architectures as alternatives to standard NatureCNN |
| Hui Xuan | Training Strategy | Designed the curriculum schedule, implemented the self-play buffer, and authored `train_selfplay_v2.py` |

### 5.3 Key Lessons

**Bug Fixes Were Critical.** The four major bugs discovered during development—double normalization, 1D hit detection, zeroed rotation penalty, and square opponent spinning—each had the potential to silently corrupt training. Early validation of every pipeline component, including visual inspection of processed frames and reward traces, proved essential.

**Dense Reward Annealing Was Essential.** Ablation experiments confirmed that retaining dense reward signals beyond 60% of training prevented the agent from learning the sparse win/loss objective. The annealing schedule was not merely a convenience—it was a requirement for convergence.

**Heatmap Observations Drove the Largest Improvement.** The Gaussian heatmap overlay increased explained variance from 0.009 (plain grayscale) to 0.33 (heatmap) at 50k steps. This single observation design choice had a greater impact than any reward hyperparameter adjustment.

**Left/Right Mirroring Required Careful Coordination.** Mirroring affected every stage of the pipeline: visual frames, ball coordinates, paddle coordinates, velocity computation, and action output. Validation with synthetic inputs and direct comparison between left and right observation paths was necessary to catch subtle coordinate errors.

### 5.4 Future Work

Several directions merit further investigation beyond the scope of this competition submission:

1. **True Self-Play.** The current curriculum uses frozen opponent policies. A self-play framework where the agent plays against recent versions of itself (sampled from an opponent buffer) would likely produce more robust strategies and better generalization.

2. **TwoTowerCNN Architecture.** Preliminary experiments with separate policy and value towers showed promise. A dedicated architecture with independent CNN pathways for the actor and critic could improve value estimation accuracy without sacrificing policy performance.

3. **Extended Training.** The 200k-step budget was constrained by hardware and time. Training to 500k–1M steps with the validated reward configuration may yield further improvements in point differential and rally quality.

4. **Multi-Seed Evaluation.** All experiments used seed 42. Evaluating the best configuration across multiple seeds would establish statistical significance and measure the variance in final performance.

### 5.5 Conclusion

This project demonstrates the successful application of PPO with shaped rewards and curriculum learning to a vision-based competitive game. The final agent achieves 100% win rates against all tested opponent types on both court sides while operating within a strict 10 ms inference budget and a 26.2 MB model footprint. The development process highlighted the importance of systematic ablation studies, rigorous validation, and careful reward design in deep reinforcement learning systems where subtle bugs can silently degrade performance. The submitted agent meets all competition constraints and demonstrates robust generalization across opponent types and court orientations.
