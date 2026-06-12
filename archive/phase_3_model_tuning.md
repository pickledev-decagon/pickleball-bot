# Phase 3: Model Tuning

## Members E+F (parallel with Phase 2)

| Workstream | Owner | File(s) | Est. Effort |
|------------|-------|---------|-------------|
| TwoTower CNN policy | Member E | `agent/two_tower_cnn.py` | 2-3 hrs |
| Grid search script | Member F | `scripts/grid_search.py` | 3-4 hrs |
| LSTM tradeoff doc | Member E (2nd) | `docs/architecture_decisions.md` | 1 hr |
| Frame stack extension | Member F (2nd) | Update `experiment_configs.py` | 30 min |

---

## 1. TwoTower CNN Policy

### Architecture

Separate CNN encoders for policy and value prevent value loss gradient from
corrupting policy features (Cobbe et al., 2019):

```
obs
  ├─→ Policy CNN ─→ MLP ─→ action dist
  └─→ Value CNN  ─→ MLP ─→ V(s)
```

### Implementation

Use SB3's `share_features_extractor=False`:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import torch


class TwinNatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn_policy = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn_value = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, *observation_space.shape[1:])
            flat_dim = self.cnn_policy(dummy).shape[1]

        self.linear_policy = nn.Linear(flat_dim, features_dim)
        self.linear_value = nn.Linear(flat_dim, features_dim)

    def forward(self, observations):
        policy_feat = self.linear_policy(self.cnn_policy(observations))
        value_feat = self.linear_value(self.cnn_value(observations))
        return torch.cat([policy_feat, value_feat], dim=-1)


class TwoTowerCnnPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        from stable_baselines3.common.torch_layers import MlpExtractor
        self.mlp_extractor = MlpExtractor(
            self.features_dim * 2,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
```

### Usage

```python
model = PPO(
    TwoTowerCnnPolicy,
    vec_env,
    policy_kwargs={
        "features_extractor_class": TwinNatureCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [dict(pi=[128], vf=[128])],
        "share_features_extractor": False,
    },
)
```

### Alternative: Spatial Softmax CNN

```python
class SpatialSoftmaxCNN(BaseFeaturesExtractor):
    def forward(self, obs):
        feat_maps = self.cnn(obs)
        B, C, H, W = feat_maps.shape
        flat = feat_maps.view(B, C, H * W)
        weights = torch.softmax(flat, dim=-1)
        pos = weights @ self.position_grid.to(obs.device)
        return pos.view(B, C * 2)
```

---

## 2. Grid Search Script

### Structure

```
scripts/
  grid_search.py
  sweep_configs/
    phase3_default.json
    phase3_focused.json
```

### Sweep dimensions (after top-2 configs selected)

| Parameter | Values |
|-----------|--------|
| ent_coef | [0.0, 0.003, 0.01, 0.03] |
| learning_rate | [1e-4, 2.5e-4, 5e-4, 1e-3] |
| n_steps | [128, 256, 512, 1024] |
| batch_size | [64, 128, 256] |
| clip_range | [0.1, 0.15, 0.2, 0.3] |
| n_epochs | [3, 4, 8] |

2304 combos full factorial → use **random search** (30-50 trials).

### JSON sweep config

```json
{
  "name": "phase3_default",
  "base_config": "baseline",
  "n_trials": 30,
  "steps_per_trial": 20000,
  "envs": 2,
  "params": {
    "ppo_kwargs.ent_coef": {"type": "loguniform", "min": 0.0, "max": 0.05},
    "ppo_kwargs.learning_rate": {"type": "loguniform", "min": 1e-4, "max": 1e-3},
    "ppo_kwargs.n_steps": {"type": "choice", "values": [128, 256, 512, 1024]},
    "ppo_kwargs.batch_size": {"type": "choice", "values": [64, 128, 256]},
    "ppo_kwargs.clip_range": {"type": "choice", "values": [0.1, 0.15, 0.2, 0.3]},
    "ppo_kwargs.n_epochs": {"type": "choice", "values": [3, 4, 8]}
  }
}
```

### Output

```
results/sweeps/phase3_default/
  trial_000.json ... trial_029.json
  summary.csv  (collated)
  best.json    (best config)
```

---

## 3. LSTM / RecurrentPPO Tradeoff

| Aspect | Frame Stack | LSTM | Winner |
|--------|-------------|------|--------|
| Temporal horizon | Fixed 4 frames | Infinite | LSTM |
| Training speed | Fast | ~2× slower | Frame stack |
| SB3 support | Native | sb3_contrib | Frame stack |
| Competition inference | Stateless | Stateful | Frame stack |
| Ball occlusion | No | Yes | LSTM |

**Decision: Skip LSTM.** Frame stack captures enough temporal info.
Speed penalty and stateful inference outweigh benefits.

```python
# If revisiting:
from sb3_contrib import RecurrentPPO
model = RecurrentPPO("CnnLstmPolicy", vec_env, ...)
```

---

## 4. Frame Stack Depth Extension

Add to `experiment_configs.py`:

```python
"stack_8": {
    "name": "8-Frame Stack",
    "description": "8 stacked grayscale frames — more temporal context",
    "env_kwargs": {"frame_stack": 8},
    "parent": "baseline",
},
"stack_16": {
    "name": "16-Frame Stack",
    "description": "16 stacked frames — maximum temporal info",
    "env_kwargs": {"frame_stack": 16},
    "parent": "baseline",
},
```

---

## When to Start Phase 3

After RESULTS.md analysis:
1. All 13 JSONs collected
2. Top-2 configs by ep_rew_mean × stability
3. TwoTower code passes 2k-step smoke test
4. Grid sweep config ready

```bash
python scripts/grid_search.py sweep_configs/phase3_focused.json
# 30 trials × 20k steps × 4 envs ≈ 30-60 min on M5 Pro
```
