"""
TwoTower CNN Policy for dPickleBall.

Design: separate visual encoders for policy and value streams prevent
value-loss gradients from corrupting policy features (Cobbe et al., 2019).

Can be extended to a true multi-modal two-tower (visual + vector/RAM)
when the environment exposes structured state features.

Usage:
    from agent.two_tower_cnn import create_two_tower_ppo
    model = create_two_tower_ppo(vec_env, learning_rate=2.5e-4, ...)
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class TwinNatureCNN(BaseFeaturesExtractor):
    """
    Separate CNN encoders for policy and value.

    Output dimension = 2 * features_dim because policy_feat and value_feat
    are concatenated. ActorCriticPolicy must account for this via
    _build_mlp_extractor.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Policy tower
        self.cnn_policy = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Value tower
        self.cnn_value = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened dimension from a dummy forward pass
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
    """
    ActorCriticPolicy using TwinNatureCNN with separate policy/value MLP heads.

    features_dim is the per-tower dimension (e.g. 256).
    The actual extractor output is 2 * features_dim, so _build_mlp_extractor
    doubles the input size to the MLP extractor.
    """

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


class SpatialSoftmaxCNN(BaseFeaturesExtractor):
    """
    Alternative CNN that uses Spatial Softmax over feature maps.

    Instead of flattening conv outputs, computes weighted spatial coordinates
    for each feature map channel. This yields a compact, translation-aware
    representation that can be useful when object location is the key signal.

    Reference: Levine et al., "End-to-End Training of Deep Visuomotor Policies"
    """

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, *observation_space.shape[1:])
            feat = self.cnn(dummy)
            _, C, H, W = feat.shape

        # Register fixed position grid for spatial softmax
        grid = self._build_grid(H, W)
        self.register_buffer("position_grid", grid)

        # After spatial softmax: C channels * 2 coords each = C*2
        self.projection = nn.Linear(C * 2, features_dim)

    def _build_grid(self, h, w):
        y = torch.linspace(-1, 1, h)
        x = torch.linspace(-1, 1, w)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=0)  # (2, H, W)

    def forward(self, observations):
        feat_maps = self.cnn(observations)
        B, C, H, W = feat_maps.shape
        flat = feat_maps.view(B, C, H * W)
        weights = torch.softmax(flat, dim=-1)

        # position_grid: (2, H, W) -> (H*W, 2)
        grid = self.position_grid.view(2, H * W).transpose(0, 1)

        # weighted sum: (B, C, H*W) @ (H*W, 2) -> (B, C, 2)
        pos = torch.matmul(weights, grid)
        return self.projection(pos.view(B, C * 2))


def create_two_tower_ppo(vec_env, **ppo_kwargs):
    """
    Convenience factory: create a PPO model with TwoTowerCnnPolicy.

    Default net_arch: small separate heads suitable for dPickleBall's
    3-branch discrete action space.
    """
    policy_kwargs = {
        "features_extractor_class": TwinNatureCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [dict(pi=[128], vf=[128])],
        "share_features_extractor": False,
    }
    return PPO(
        TwoTowerCnnPolicy,
        vec_env,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs,
    )


def create_spatial_softmax_ppo(vec_env, **ppo_kwargs):
    """
    Alternative factory using SpatialSoftmaxCNN.
    """
    policy_kwargs = {
        "features_extractor_class": SpatialSoftmaxCNN,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [dict(pi=[128], vf=[128])],
        "share_features_extractor": False,
    }
    return PPO(
        ActorCriticPolicy,
        vec_env,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs,
    )
