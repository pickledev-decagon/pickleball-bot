EXPERIMENTS = {

    "baseline": {
        "name": "Baseline v2",
        "description": "Current default: grayscale, 4-stack, entropy 0.01, curriculum, proximity reward",
        "ppo_kwargs": {
            "ent_coef": 0.01,
            "batch_size": 128,
        },
        "env_kwargs": {
            "use_rgb": False,
            "use_frame_diff": False,
            "use_heatmap": False,
            "frame_stack": 4,
        },
        "reward_kwargs": {
            "aggression_bonus": 0.0,
            "proximity_weight": 1.0,
            "hit_bonus": 0.5,
        },
        "curriculum": True,
    },

    "entropy_0.0": {
        "name": "No Entropy",
        "description": "ent_coef=0.0 — no exploration bonus, may converge faster or get stuck",
        "ppo_kwargs": {"ent_coef": 0.0},
        "parent": "baseline",
    },

    "entropy_0.05": {
        "name": "High Entropy",
        "description": "ent_coef=0.05 — maximum exploration",
        "ppo_kwargs": {"ent_coef": 0.05},
        "parent": "baseline",
    },

    "rgb_obs": {
        "name": "RGB Observation",
        "description": "3-channel RGB input instead of grayscale — more color info, 3x channels",
        "env_kwargs": {"use_rgb": True},
        "parent": "baseline",
    },

    "frame_diff": {
        "name": "Frame Difference",
        "description": "Stack frame-to-frame differences instead of raw frames — motion pops out",
        "env_kwargs": {"use_frame_diff": True},
        "parent": "baseline",
    },

    "heatmap_obs": {
        "name": "Ball Heatmap",
        "description": "Ball position (x,y) overlaid as Gaussian peaks on stacked grayscale",
        "env_kwargs": {"use_heatmap": True},
        "parent": "baseline",
    },

    "aggression_bonus": {
        "name": "Aggression Bonus",
        "description": "+0.1 per frame ball is on opponent side — encourage attacking play",
        "reward_kwargs": {"aggression_bonus": 0.1},
        "parent": "baseline",
    },

    "no_curriculum": {
        "name": "No Curriculum",
        "description": "Skip stage 1-2, straight to self-play opponent buffer",
        "curriculum": False,
        "parent": "baseline",
    },

    "large_batch": {
        "name": "Large Batch (256)",
        "description": "batch_size=256 for more stable gradient estimates",
        "ppo_kwargs": {"batch_size": 256},
        "parent": "baseline",
    },

    "small_batch": {
        "name": "Small Batch (64)",
        "description": "batch_size=64 for faster but noisier updates",
        "ppo_kwargs": {"batch_size": 64},
        "parent": "baseline",
    },

    "hit_bonus_high": {
        "name": "High Hit Bonus",
        "description": "+1.0 per paddle contact (up from 0.5) — reinforce hitting",
        "reward_kwargs": {"hit_bonus": 1.0},
        "parent": "baseline",
    },

    "lr_1e-4": {
        "name": "Low Learning Rate",
        "description": "lr=0.0001 — slower, more stable learning",
        "ppo_kwargs": {"learning_rate": 0.0001},
        "parent": "baseline",
    },

    "stack_2": {
        "name": "2-Frame Stack",
        "description": "Only 2 stacked frames — faster inference, less temporal info",
        "env_kwargs": {"frame_stack": 2},
        "parent": "baseline",
    },

    "stack_16": {
        "name": "16-Frame Stack",
        "description": "16 stacked frames — 400ms temporal context for wall-bounce prediction",
        "env_kwargs": {"frame_stack": 16},
        "parent": "baseline",
    },

    "stack_32": {
        "name": "32-Frame Stack",
        "description": "32 stacked frames — 800ms history, matches official example depth",
        "env_kwargs": {"frame_stack": 32},
        "parent": "baseline",
    },

    "placement_bonus": {
        "name": "Shot Placement Reward",
        "description": "+reward for landing ball in corner/far-back zones — teaches shot quality",
        "reward_kwargs": {"placement_bonus": 0.3},
        "parent": "baseline",
    },

    "heatmap_obs_stack16": {
        "name": "Ball Heatmap + 16-Frame Stack",
        "description": "Heatmap obs with 16-frame stack — P0 baseline",
        "env_kwargs": {
            "use_heatmap": True,
            "frame_stack": 16,
        },
        "parent": "baseline",
    },

    "heatmap_stack_32": {
        "name": "Ball Heatmap + 32-Frame Stack",
        "description": "Heatmap obs with 32-frame stack — P1",
        "env_kwargs": {
            "use_heatmap": True,
            "frame_stack": 32,
        },"parent": "baseline",
    },

    "heatmap_stack_24": {
        "name": "Ball Heatmap + 24-Frame Stack",
        "description": "Heatmap with 24-frame stack — middle ground between 16 and 32",
        "env_kwargs": {
            "use_heatmap": True, 
            "frame_stack": 24},
        "parent": "baseline",
    },

    "heatmap_weight_03": {
        "name": "Heatmap Weight 0.3",
        "description": "Subtle heatmap blend — 0.3 weight",
        "env_kwargs": {"use_heatmap": True, 
                       "frame_stack": 16, 
                       "heatmap_weight": 0.3},
        "parent": "baseline",
    },

    "heatmap_weight_07": {
        "name": "Heatmap Weight 0.7",
        "description": "Strong heatmap blend — 0.7 weight",
        "env_kwargs": {"use_heatmap": True, 
                       "frame_stack": 16, 
                       "heatmap_weight": 0.7},
        "parent": "baseline",
    },

    "heatmap_weight_10": {
        "name": "Heatmap Weight 1.0",
        "description": "Maximum heatmap blend — 1.0 weight",
        "env_kwargs": {"use_heatmap": True, 
                       "frame_stack": 16, 
                       "heatmap_weight": 1.0},
        "parent": "baseline",
    },

    "heatmap_w07_s3": {
        "name": "Heatmap Tight Blob (weight=0.7, sigma=3, stack=16)",
        "description": "Tighter blob with best weight — sigma=3 showed 0.238 at 10k envs4",
        "env_kwargs": {
            "use_heatmap": True,
            "frame_stack": 16,
            "heatmap_weight": 0.7,
            "heatmap_sigma": 3,
        },
        "parent": "baseline",
    },

    "heatmap_w07_s4": {
        "name": "Heatmap Mid Blob (weight=0.7, sigma=4, stack=16)",
        "description": "Between sigma=3 and sigma=5 —  middle ground",
        "env_kwargs": {
            "use_heatmap": True,
            "frame_stack": 16,
            "heatmap_weight": 0.7,
            "heatmap_sigma": 4,
        },
        "parent": "baseline",
    },

    "heatmap_w07_s6": {
        "name": "Heatmap Mid Blob (weight=0.7, sigma=6, stack=16)",
        "description": "Mid size gaussian blob — sigma=6 showed 0.218 at 10k envs4",
        "env_kwargs": {"use_heatmap": True, 
                       "frame_stack": 16, 
                       "heatap_weight": 0.7,
                       "heatmap_sigma": 6},
        "parent": "baseline",
    },

    "heatmap_w07_s9": {
        "name": "Heatmap Large Blob 9 (weight=0.7, sigma=9, stack=16)",
        "description": "Very large gaussian blob — sigma=9 showed 0.117 at 10k envs4",
        "env_kwargs": {"use_heatmap": True, 
                       "frame_stack": 16, 
                       "heatmap_weight": 0.7,
                       "heatmap_sigma": 9},
        "parent": "baseline",
    },

    "heatmap_w07_s5": {
        "name": "Heatmap Best (weight=0.7, sigma=5, stack_16)",
        "description": "best confirmed combo from sweeps - w-0.7, s=5",
        "env_kwargs": {"use_heatmap": True, 
                       "frame_stack": 16, 
                       "heatmap_weight": 0.7,
                       "heatmap_sigma": 5},
        "parent": "baseline",
    },

    
}

def resolve_config(name):
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown config: {name}. Choose from: {list(EXPERIMENTS.keys())}")

    config = EXPERIMENTS[name]
    if "parent" in config:
        parent = resolve_config(config["parent"])
        merged = dict(parent)
        for key in ("ppo_kwargs", "env_kwargs", "reward_kwargs"):
            if key in config:
                merged[key] = {**merged.get(key, {}), **config[key]}
        for key in ("curriculum", "description", "name"):
            if key in config:
                merged[key] = config[key]
        return merged
    return dict(config)
