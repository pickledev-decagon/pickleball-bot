import os
import json
import numpy as np


class OpponentBuffer:
    def __init__(self, save_dir, max_size=5):
        self.save_dir = save_dir
        self.max_size = max_size
        self.checkpoints = []
        os.makedirs(save_dir, exist_ok=True)
        self._load_manifest()

    def _manifest_path(self):
        return os.path.join(self.save_dir, "manifest.json")

    def _load_manifest(self):
        path = self._manifest_path()
        if os.path.exists(path):
            with open(path) as f:
                self.checkpoints = json.load(f)
        else:
            self.checkpoints = []

    def _save_manifest(self):
        with open(self._manifest_path(), "w") as f:
            json.dump(self.checkpoints, f, indent=2)

    def add(self, name, win_rate=0.0):
        self.checkpoints.append({"name": name, "win_rate": win_rate})
        self.checkpoints.sort(key=lambda x: x["win_rate"], reverse=True)
        while len(self.checkpoints) > self.max_size:
            removed = self.checkpoints.pop()
            path = os.path.join(self.save_dir, removed["name"])
            if os.path.exists(path):
                os.remove(path)
        self._save_manifest()

    def sample(self, exclude=None):
        if not self.checkpoints:
            return None
        candidates = [c for c in self.checkpoints if c["name"] != exclude]
        if not candidates:
            candidates = self.checkpoints
        import random
        return random.choice(candidates)["name"]

    def get_best(self):
        if not self.checkpoints:
            return None
        return self.checkpoints[0]["name"]

    def get_worst(self):
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]["name"]

    def list_checkpoints(self):
        return [c["name"] for c in self.checkpoints]

    def __len__(self):
        return len(self.checkpoints)

    def __repr__(self):
        return f"OpponentBuffer({len(self.checkpoints)} checkpoints, dir={self.save_dir})"
