"""Quick eval of trained model: both sides vs no_op, random, square."""
import sys, time
sys.path.insert(0, '.')
from stable_baselines3 import PPO
from agent.eval_harness import MatchEvaluator
from agent.wrapper_vec import make_unity_env_fn
from config import BUILD_PATH

CHECKPOINT = "results/hit_placement_heatmap_20260624_165154.zip"
N_MATCHES = 20  # ~4.5 min/match — adjust based on available time

model = PPO.load(CHECKPOINT, device="cpu")

opponents = ["no_op", "random", "square"]

results = {}
for side_label, play_as_left in [("right", False), ("left", True)]:
    results[side_label] = {}
    port = 7005 if play_as_left else 5005
    rank = 99 if play_as_left else 0

    env_fn = make_unity_env_fn(
        rank=rank, base_port=port, build_path=BUILD_PATH,
        frame_stack=16, use_heatmap=True, play_as_left=play_as_left,
    )
    evaluator = MatchEvaluator(model, env_fn)

    for opp in opponents:
        print(f"\n  [{side_label.upper()} side vs {opp}] ...", flush=True)
        start = time.time()
        r = evaluator.evaluate_vs(opp, n_matches=N_MATCHES)
        elapsed = time.time() - start
        results[side_label][opp] = r
        print(f"    Win: {r['win_rate']*100:.0f}% ({r['wins']}W {r['losses']}L {r['draws']}D) "
              f"pt_diff: {r['avg_point_diff']}  time: {elapsed:.0f}s")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for side in ["right", "left"]:
    print(f"\n--- {side.upper()} SIDE ---")
    for opp in opponents:
        r = results[side][opp]
        print(f"  vs {opp:8s}: {r['win_rate']*100:5.1f}%  ({r['wins']:2d}W {r['losses']:2d}L {r['draws']:2d}D)  "
              f"pt_diff: {r['avg_point_diff']:+5.1f}")
