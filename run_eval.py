
from stable_baselines3 import PPO
from agent.eval_harness import MatchEvaluator
from agent.wrapper_vec import make_unity_env_fn
from config import BUILD_PATH

CHECKPOINT = "results/hit_aggression_placement_combo_heatmap_20260619_015059.zip"
N_MATCHES = 50  # quick diagnostic

model = PPO.load(CHECKPOINT, device="cpu")

right_fn = make_unity_env_fn(
    rank=0, build_path=BUILD_PATH,
    frame_stack=16, use_heatmap=True,
    play_as_left=False,
)

evaluator = MatchEvaluator(model, right_fn)

opponents = ["no_op", "random", "square"]
#opponents = ["square"]
results = {}

for opp in opponents:
    print(f"\n=== RIGHT SIDE vs {opp} ===")
    r = evaluator.evaluate_vs(opp, n_matches=N_MATCHES, env_fn=right_fn)
    results[opp] = r
    print(f"Win rate: {r['win_rate']*100:.1f}%  "
          f"({r['wins']}W {r['losses']}L {r['draws']}D)  "
          f"avg point diff: {r['avg_point_diff']}")

print("\n=== SUMMARY ===")
print(f"vs no_op  : {results['no_op']['win_rate']*100:.1f}%")
print(f"vs random : {results['random']['win_rate']*100:.1f}%")
print(f"vs square : {results['square']['win_rate']*100:.1f}%")

