from stable_baselines3 import PPO
from agent.eval_harness import MatchEvaluator
from agent.wrapper_vec import make_unity_env_fn
from config import BUILD_PATH

CHECKPOINT = "results/hit_placement_heatmap_20260624_021042.zip"
N_MATCHES = 5

model = PPO.load(CHECKPOINT, device="cpu")

left_fn = make_unity_env_fn(
    rank=1,
    base_port=7005,
    build_path=BUILD_PATH,
    frame_stack=16,
    use_heatmap=True,
    play_as_left=True,
)

evaluator = MatchEvaluator(model, left_fn)

print("\n=== LEFT SIDE vs square ===")
print("Using worker rank=1, base_port=7005, expected port=7006")

r = evaluator.evaluate_vs("square", n_matches=N_MATCHES)

print(f"Win rate: {r['win_rate']*100:.1f}% "
      f"({r['wins']}W {r['losses']}L {r['draws']}D) "
      f"avg point diff: {r['avg_point_diff']}")

print("\n=== SUMMARY ===")
print(f"Left side vs square: {r['win_rate']*100:.1f}% "
      f"({r['wins']}W {r['losses']}L {r['draws']}D), "
      f"avg point diff: {r['avg_point_diff']}")