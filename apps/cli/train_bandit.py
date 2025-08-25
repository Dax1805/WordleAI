from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from packages.agent.env import WordleBanditEnv
from packages.agent.features import make_features
from packages.agent.bandit_linucb import LinUCB


def main():
    ap = argparse.ArgumentParser(description="Train LinUCB bandit to pick solvers per turn.")
    ap.add_argument("--answers", required=True)
    ap.add_argument("--allowed", required=True)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--alpha-time", type=float, default=0.2,
                    help="compute-time weight in per-turn reward")
    ap.add_argument("--ucb-alpha", type=float, default=0.5, help="LinUCB exploration strength")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--actions", nargs="*",
                    default=["positional_freq", "expected_left", "max_patterns", "letter_freq"])
    ap.add_argument("--outdir", default="reports/bandit_train")
    args = ap.parse_args()

    outdir = Path(args.outdir);
    outdir.mkdir(parents=True, exist_ok=True)

    # Warm-up env to get feature dimension
    env = WordleBanditEnv(answers_path=args.answers, allowed_path=args.allowed, N=args.N,
                          seed=args.seed, actions=args.actions, alpha_time=args.alpha_time)
    obs = env.reset(seed=args.seed)
    d = len(obs["features"])

    bandit = LinUCB(actions=args.actions, d=d, alpha=args.ucb_alpha, seed=args.seed)

    total_reward = 0.0
    wins = 0
    steps_total = 0
    time_ms_total = 0.0

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        last_answer = None
        while not done:
            x = np.array(obs["features"], dtype=float)
            action = bandit.select(x)
            obs, r, done, info = env.step(action)
            bandit.update(action, x, r)
            total_reward += r
            steps_total += 1
            time_ms_total += float(info["time_ms"])
            if done and info.get("answer") is not None and info["pattern"] == "G" * args.N:
                wins += 1
                last_answer = info["answer"]

        if ep % 1000 == 0:
            print(f"[ep {ep}] avg_reward/step={total_reward / steps_total:.4f} "
                  f"win_rate={wins / ep:.3f} avg_time_ms/step={time_ms_total / steps_total:.2f}")

    # Save model + metadata
    model_path = outdir / "linucb_model.json"
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(bandit.to_json())
    meta = {
        "episodes": args.episodes,
        "N": args.N,
        "actions": args.actions,
        "alpha_time": args.alpha_time,
        "ucb_alpha": args.ucb_alpha,
        "answers": str(Path(args.answers)),
        "allowed": str(Path(args.allowed)),
        "seed": args.seed,
        "avg_reward_per_step": total_reward / steps_total if steps_total else None,
        "train_steps": steps_total,
        "win_rate_estimate": wins / args.episodes,
        "avg_time_ms_per_step": time_ms_total / steps_total if steps_total else None,
        "model_path": str(model_path),
    }
    with open(outdir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model -> {model_path}")


if __name__ == "__main__":
    main()
