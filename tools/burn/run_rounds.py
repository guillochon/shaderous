"""DAgger rounds on one GPU until a deadline, three training variants per round run side by side
(a small network leaves the GPU mostly idle: three jobs at once each ran about 15% slower than
one alone), the best of them, scored in float16 (score.py), the parent of the next round. A round
whose variants all score worse than their parent keeps the parent: repeated fine-tuning drifted
before (rounds 26 and 29).

    python run_rounds.py first_round deadline_epoch_s model.pt [model.pt ...]

The models given are scored and the best is the first parent. Variants (train.py's last two
arguments, the overshoot penalty and the temperature-slope weight):
  a  the rounds 27-29 recipe (teacher distillation, weight averaging)
  b  a + the overshoot penalty (ow 20)
  c  b + the temperature slope (jw 1; needs tpairs.npz)
Each round writes round_<r>.json: every variant's scores, the choice, and the time it took.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
VARIANTS = {"a": ("0", "0"), "b": ("20", "0"), "c": ("20", "1")}


def metric(s):
    """What the simulator sees: the worst 1% of general rollouts, averaged over the 4 times."""
    return sum(s["general_p99"]) / len(s["general_p99"])


def run_all(cmds, logs, env=None):
    procs = [subprocess.Popen(c, cwd=HERE, stdout=open(HERE / log, "w"), stderr=subprocess.STDOUT, env=env)
             for c, log in zip(cmds, logs)]
    return [p.wait() for p in procs]


def score_all(models, tag):
    outs = [f"score_{tag}_{Path(m).stem}.json" for m in models]
    run_all([[PY, "score.py", m, "--json", o] for m, o in zip(models, outs)], [o.replace(".json", ".log") for o in outs])
    scores = {}
    for m, o in zip(models, outs):
        try:
            scores[m] = json.load(open(HERE / o))[0]
        except (OSError, ValueError, IndexError):
            print(f"no score for {m}", flush=True)
    return scores


def main():
    r, deadline = int(sys.argv[1]), float(sys.argv[2])
    start = score_all(sys.argv[3:], "start")
    best = min(start, key=lambda m: metric(start[m]))
    best_score = start[best]
    print(f"start: {', '.join(f'{m} {metric(s):.3f}' for m, s in start.items())}; parent {best}", flush=True)
    last = 0.0
    while time.time() + last < deadline:
        t0 = time.time()
        subprocess.run([PY, "dagger.py", best, str(r)], cwd=HERE, stdout=open(HERE / f"dagger_{r}.log", "w"), stderr=subprocess.STDOUT)
        env = dict(os.environ, BURN_NO_ROLLOUT="1")
        models, cmds = [], []
        for i, (v, (ow, jw)) in enumerate(VARIANTS.items()):
            if jw != "0" and not (HERE / "tpairs.npz").exists():
                continue
            m = f"model_dagger_{r}{v}.pt"
            models.append(m)
            cmds.append([PY, "train.py", "30000", "128", "4", "4", m, "0.3", "0.2", "flow2f", "1", best, "0", "5e-4",
                         "1", "0.3", str(10 * r + i), "aprox13_surrogate.pt", "1", ow, jw])
        codes = run_all(cmds, [f"train_dagger_{Path(m).stem[len('model_dagger_'):]}.log" for m in models], env)
        models = [m for m, c in zip(models, codes) if c == 0 and (HERE / m).exists()]
        scores = score_all(models, str(r))
        pick = min(scores, key=lambda m: metric(scores[m])) if scores else None
        keep = pick is None or metric(scores[pick]) >= metric(best_score)
        summary = {"round": r, "parent": best, "parent_metric": metric(best_score),
                   "scores": scores, "pick": pick, "kept_parent": keep, "seconds": time.time() - t0}
        if not keep:
            best, best_score = pick, scores[pick]
        json.dump(summary, open(HERE / f"round_{r}.json", "w"), indent=1)
        print(f"round {r}: " + ", ".join(f"{m} {metric(s):.3f}" for m, s in scores.items())
              + f"; parent now {best} ({time.time() - t0:.0f} s)", flush=True)
        last = time.time() - t0
        r += 1
    print("deadline: stopping", flush=True)


if __name__ == "__main__":
    main()
