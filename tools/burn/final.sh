#!/bin/sh
# Burner retraining, end to end on a pod: DAgger rounds with the overshoot variant only until
# the deadline (epoch seconds, $1), then a final fine-tune of the best-scoring model, scored.
# Writes FINAL_DONE (the best model of all, by score.py's metric) when it is finished.
cd "$(dirname "$0")"
BURN_TEACHER=- BURN_VARIANTS=b python -u run_rounds.py "$2" "$1" "$3" >> rounds.log 2>&1
BEST=$(python - <<'PY'
import glob, json
def metric(s): return sum(s["general_p99"]) / len(s["general_p99"])
best = {}
for f in glob.glob("score_*.json"):
    for s in json.load(open(f)):
        best[s["model"]] = metric(s)
m = min(best, key=best.get)
print(m)
PY
)
echo "final fine-tune from $BEST" >> rounds.log
BURN_NO_ROLLOUT=1 python -u train.py 60000 128 4 4 model_final.pt 0.3 0.2 flow2f 1 "$BEST" 0 2e-4 1 0.3 7 - 0 20 0 > train_final.log 2>&1
python score.py model_final.pt --json score_final.json > score_final.log 2>&1
python - <<'PY' > FINAL_DONE
import glob, json
def metric(s): return sum(s["general_p99"]) / len(s["general_p99"])
best = {}
for f in glob.glob("score_*.json"):
    for s in json.load(open(f)):
        best[s["model"]] = metric(s)
for m in sorted(best, key=best.get):
    print(f"{best[m]:.4f} {m}")
PY
echo "done: $(head -1 FINAL_DONE)" >> rounds.log
