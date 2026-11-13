# Rounds of DAgger for the burning surrogate (dagger.py, train.py), at below-normal priority:
# roll the current model out, burn the states it visits, retrain on everything.
#   powershell -ExecutionPolicy Bypass -File tools/burn/run_dagger.ps1 [rounds] [first] [model] [rel] [fresh] [arch] [aux] [steps] [width] [depth] [logw] [lr]
#   lr given: each round fine-tunes the model before it at that rate (rates2 models collapse
#   to burning nothing when trained from scratch) instead of starting afresh.
param([int]$rounds = 3, [int]$first = 1, [string]$model = "model_w128d3h4.pt", [string]$rel = "0", [string]$fresh = "0.2", [string]$arch = "mixed", [string]$aux = "1", [string]$steps = "60000", [string]$width = "128", [string]$depth = "3", [string]$logw = "0", [string]$lr = "")
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\guill\shaderous\tools\burn"
$burn = "C:\Users\guill\shaderous\tools\burn"
$py = "$burn\.venv\Scripts\python.exe"
function Run($argv, $log) {
  $p = Start-Process -FilePath $py -ArgumentList $argv -WorkingDirectory $burn -RedirectStandardOutput "$burn\$log.log" -RedirectStandardError "$burn\$log.err" -NoNewWindow -PassThru
  $p.PriorityClass = 'BelowNormal'
  $p.WaitForExit()
}
for ($r = $first; $r -lt $first + $rounds; $r++) {
  Run @("dagger.py", $model, "$r") "dagger_$r"
  $init = if ($lr) { $model } else { "-" }
  $model = "model_dagger_$r.pt"
  Run @("train.py", $steps, $width, $depth, "4", $model, $rel, $fresh, $arch, $aux, $init, "0", $(if ($lr) { $lr } else { "2e-3" }), $logw) "train_dagger_$r"
}
