# Rounds of DAgger for the burning surrogate (dagger.py, train.py), at below-normal priority:
# roll the current model out, burn the states it visits, retrain on everything.
#   powershell -ExecutionPolicy Bypass -File tools/burn/run_dagger.ps1 [rounds] [first]
param([int]$rounds = 3, [int]$first = 1, [string]$model = "model_w128d3h4.pt")
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
  $model = "model_dagger_$r.pt"
  Run @("train.py", "60000", "128", "3", "4", $model) "train_dagger_$r"
}
