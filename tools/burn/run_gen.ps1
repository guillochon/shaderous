$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "C:\Users\guill\shaderous\tools\burn"
Set-Location "C:\Users\guill\shaderous"
$py = "C:\Users\guill\shaderous\tools\burn\.venv\Scripts\python.exe"
foreach ($kind in "fuel", "mix") {
  $p = Start-Process -FilePath $py -ArgumentList "tools/burn/gen_data.py", $kind, "4000" -RedirectStandardOutput "tools/burn/gen_$kind.log" -RedirectStandardError "tools/burn/gen_$kind.err" -NoNewWindow -PassThru
  $p.PriorityClass = 'BelowNormal'
  $p.WaitForExit()
}
