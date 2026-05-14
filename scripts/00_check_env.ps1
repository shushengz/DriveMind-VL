Write-Host "== DriveMind-VL Environment Check =="
Write-Host "Current path: $(Get-Location)"
Write-Host ""

Write-Host "== Python =="
python --version
pip --version
Write-Host ""

Write-Host "== NVIDIA =="
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvidia-smi
} else {
  Write-Host "nvidia-smi: not found"
}
Write-Host ""

Write-Host "== PyTorch =="
python -c "import importlib.util; spec=importlib.util.find_spec('torch'); print('torch installed:', bool(spec));"
python -c "import torch; print('torch version:', torch.__version__); print('torch.cuda.is_available():', torch.cuda.is_available())" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "torch import failed. For the local MVP this is acceptable; run scripts/01_install_env.sh later for full deps."
}
Write-Host ""

Write-Host "== Disk =="
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free,Root | Format-Table -AutoSize
Write-Host ""

Write-Host "== Memory =="
Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | Format-List

