param(
  [Parameter(Mandatory = $true)]
  [string]$InputDirectory,

  [string]$OutputDirectory = "runs/real_reconstruction",

  [string]$Pattern = "*.png"
)

$ErrorActionPreference = "Stop"

python main.py reconstruct `
  --config configs/default.yaml `
  --input $InputDirectory `
  --input-pattern $Pattern `
  --output $OutputDirectory `
  --registration estimate `
  --save-debug
