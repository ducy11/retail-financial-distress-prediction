<#
.SYNOPSIS
    Build a clean, submission-ready copy of the project (target: total < 50 MB).

.DESCRIPTION
    Copies ONLY an explicit allow-list of files (core Python source, tests, light
    config/metadata, the minimal data + report artifacts needed so that
    `python -m scripts.run_all` and `python -m unittest discover -s tests` actually run)
    into a staging folder, then reports the size and, with -Zip, produces the archive.

    Nothing is deleted from the original tree: this script only COPIES.

    Deliberately EXCLUDED (recreate them, do not ship them):
      * Caches/bytecode ....... __pycache__/, *.pyc
      * Envs/IDE/OS .......... venv/, .venv/, .pytest_cache/, .vscode/, .idea/, .DS_Store, Thumbs.db
      * Model artifacts ....... reports/models/*.joblib
      * Logs .................. *.log (incl. reports/results/run_all.log)
      * Raw SEC snapshots ..... data/sec/raw/  (re-download with: python -m scripts.crawl_sec --refresh)
      * Regenerable tables .... data/retail-expanded-rebuilt/ (scripts.prepare_sec), data/prepared-rule/ (scripts.relabel)
      * Report deliverables ... docs/ (handled separately by the author)
      * Archives/VCS ........... *.zip, *.tar.gz, .git/

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\tools\export_submission.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File .\tools\export_submission.ps1
    powershell -ExecutionPolicy Bypass -File .\tools\export_submission.ps1 -Zip
#>
[CmdletBinding()]
param(
    [string]$Source = '',
    [string]$Dest   = '',
    [switch]$Zip,
    [switch]$DryRun,
    [int]$LimitMB = 50
)

$ErrorActionPreference = 'Stop'

# Resolve paths inside the body: $PSScriptRoot is only reliable once the script is running.
$root = Split-Path -Parent $PSScriptRoot
if (-not $Source) { $Source = $root }
if (-not $Dest)   { $Dest = Join-Path $root '_submission\do-an-src' }
$Source = (Resolve-Path $Source).Path

$keepDirs  = @('forecasting', 'scripts', 'labs', 'tests', '.github')
$keepData  = @('data\prepared', 'data\retail-expanded', 'data\events', 'data\samples')
$keepRoot  = @('app.py', 'runtime_warnings.py', 'requirements.txt', 'requirements-app.txt',
               'requirements-labs.txt', 'README.md', 'README.vi.md', 'README.zh-TW.md',
               'LICENSE', '.gitignore', '.gitattributes', '.env.example')
$keepDataFiles = @('data\README.md', 'data\sec\downloads.json')

# A file is dropped if ANY of these match its relative path.
$denyPattern = '__pycache__|\.venv|/venv/|\\venv\\|\.pytest_cache|\.vscode|\.idea|' +
               '\\docs\\|/docs/|retail-expanded-rebuilt|prepared-rule|data\\sec\\raw|data/sec/raw|' +
               '\.joblib$|\.log$|\.pyc$|\.zip$|\.tar\.gz$|\.7z$|\.rar$'

function Get-KeepFiles {
    $files = @()
    foreach ($d in $keepDirs) {
        $p = Join-Path $Source $d
        if (Test-Path $p) { $files += Get-ChildItem -Path $p -Recurse -File -Force }
    }
    foreach ($f in $keepRoot) {
        $p = Join-Path $Source $f
        if (Test-Path $p) { $files += Get-Item $p -Force }
    }
    foreach ($d in $keepData) {
        $p = Join-Path $Source $d
        if (Test-Path $p) { $files += Get-ChildItem -Path $p -Recurse -File -Force }
    }
    foreach ($f in $keepDataFiles) {
        $p = Join-Path $Source $f
        if (Test-Path $p) { $files += Get-Item $p -Force }
    }
    $rep = Join-Path $Source 'reports'
    if (Test-Path $rep) { $files += Get-ChildItem -Path $rep -Recurse -File -Force }

    $files | Where-Object {
        $rel = $_.FullName.Substring($Source.Length).TrimStart('\', '/')
        $rel -notmatch $denyPattern
    } | Sort-Object FullName -Unique
}

$keep = @(Get-KeepFiles)
$bytes = ($keep | Measure-Object -Property Length -Sum).Sum
$mb = [math]::Round($bytes / 1MB, 2)
Write-Host ("Keep-set: {0} files, {1} MB  (limit {2} MB)" -f $keep.Count, $mb, $LimitMB)

if ($DryRun) {
    Write-Host "`n--- files that will be copied ---"
    $keep | ForEach-Object { $_.FullName.Substring($Source.Length).TrimStart('\', '/') }
    if ($mb -ge $LimitMB) { throw "Over the $LimitMB MB budget." }
    return
}

if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
foreach ($f in $keep) {
    $rel = $f.FullName.Substring($Source.Length).TrimStart('\', '/')
    $target = Join-Path $Dest $rel
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath $f.FullName -Destination $target -Force
}
# Keep empty runtime folders present (so a fresh clone matches the layout).
foreach ($d in @('reports\models', 'reports\figures', 'reports\results', 'data\sec\raw')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Dest $d) | Out-Null
}

$final = (Get-ChildItem -Path $Dest -Recurse -File -Force | Measure-Object -Property Length -Sum).Sum
$finalMB = [math]::Round($final / 1MB, 2)
Write-Host ("Staged {0} -> {1} MB" -f $Dest, $finalMB)

if ($Zip) {
    $zipPath = "$Dest.zip"
    if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
    Compress-Archive -Path (Join-Path $Dest '*') -DestinationPath $zipPath -CompressionLevel Optimal
    $zipMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
    Write-Host ("Archive: {0} -> {1} MB" -f $zipPath, $zipMB)
}

if ($finalMB -ge $LimitMB) { throw "Over the $LimitMB MB budget ($finalMB MB)." }
Write-Host "OK: within budget." -ForegroundColor Green
