[CmdletBinding()]
param([ValidateSet('Debug','Release')][string]$Configuration = 'Release')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $root 'build'
$outputDir = Join-Path $buildRoot $Configuration
$app = Join-Path $outputDir 'radbruter.exe'
if (!(Test-Path -LiteralPath $app)) { throw "Build $Configuration first." }
$testRoot = Join-Path (Join-Path (Join-Path $buildRoot 'obj') $Configuration) 'test-results'
New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
function Check-Native {
    param([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$Directory, [string]$Expected = '')
    $ErrorActionPreference = 'Continue'
    Push-Location $Directory
    try {
        $errorFile = Join-Path $testRoot "$Name.stderr.log"
        $lines = & $Executable @Arguments 2> $errorFile
        $result = $LASTEXITCODE
        $text = $lines -join "`n"
        $text | Set-Content -LiteralPath (Join-Path $testRoot "$Name.stdout.log") -Encoding UTF8
        $errors = Get-Content -LiteralPath $errorFile -Raw
        if ($result -ne 0 -or $errors -or ($Expected -and !$text.Contains($Expected))) {
            throw "$Name failed (exit $result).`n$text`n$errors"
        }
        Write-Host "$Name passed"
    } finally { Pop-Location }
}
Check-Native 'native_infrastructure' $app @('--self-test','--backend','cpu') $outputDir
foreach ($family in @('zhlt','vhlt','sdhlt')) {
    Check-Native $family $app @('--self-test','--backend','cpu','--compiler',$family) $outputDir
}
Check-Native 'automatic_backend' $app @('--self-test') $outputDir 'Automatic backend:'
$previousDevices = [Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES')
try {
    [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', '-1')
    Check-Native 'no_visible_gpu' $app @('--self-test') $outputDir 'Automatic backend: cpu'
} finally { [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', $previousDevices) }

$stage = Join-Path $buildRoot ('test-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    Copy-Item -LiteralPath $app -Destination $stage
    Copy-Item -LiteralPath (Join-Path $outputDir 'compilers') -Destination $stage -Recurse
    $portableApp = Join-Path $stage 'radbruter.exe'
    if ((Get-FileHash -LiteralPath $app).Hash -ne (Get-FileHash -LiteralPath $portableApp).Hash) {
        throw 'CPU fallback must use the same executable.'
    }
    Check-Native 'cpu_fallback' $portableApp @('--self-test') $stage 'Automatic backend: cpu'
} finally {
    $resolvedStage = (Resolve-Path -LiteralPath $stage).ProviderPath
    $resolvedRoot = (Resolve-Path -LiteralPath $buildRoot).ProviderPath + [IO.Path]::DirectorySeparatorChar
    if (!$resolvedStage.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Test cleanup target is outside the test directory.'
    }
    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
}
Write-Host "All 7 $Configuration checks passed."
