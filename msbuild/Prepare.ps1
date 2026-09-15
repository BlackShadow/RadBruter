# Restore pinned sources and generate native build inputs; compilation belongs to MSBuild.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$deps = Join-Path $root '.deps'
$generated = Join-Path $deps 'generated/windows'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$sourceEncoding = [Text.Encoding]::GetEncoding(28591)
New-Item -ItemType Directory -Path $deps,$generated -Force | Out-Null

function Assert-DependencyPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    $boundary = [IO.Path]::GetFullPath($deps).TrimEnd('\') + '\'
    if (!$full.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Dependency path is outside .deps: $full"
    }
    return $full
}
function Write-Changed([string]$Path, [string]$Text, [Text.Encoding]$Encoding = $utf8) {
    if ((Test-Path -LiteralPath $Path) -and [IO.File]::ReadAllText($Path, $Encoding) -ceq $Text) { return }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
    [IO.File]::WriteAllText($Path, $Text, $Encoding)
}
function Restore-Package {
    param([string]$Name, [string]$Url, [string]$Hash, [string]$Marker,
          [string[]]$Keep, [string]$Revision = '')
    $destination = Assert-DependencyPath (Join-Path $deps $Name)
    if (Test-Path -LiteralPath (Join-Path $destination $Marker)) {
        if ($Revision -and (!(Test-Path -LiteralPath (Join-Path $destination '.revision')) -or
            (Get-Content -LiteralPath (Join-Path $destination '.revision') -Raw).Trim() -ne $Revision)) {
            throw "Unexpected compiler revision in $destination"
        }
        return
    }
    if (Test-Path -LiteralPath $destination) { throw "Incomplete dependency: $destination" }
    Write-Host "Restoring $Name..."
    $stage = Assert-DependencyPath (Join-Path $deps ('.unpack-' + [guid]::NewGuid().ToString('N')))
    New-Item -ItemType Directory -Path $stage | Out-Null
    try {
        $archive = Join-Path $stage 'archive'
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $archive
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Hash) {
            throw "Checksum mismatch for $Name"
        }
        $source = Join-Path $stage 'source'
        $package = Join-Path $stage 'package'
        New-Item -ItemType Directory -Path $source,$package | Out-Null
        & tar.exe -xf $archive --strip-components=1 -C $source
        if ($LASTEXITCODE -ne 0) { throw "Extraction failed for $Name" }
        foreach ($pattern in $Keep) {
            foreach ($entry in Get-ChildItem -Path (Join-Path $source $pattern) -ErrorAction SilentlyContinue) {
                $relative = $entry.FullName.Substring($source.Length + 1)
                $target = Join-Path $package $relative
                New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
                Copy-Item -LiteralPath $entry.FullName -Destination $target -Recurse
            }
        }
        if (!(Test-Path -LiteralPath (Join-Path $package $Marker))) { throw "Missing $Marker in $Name" }
        if ($Revision) { [IO.File]::WriteAllText((Join-Path $package '.revision'), "$Revision`n", $utf8) }
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        $checkedPackage = Assert-DependencyPath ((Resolve-Path -LiteralPath $package).ProviderPath)
        $checkedDestination = Assert-DependencyPath $destination
        Move-Item -LiteralPath $checkedPackage -Destination $checkedDestination
    } finally {
        $checkedStage = Assert-DependencyPath ((Resolve-Path -LiteralPath $stage).ProviderPath)
        Remove-Item -LiteralPath $checkedStage -Recurse -Force
    }
}
function Replace-Hook([string]$Code, [string]$Before, [string]$After, [string]$File) {
    if (!$Code.Contains($Before)) { throw "Compiler hook is missing in ${File}: $Before" }
    return $Code.Replace($Before, $After)
}

$lock = $null
$deadline = [DateTime]::UtcNow.AddMinutes(5)
while (!$lock) {
    try { $lock = [IO.File]::Open((Join-Path $deps 'native-prepare.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
    catch [IO.IOException] {
        if ([DateTime]::UtcNow -ge $deadline) { throw 'Timed out waiting for dependency setup.' }
        Start-Sleep -Milliseconds 100
    }
}
try {
    Restore-Package -Name 'eigen' -Marker 'Eigen/Core' -Keep @('Eigen','COPYING.*') `
        -Url 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz' `
        -Hash '8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72'
    Restore-Package -Name 'lbfgspp' -Marker 'include/LBFGSB.h' -Keep @('include','LICENSE.md','AUTHORS.md') `
        -Url 'https://github.com/yixuan/LBFGSpp/archive/refs/tags/v0.3.0.tar.gz' `
        -Hash '490720b9d5acce6459cb0336ca3ae0ffc48677225f0ebfb35c9bef6baefdfc6a'
    $cuda = 'https://developer.download.nvidia.com/compute/cuda/redist'
    Restore-Package -Name 'cuda/Windows/runtime' -Marker 'include/cuda.h' -Keep @('include','LICENSE') `
        -Url "$cuda/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip" `
        -Hash '179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8'
    Restore-Package -Name 'cuda/Windows/nvrtc' -Marker 'include/nvrtc.h' `
        -Keep @('include','bin/nvrtc64_120_0.dll','bin/nvrtc-builtins64_129.dll','LICENSE') `
        -Url "$cuda/cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-12.9.86-archive.zip" `
        -Hash '1aa0644fa53c8ca34cdc73db17bcc73530557bdd3f582c7bfdbd7916c8b48f65'

    $compilers = @(
        @{ Family='zhlt'; Repo='kriswema/zhlt'; Revision='da4f9d76cf425b06864c29b32dab73892e36a6c6';
           Hash='2ad186843264230bcd460be8ed34ad6fd7f1d8974de95696af8e7e4ff81d981e'; Prefix=''; Rad='hlrad' },
        @{ Family='vhlt'; Repo='FreeSlave/vhlt'; Revision='13b83f91d093ef146a9a78834578994d85f99d96';
           Hash='6626f23f379bd73bcb1e7b7645a0b17647bd163e59cf0fc1a8920b54c9cf8ad3'; Prefix=''; Rad='hlrad' },
        @{ Family='sdhlt'; Repo='seedee/SDHLT'; Revision='df45198b3c03a5a09e9d1aead9c9457e51753e39';
           Hash='5fca807f6da7db4dfb06744676ebdf18ffaadff820dd90d6744b8298a0cba11f'; Prefix='src/sdhlt/'; Rad='sdHLRAD' }
    )
    foreach ($compiler in $compilers) {
        $keep = @('LICENSE.md','Terms of Use.txt')
        foreach ($directory in @('common','template',$compiler.Rad)) {
            $keep += $compiler.Prefix + $directory + '/*.cpp'
            $keep += $compiler.Prefix + $directory + '/*.h'
        }
        Restore-Package -Name ('compiler-sources/' + $compiler.Family) `
            -Url ('https://codeload.github.com/' + $compiler.Repo + '/tar.gz/' + $compiler.Revision) `
            -Hash $compiler.Hash -Marker ($compiler.Prefix + $compiler.Rad + '/qrad.cpp') `
            -Revision $compiler.Revision -Keep $keep
        $upstream = Join-Path (Join-Path $deps ('compiler-sources/' + $compiler.Family)) $compiler.Prefix
        $output = Join-Path $generated $compiler.Family
        $common = @('blockmem','bspfile','cmdlib','filelib','log','mathlib','messages','scriplib','threads','winding')
        $engine = @('lerp','lightmap','mathutil','nomatrix','qrad','qradutil','sparse','trace','transfers','transparency','vismatrix','vismatrixutil')
        if ($compiler.Family -eq 'zhlt') { $common += 'resourcelock' }
        else { $common += 'cmdlinecfg'; $engine += @('compress','loadtextures') }
        if ($compiler.Family -eq 'sdhlt') { $engine += @('meshdesc','meshtrace','progmesh','stringlib','studio') }
        $files = @($common | ForEach-Object { "common/$_.cpp" })
        $files += @($engine | ForEach-Object { "$($compiler.Rad)/$_.cpp" })
        foreach ($directory in @('common','template',$compiler.Rad)) {
            $files += @(Get-ChildItem -LiteralPath (Join-Path $upstream $directory) -File -Filter '*.h' | ForEach-Object { "$directory/$($_.Name)" })
        }
        foreach ($relative in $files) {
            $code = [IO.File]::ReadAllText((Join-Path $upstream $relative), $sourceEncoding).Replace("`r`n", "`n")
            if ($relative -eq "$($compiler.Rad)/qrad.cpp") {
                $hooks = @(
                    @('int             main(const int argc, char** argv)', 'int native_rad_main(const int argc, char** argv)'),
                    @('    char global_lights[_MAX_PATH];', "    ReadLightFile(user_rad);`n    ReadInfoTexlights();`n    return;`n    char global_lights[_MAX_PATH];"),
                    @('LoadBSPFile(g_source);', "LoadBSPFile(getenv(`"LM_BSP_INPUT`"));`n    lm_native_loaded();"),
                    @('WriteBSPFile(g_source);', 'lm_native_finish();'),
                    @('    RadWorld();', "    lm_mark(LM_GEOMETRY);`n    RadWorld();"),
                    @('    CreateDirectLights();', "    lm_mark(LM_DIRECT);`n    CreateDirectLights();"),
                    @('    DeleteDirectLights();', "    lm_native_structure();`n    DeleteDirectLights();"),
                    @('        MakeScalesStub();', "        lm_mark(LM_TRANSFERS);`n        MakeScalesStub();"),
                    @('        BounceLight();', "        lm_mark(LM_BOUNCE);`n        BounceLight();"),
                    @('    FreeTransfers();', "    lm_mark(LM_ENCODE);`n    FreeTransfers();")
                )
                if ($compiler.Family -ne 'zhlt') {
                    $hooks += ,@('ParseParamFile (argcold, argvold, argc, argv);', 'argc = argcold; argv = argvold; /* Explicit runner arguments only. */')
                    $hooks += ,@("`tReduceLightmap();", "`tlm_native_capture();`n`tReduceLightmap();")
                }
                foreach ($hook in $hooks) { $code = Replace-Hook $code $hook[0] $hook[1] $relative }
                if ($compiler.Family -eq 'sdhlt') { $code = $code.Replace("ReadInfoTexlights();`n    return;", "ReadInfoTexAndMinlights();`n    return;") }
                $code = "// RadBruter transport/timing hooks, September 2026. Upstream lighting is unchanged.`n#include `"runtime.h`"`n#include `"compiler.h`"`n$code"
            } elseif ($relative -eq "$($compiler.Rad)/lightmap.cpp") {
                $code = Replace-Hook $code "`tif (!(tex->flags & TEX_SPECIAL))" "`tlm_original_grid(l->surfnum, l->texmins, l->texsize);`n`tif (!(tex->flags & TEX_SPECIAL))" $relative
                $code = Replace-Hook $code "`t        lb[0] *= g_colour_lightscale[0];" "            lm_native_sample(facenum, k, j, fl->numsamples, lb, minlight);`n`t        lb[0] *= g_colour_lightscale[0];" $relative
                if ($compiler.Family -eq 'zhlt') { $code = Replace-Hook $code '    hlassume(numdlights, assume_NoLights);' '    /* Zero-light probes are intentional in the inverse solver. */' $relative }
                $code = "// RadBruter raw-sample observation hook, September 2026.`n#include `"runtime.h`"`n#include `"compiler.h`"`n$code"
            } elseif ($relative -eq "$($compiler.Rad)/loadtextures.cpp") {
                $code = Replace-Hook $code 'Warning ("Texture ''%s'': texture is not found in wad files.", tex->name);' 'Error ("Texture ''%s'' is missing; supply its WAD directory with --waddir.", tex->name);' $relative
            }
            if ($compiler.Family -eq 'zhlt') {
                $code = $code.Replace('strcpy_s(', 'strcpy(').Replace('sprintf_s(', 'sprintf(').Replace('sscanf_s(', 'sscanf(').Replace('_strlwr(', 'strlwr(')
                foreach ($function in @('strdup','open','read','close','unlink')) { $code = $code.Replace("_$function(", "$function(") }
                $code = $code.Replace('q_entry((int)pParam);', 'q_entry(int(reinterpret_cast<intptr_t>(pParam)));')
                $code = $code.Replace('ThreadEntryStub, (void*)i', 'ThreadEntryStub, reinterpret_cast<void*>(intptr_t(i))').Replace('(int)tnodes', '(uintptr_t)tnodes')
                if ($relative -eq 'common/threads.cpp') { $code = "#include <cstdint>`n$code" }
                if ($relative -eq 'common/cmdlib.cpp') { $code = "#include <algorithm>`nusing std::max;`n$code" }
                if ($relative -eq "$($compiler.Rad)/mathutil.cpp") { $code = $code.Replace('inline ', '').Replace('INLINE ', '') }
            }
            if ($relative -eq 'common/filelib.cpp') { $code = "#include `"platform_io.h`"`n#define fopen lm_fopen`n$code" }
            if ($relative -eq 'common/win32fix.h') { $code = $code.Replace('#define vsnprintf _vsnprintf', '/* The UCRT supplies vsnprintf. */').Replace('#define snprintf  _snprintf', '/* The UCRT supplies snprintf. */') }
            $code = [regex]::Replace($code, '__asm[ \t\n]*\{[ \t\n]*int 3;?[ \t\n]*\}', '__debugbreak();')
            if ($relative -eq 'common/threads.cpp') { $code = $code.Replace('q_entry((int)pParam);', 'q_entry(int(reinterpret_cast<intptr_t>(pParam)));').Replace('(LPVOID) i,', 'reinterpret_cast<LPVOID>(intptr_t(i)),') }
            if ($compiler.Family -eq 'sdhlt' -and $relative -eq "$($compiler.Rad)/meshtrace.h") { $code = $code.Replace('((t *)((byte *)l - (int)(long int)&(((t *)0)->m)))', 'reinterpret_cast<t *>(reinterpret_cast<byte *>(l) - offsetof(t, m))') }
            if ($compiler.Family -eq 'sdhlt' -and $relative -eq "$($compiler.Rad)/stringlib.cpp") { $code = $code.Replace('register ', '') }
            if ($compiler.Family -ne 'zhlt' -and $relative -eq 'common/winding.cpp') {
                $code = Replace-Hook $code "Winding::Winding(UINT32 numpoints)`n{`n    hlassert(numpoints >= 3);" "Winding::Winding(UINT32 numpoints)`n{`n    // Clipping a sample fragment can produce an empty winding.`n    hlassert(numpoints == 0 || numpoints >= 3);" $relative
            }
            if ($compiler.Family -ne 'zhlt' -and $relative -eq 'common/bspfile.cpp') {
                $code = Replace-Hook $code "void GetFaceExtents (int facenum, int mins_out[2], int maxs_out[2])`n{" "void GetFaceExtents (int facenum, int mins_out[2], int maxs_out[2])`n{`n    int original_size[2];`n    if (lm_original_grid(facenum, mins_out, original_size)) {`n        maxs_out[0] = mins_out[0] + original_size[0];`n        maxs_out[1] = mins_out[1] + original_size[1];`n        return;`n    }" $relative
                $code = "#include `"runtime.h`"`n$code"
            }
            $code = "// Modified for RadBruter native transport and Linux/64-bit portability, September 2026.`n$code"
            Write-Changed (Join-Path $output $relative) $code $sourceEncoding
        }
        Write-Changed (Join-Path $output 'build_identity.h') "#pragma once`n#define LM_COMPILER_NAME `"$($compiler.Family)`"`n#define LM_COMPILER_COMMIT `"$($compiler.Revision)`"`n"
    }
    $types = [IO.File]::ReadAllText((Join-Path $root 'src/gpu_types.h'))
    $kernels = [IO.File]::ReadAllText((Join-Path $root 'src/kernels.cu'))
    $template = [IO.File]::ReadAllText((Join-Path $root 'src/kernels.h.in'))
    Write-Changed (Join-Path $generated 'kernels.h') ($template.Replace('@LM_CUDA_SOURCE@', "#define LM_WINDOWS_ARITHMETIC 1`n$types`n$kernels"))
    [IO.File]::WriteAllText((Join-Path $generated 'ready.stamp'), [DateTime]::UtcNow.ToString('O'), $utf8)
} finally {
    $lock.Dispose()
}
