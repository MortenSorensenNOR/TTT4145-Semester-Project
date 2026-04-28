# iio_setup.ps1 — Windows equivalent of iio_setup.sh
# Builds libiio v0.25 and libad9361-iio from the vendor submodules,
# installs them into vendor\install (no admin rights required), then runs uv sync.
#
# Requirements: git, Visual Studio Build Tools (cl.exe + cmake component), uv,
#               chocolatey (winflexbison3), vcpkg with libxml2:x64-windows
$ErrorActionPreference = "Stop"

$Root    = Split-Path -Parent $PSScriptRoot
$Install = Join-Path $Root "vendor\install"

# ---------------------------------------------------------------------------
# Locate cmake — bundled inside VS and typically not on PATH
# ---------------------------------------------------------------------------
function Find-CMake {
    $inPath = Get-Command cmake -ErrorAction SilentlyContinue
    if ($inPath) { return $inPath.Source }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.CMake.Project `
                             -property installationPath 2>$null | Select-Object -First 1
        if ($vsRoot) {
            $candidate = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
            if (Test-Path $candidate) { return $candidate }
        }
    }

    throw "cmake not found. Install the 'C++ CMake tools for Windows' component in Visual Studio Installer."
}

# ---------------------------------------------------------------------------
# Locate vcpkg toolchain file
# ---------------------------------------------------------------------------
function Find-VcpkgToolchain {
    $candidates = @(
        $env:VCPKG_ROOT,
        "C:\vcpkg",
        "$env:USERPROFILE\vcpkg",
        "$env:LOCALAPPDATA\vcpkg"
    )
    foreach ($dir in $candidates) {
        if (-not $dir) { continue }
        $tc = Join-Path $dir "scripts\buildsystems\vcpkg.cmake"
        if (Test-Path $tc) { return $tc }
    }
    throw "vcpkg not found. Clone it to C:\vcpkg and run bootstrap-vcpkg.bat, then: vcpkg install libxml2:x64-windows"
}

$cmake    = Find-CMake
$toolchain = Find-VcpkgToolchain

Write-Host "==> Using cmake:  $cmake"
Write-Host "==> Using vcpkg toolchain: $toolchain"

# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------
Write-Host "==> Initializing submodules..."
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# libiio v0.25
# ---------------------------------------------------------------------------
Write-Host "==> Building libiio v0.25..."

Push-Location (Join-Path $Root "vendor\libiio")
git checkout v0.25

if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
New-Item -ItemType Directory "build" | Out-Null
Push-Location "build"

& $cmake .. `
    -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
    -DVCPKG_TARGET_TRIPLET="x64-windows" `
    -DCMAKE_INSTALL_PREFIX="$Install" `
    -DCMAKE_BUILD_TYPE=Release `
    -DWITH_USB_BACKEND=OFF `
    -DWITH_NETWORK_BACKEND=ON `
    -DNO_AVAHI=ON

& $cmake --build . --config Release --parallel
& $cmake --install . --config Release

Pop-Location  # build
Pop-Location  # vendor/libiio

# ---------------------------------------------------------------------------
# libad9361-iio
# ---------------------------------------------------------------------------
Write-Host "==> Building libad9361-iio..."

Push-Location (Join-Path $Root "vendor\libad9361-iio")

if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
New-Item -ItemType Directory "build" | Out-Null
Push-Location "build"

& $cmake .. `
    -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
    -DVCPKG_TARGET_TRIPLET="x64-windows" `
    -DCMAKE_INSTALL_PREFIX="$Install" `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
    -DLIBIIO_INCLUDEDIR="$Install\include" `
    -DLIBIIO_LIBRARIES="$Install\lib\libiio.lib"

& $cmake --build . --config Release --parallel --target ad9361
& $cmake --install . --config Release

Pop-Location  # build
Pop-Location  # vendor/libad9361-iio

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
Write-Host "==> Syncing Python dependencies..."
uv sync

# ---------------------------------------------------------------------------
# Write sitecustomize.py so Python finds libiio.dll and libxml2.dll
# Python 3.8+ no longer searches PATH for DLL dependencies loaded via ctypes;
# os.add_dll_directory() must be called explicitly before `import iio`.
# sitecustomize.py runs automatically on every Python startup inside the venv.
# ---------------------------------------------------------------------------
Write-Host "==> Writing sitecustomize.py into venv..."

$VcpkgRoot  = Split-Path (Split-Path (Split-Path $toolchain))  # ..\scripts\buildsystems\vcpkg.cmake -> root
$VcpkgBin   = Join-Path $VcpkgRoot "installed\x64-windows\bin"
$SiteCustomize = Join-Path $Root ".venv\Lib\site-packages\sitecustomize.py"

@"
import os

# Python 3.8+ no longer searches PATH for DLL dependencies loaded via ctypes.
# These directories must be registered explicitly so libiio.dll and libxml2.dll
# are found when ``import iio`` (via pyadi-iio) loads them with ctypes.
#
# We also prepend the dirs to PATH so ctypes.util.find_library() — which
# iio.py calls before LoadLibrary and which does NOT consult
# os.add_dll_directory() — can locate libiio.dll on Windows.
_dll_dirs = [
    r"$(Join-Path $Install 'bin')",
    r"$VcpkgBin",
]

for _d in _dll_dirs:
    if os.path.isdir(_d):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
"@ | Set-Content -Encoding UTF8 $SiteCustomize

Write-Host "==> Done!"
Write-Host ""
Write-Host "Libraries installed to: $Install"
