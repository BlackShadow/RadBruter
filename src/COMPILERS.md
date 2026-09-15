# Native compiler sources

Build setup downloads checksum-verified source archives automatically. It keeps
only RAD/common/template C++ sources and headers, the revision marker and
upstream license notices. Source patches are applied to generated copies;
the project adds one shared C++ adapter.

| Backend | Upstream revision |
| --- | --- |
| ZHLT 3.4 | [kriswema/zhlt](https://github.com/kriswema/zhlt/tree/da4f9d76cf425b06864c29b32dab73892e36a6c6) |
| VHLT VL34 | [FreeSlave/vhlt](https://github.com/FreeSlave/vhlt/tree/13b83f91d093ef146a9a78834578994d85f99d96) |
| SDHLT 1.3.0 | [seedee/SDHLT](https://github.com/seedee/SDHLT/tree/df45198b3c03a5a09e9d1aead9c9457e51753e39) |

Valve QRAD's source and notices remain in `qrad/`.
Each additional engine is an independent 64-bit executable in the `compilers/`
directory beside the application. Windows names end in `.exe`. Copying the
application alone omits those workers.
These builds are not claimed to reproduce every released Windows binary's
floating-point arithmetic. CUDA acceleration remains specific to QRAD.

Changes to upstream source are limited to the transport, profiling and
observation hooks, 64-bit platform support, disabling automatic RAD/config
discovery, allowing ZHLT's intentional zero-light baseline, and rejecting
missing textures. Workers read the input BSP through a memory descriptor and
use the original BSP's sample grid so platform rounding cannot add or remove rows.
They return samples and lighting without writing a BSP. Probe RAD files and any
upstream scratch files remain in the parent's temporary run directory.

The fitter observes each engine's raw samples and encoder settings. VHLT/SDHLT
use per-channel reflectivity and casting weights. Their lightmap reduction can
share offsets and append a safety buffer, so the parser accepts non-contiguous
lighting storage. The inverse solver currently retains the original BSP30
geometry, sample-grid and size limits; BSP2/Xash extended formats and embedded
lightmap texinfo rewrites are not supported. External studio-shadow assets and
non-default compiler options may require further integration. A low residual
does not establish a unique engine, version, settings or original RAD recipe.

The VHLT and SDHLT CSG sources can both stamp worldspawn with
`ZHLT v3.4 VL34 (...)`. This is family evidence and identifies the CSG stage,
not necessarily RAD. Their `ReduceLightmap` implementations also append
zero-filled space up to a 17-by-17 lightmap footprint on standard BSP30 faces;
the fast detector checks the exact footprint and padding contents. Optimizers
or later tools can remove these clues. Neither retail `c1a0` nor `c1a0d` has a
distinctive marker under these checks.

The supplied [Paranoia2 repository](https://github.com/a1batross/Paranoia2_original/tree/0455f5d6e2a9b42a4199c40fe981f641d35574c0/utils/p2rad)
contains P2RAD, with its own ray tracer and model/vertex/ambient lighting
extensions. It is not Sven Co-op's SCHLT. A public SCHLT source repository
was not found in this source review. These two engines are not silently
substituted with one of the integrated backends.
