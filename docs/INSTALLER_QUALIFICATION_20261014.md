# WarpSPH Installer Qualification — 2026-10-14

**Isolation:** implemented on feat/isolated-warp-installer, based on main
eb0296452615dd9c923e32c69c0877f8743e8f8f. Changes exclude rendering
PR #1, production Blender profiles, and unrelated extensions.

## Tested Windows environment

- Blender 5.2.0 LTS x64, fbe6228777e7; Python 3.13.13; NumPy 2.3.4
- NVIDIA RTX 4080 SUPER, sm_89, 16 GiB; driver 610.88
- NVIDIA Warp 1.17.0 PyPI CUDA Toolkit 12.9; cuda:0
- CUDA Toolkit 12.9 CUDART cudart64_12.dll reports 12090
- Pinned Windows wheel 151236515 bytes; SHA-256:
  ca35c82242a7553046f09023bbe56750b5c3d9af72625bd1a905a56d3189771d

## Verified acceptance gates

1. Blender 5.2 LTS builds and installs the extension in a separate F: user
   profile and registers/unregisters without importing Warp. Offline
   install refuses download and leaves the owned runtime directory empty.
2. The exact locked PyPI wheel downloaded, verified and extracted into
   extension-owned storage. A second installation is idempotent.
   The Blender installer smoke used a local verified copy of the same
   downloaded wheel to avoid a duplicate 151-MB transfer.
3. Fresh Blender process imports that owned Warp, initializes CUDA and
   successfully aliases external Warp device memory.
4. GPU solver runs captured-graph and eager steps; large particle
   boundedness, physics and demo tests passed.
5. Blender gpu.init() establishes an OpenGL context in background mode:
   CUDA device association, VBO registration, map, GPU upload, unmap and
   native resource release all succeed.
6. A repair with Warp already loaded stages a verified wheel without
   modifying the loaded native files. A separate fresh process applies
   the repair and reinitializes CUDA. Unrelated files persist.
7. The full existing and new test suite initially passed 36/36; two more
   rollback and Windows long-path regressions then passed. The separate
   opt-in fresh install plus CUDA/OpenGL integration test passed.

## Reproduction

Fast installer tests:
    python -m unittest discover -s tests -p test_warp_installation.py -v

Blender 5.2 LTS isolated package install:
    set WARPSPH_BLENDER_EXE=<absolute Blender 5.2 blender.exe>
    python -m unittest discover -s tests -p test_blender52_isolated_install.py -v

Real locked-wheel CUDA/OpenGL test:
    set WARPSPH_BLENDER_EXE=<absolute Blender 5.2 blender.exe>
    set WARPSPH_CERTIFIED_WHEEL=<absolute SHA-locked 1.17 wheel>
    python -m unittest discover -s tests -p test_blender52_cuda_gl.py -v

The opt-in CUDA test builds and installs the extension into a fresh
temporary isolated user profile. It uses a verified local wheel fixture,
not an unverified network response.

## Qualification matrix

| Environment | Locked install | GPU solver | CUDA/OpenGL |
| --- | --- | --- | --- |
| Windows Blender 5.2 LTS, driver 610.88, Warp 1.17 CUDA 12.9, RTX 4080 SUPER | Verified | Verified | Verified |
| Linux x64 Warp 1.17 CUDA 12.9 | Locked only | Not tested | Not tested |
| Warp 1.16.x | Not certified | Not tested | Not tested |
| Warp 1.17 CUDA 13 variant | Not certified | Not tested | Not tested |
| Warp 1.18 CUDA 13 | Not certified | Not tested | Not tested |
| Other GPUs, drivers, and CUDA 12 minor versions | Not certified | Not tested | Not tested |

## Remaining boundaries

Independent PR review and a visible foreground Blender UI
installation/particle-render validation remain before release.
Native GL buffer transfer does not prove the viewport output or
appearance; rendering fixes remain in PR #1.

Blender's official Extensions repository forbids on-demand Python-module
installation. This installer is for self-hosted extension distribution.
