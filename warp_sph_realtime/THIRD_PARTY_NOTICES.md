# Third-Party Notices

This add-on is distributed under GPL-3.0-or-later (see `LICENSE`). It is not
affiliated with or endorsed by NVIDIA. It does not bundle NVIDIA Warp: the
user installs Warp from Add-on Preferences, either by a confirmed download of
the certified wheel from PyPI or from a wheel or Warp folder they already have.

The certified runtime is NVIDIA Warp `warp-lang 1.17.0` (the PyPI CUDA 12
build). The Windows x64 and Linux x64 wheels are pinned by filename, size,
SHA-256 and HTTPS URL in `warp_wheels.lock.json`. The installer refuses a
wheel that does not match, and extracts it unmodified into this extension's
own user directory.

## Third-party components

| Dependency or resource | Certified version/status | License or terms | Included in the add-on zip? |
|---|---:|---|---:|
| NVIDIA Warp (`warp-lang`) | 1.17.0 (CUDA 12 build) | Apache License, Version 2.0 | No. Installed by the user from Add-on Preferences |
| CUDA runtime/compiler components used by Warp | As shipped inside the Warp wheel | NVIDIA Software License Agreement | No. They arrive inside the Warp wheel |
| NVIDIA libmathdx components used by Warp | As shipped inside the Warp wheel | NVIDIA upstream terms | No. They arrive inside the Warp wheel |
| NumPy | Version supplied by Blender | BSD 3-Clause License | No separate copy |
| Blender Python API | Blender 5.2.0 or newer | GNU General Public License | No |

The Warp wheel carries its own license files in
`warp_lang-1.17.0.dist-info/licenses/`. Because the wheel is installed
unmodified, those notices are installed alongside Warp.

"NVIDIA", "CUDA" and "Warp" are trademarks of NVIDIA Corporation, used here
only to name the software this add-on requires.
