# PR #32 type-check follow-up

The follow-up applies the archived two-file patch to merged main `134368e232e894a4fd92f1ee35c95bbb1948dc24`. The bake callbacks keep their original inputs and branch selection. The readback fault test now uses pytest's scoped monkeypatch to restore the stand-in array method.

The Windows host used Python 3.11.9, Warp 1.17.0, pytest 9.0.1, Ruff 0.14.11 and Pyright 1.1.408. The SDK source at `5bfc26188f973a181cd47b6fe8515e84d58164da` matched all 33 locked payload digests after LF normalization. CUDA was masked for the host checks.

| Check | Result |
| --- | --- |
| `python -m pytest -q tests/test_sdk_snapshot.py tests/test_cache_scene_provenance.py --tb=short` | 36 passed, exit 0 |
| `ruff check .` | Passed, exit 0 |
| `pyright --pythonpath <verification Python>` | Zero errors, warnings or information, exit 0 |
| `git diff --check` | Passed, exit 0 |

The initial full Pyright run reported missing plotting imports. Installing Matplotlib 3.10.8 in the isolated verification environment resolved them. Inspected checks cover the two-file host follow-up. Blender and CUDA runtime verification and GitHub Actions private SDK access remain open.
