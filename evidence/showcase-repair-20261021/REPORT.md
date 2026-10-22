# Showcase verification repair

Baseline: main e55dc1dec3de6d4ab92e011fa6f105a281c12536, PR #17 merge. No subsequent main fix existed at checkout.
Candidate: branch `fix/showcase-verification`; exact changed source SHA-256 values in `source-manifest.json`.

## Results and limits

| Claim | Status | Evidence |
|---|---|---|
| Wrong GPU route cannot pass display check; labelled Readback is fallback, not acceleration | VERIFIED in host controls | `focused-tests.txt`; actual route table plus wrong-label/capability controls |
| Gravity/emission alone cannot pass the new paired paddle metric | VERIFIED in synthetic host controls | gravity-only, stable-ID reorder and reverse-direction controls in `tests/test_showcase_failures.py` |
| Actual animated paddle produces required influence | BLOCKED | No new paired Blender bakes; thresholds fixed before GPU execution |
| Corrupt/unreadable receipts, setup/launch failures and timeouts preserve batch reporting | VERIFIED in injection tests | `focused-tests.txt`; setup timer stub, real child timeout, batch/media injection |
| Concurrent runner exclusion | VERIFIED in real separate Python processes | OS lock contender rejected, subsequent contender acquires after release |
| Supported Blender 5.2 GPU routes and complete rendering/media production | BLOCKED | `runtime-preflight.json`, `runtime-busy-attempt/summary.json` |
| Independent source review | VERIFIED, bounded | `REVIEW.md`; reviewer independently ran 35 tests |

No new Blender showcase was launched. The runtime-accessible runner exited 1 with all three requested rows
retained because another Blender remained active. It created no extension profile. A separate sandbox attempt
could not inspect running processes and also failed closed with all rows. Existing Blender processes were left
alone. GPU utilization varied between 0% and 99% during the session; occupied Blender sessions still precluded
the isolated showcase harness. This is not GPU/GUI/physics-accuracy certification, nor Linux certification.

**Original six successful PR #17 jobs are historical under the original checks. They have not passed the
strengthened checks.**

## Changes

- Exact expected accelerated route and backend checks for all display modes; Readback explicitly classified.
- Paired public stationary/animated bakes with identical initial state and ID-matched near-paddle signed
  tangential velocity response. Two of frames 11/21/31 must exceed 0.02 m/s, each with at least 20 matched
  particles. Original animated cache/render assets retained; control cache separate. Correctness readback is
  outside transport performance. This measures influence, not exact dynamics. Actual thresholds remain unrun.
- Build callbacks inside the existing protected timer driver for all five scenes, atomic summaries and quit
  in `finally` even if receipt writing fails.
- Fail-closed schema/result checks, per-job fault containment and incremental aggregate receipts. Setup failures
  retain every planned job; media failures do not erase job results.
- OS lease before mutation, unique output-owned profile and no output reuse. Process inventory is bounded and
  fails closed. No shared-profile destructive replacement in normal runner execution.
- ffmpeg timeout of 300 seconds per sequence, retained PNGs, per-encode errors and continued encoding. Contact
  sheet timeout/errors retained. Optional absence of ffmpeg still permits PNG-only success.

## Reproduction and verification

Windows 11, Python 3.14.2 host tests; Pyright 1.1.408. Installed Warp reports 1.17.0, CUDA Toolkit 12.9,
driver 610.88, RTX 4080 SUPER. This host is separate from Blender's required Python 3.13 acceptance lane.

`python evidence/showcase-repair-20261021/reproduce_baseline.py` extracts source at the pinned baseline and
evaluates the original weak check expressions against a wrong route and gravity-only speed. Both falsely
pass. It directly injects a truncated summary into the original runner and observes uncaught JSONDecodeError.
Remaining setup/profile/encoding flaws are source-structure checks, not exercised GPU failure consequences.
See `baseline-reproduction.json`. The initial regression run failed before repair (`red-tests.txt`).

Commands (clear ambient PYTHONPATH first):

```powershell
$env:PYTHONPATH=''
python -m pytest -p no:metadata --basetemp C:/workspace/warpsph-symbol-tests-20261022a tests/test_blender_symbols.py tests/test_showcase_runner.py tests/test_showcase_failures.py tests/test_examples.py -q
python -m ruff check .
python -m pyright
```

Focused final result: **86 passed, 17 subtests passed**, 2.71 seconds; Ruff clean; Pyright zero diagnostics
(run through the gateway's Python, where Pyright is installed). Logs are retained here.

The first full suite attempt produced **536 passed, 2 skipped, 40 setup errors** in 96.79 seconds; all errors
were inaccessible shared `C:/Windows/Temp/pytest-of-Tyler` fixtures in `test_blender_symbols.py`. Rerunning those
tests with an owned `--basetemp` passed as part of the 86-test result. A second full attempt was interrupted;
its partial output is retained as `interrupted-suite.txt`. The attempted CUDA_VISIBLE_DEVICES and
`wp.config.enable_cuda=False` controls were ineffective in this Warp build: those full attempts were
**CUDA-enabled**, not CPU-only receipts, and are not claimed as complete success. CI is the independent
GPU-free full-suite lane; its exact-head result is recorded on the PR separately.

Runtime retry when no unrelated Blender is active and GPU utilization is below 10%:

```powershell
python scripts/render_showcase.py --blender <supported-Blender-5.2-exe> --demos display,emitter_mixer --wait 0 --out <new-output-directory>
```

Require actual expected accelerated routes, labelled fallback rejection, paired initial-state equality,
successful influence samples and stationary negative control. Preserve failed attempts; do not relax
thresholds to pass. No claim about unchanged renderer correctness extends beyond the tests and original assets.
