# SSFR PR #1 repair and runtime verification

Status: **VERIFIED for the three bounded review findings and the executed Windows GUI smoke lanes.** This is not a general visual-quality or sustained-performance certification.

Baseline PR head: `e6bd07cdbd650bd5f6d5fcdc7088efb474f3df8e`.
Existing local framebuffer-binding commits `ff4a773` and `a99c8a5` were preserved. No physics, simulation workload, modal operator, or shader mathematics changed in this repair.

> Layout note: this evidence was captured at `5d8a5ff`, before the package restructure. Files named here as `interop_bridge.py` / `viewport_draw.py` / `ssfr_shaders.py` / `sim_core.py` now live at `warp_sph_realtime/render/` and `warp_sph_realtime/sim/`; the probe scripts were updated to those paths, so recorded hashes will not match a rerun.

## Findings and repairs

| Finding | Independent verification | Repair |
|---|---|---|
| Multisampled scene-depth downscale | Real 4x MSAA negative control reproduces `GL_INVALID_OPERATION`; direct original path could clear obstacle depth and accumulate hidden thickness. | Resolve scene-only depth at full size into the scene-color FBO's new depth attachment, then downsample that single-sample depth before particle writes. Keep separate particle-depth storage. Failed transfer raises; never continue with cleared obstacle depth. |
| Assumed viewport depth format | Actual Blender 5.2 viewport texture is `GL_DEPTH32F_STENCIL8` (`0x8cad`); the first public GUI run rejected its copy into assumed D24/S8. | Inspect the source depth texture/renderbuffer format, allocate matching full/half targets, and resize if the format changes. |
| Incomplete state restoration | Driver-free controls first failed for mask, blend factors, exception exits and other bindings. Actual driver round-trip now passes. | Guard liquid and sphere drawing before allocation with `try/finally`; restore depth mask/function, separate RGB/alpha blend factors/equations, clear values, viewport, read/draw FBOs, program, VAO, touched texture units, active unit, color mask, scissor and enable bits. |
| Failed fragment compilation leaks VS | Failing regression observed the VS missing from deletion calls. | Program compilation deletes completed stages in `finally`, deletes failed programs, and context initialization deletes earlier programs and its empty VAO on later compilation failure. |
| Independent follow-up finding | Reviewer identified lazy VAO initialization changing context-global `GL_ARRAY_BUFFER_BINDING`. A regression using the actual lazy-VAO production path failed. | Capture/restore array-buffer binding after restoring the caller's VAO; regression passes. |

The zero-copy path remains `solver GPU arrays -> mapped CUDA/OpenGL VBO -> viewport`. GPU pixel readbacks occur only in the verifier. Scalar GL state queries in production do not read simulation buffers.

## Executed acceptance map

Environment: Windows, Blender **5.2.0 LTS** build `fbe6228777e7`, NVIDIA **RTX 4080 SUPER**, driver **610.88**, OpenGL backend, Warp **1.17.0**, actual `cuda:0` execution. Blender's extension builder produced a 32,084-byte ZIP, installed using `extension install-file` into a disposable profile. Both final GUI probes loaded `bl_ext.user_default.warp_sph_realtime` from that installed package.

Package SHA256: `95771909dca724faad14ddf417744648adf0f524afaa6a2414d4e7af4265db4a`.
The [manifest](ssfr-evidence/manifest.json) binds production source, probes, package, and raw receipts. Loaded source hashes match the working candidate and both installed GUI lanes.

| Claim | Result | Evidence |
|---|---|---|
| Full unit suite, including CUDA physics regressions | PASS, **33 tests** | [Unit log](ssfr-evidence/unit-tests-final.log); baseline 25 tests and failing-before-fix receipts retained alongside it. |
| Liquid rendering and scene-depth rejection | PASS, **8 controlled cases**: 0/4 samples x perspective/orthographic x 64x48/65x49 | [GPU controls](ssfr-evidence/gpu-controls.json). Visible summed thickness 12.0332 orthographic / 46.6274 perspective; hidden summed thickness exactly 0 in every case. |
| Forbidden MSAA downscale detection | PASS, actual driver negative control | Same GPU controls; each multisampled case checks `GL_INVALID_OPERATION`. |
| Real GL mask/blend/binding round-trip | PASS | Same GPU controls, `graphics_state_roundtrip`. Driver-free exception-exit test is separate. |
| Failed shader resource deletion | PASS, three invalid fragment attempts; all **6 shader objects** deleted according to actual `glIsShader` | Same GPU controls. Partial-context-init failure is also covered by a driver-free test. |
| Public simulation workflow | PASS, **15 mode/camera cases + 3 stop records** | [Workflow receipt](ssfr-evidence/gui-workflow.json): public start/stop operators, advancing solver steps, actual viewport callbacks, perspective/orthographic, liquid/spheres/points, actual area split and FBO resizing. |
| Unchanged workload | PASS | 1,000,000 particles, 2 substeps, CUDA graph enabled, stiffness 90000, time-step scale 0.4. `sim_core.py`, `demo_scenes.py`, `__init__.py` unchanged from the PR head. |
| Resource behavior | PASS for observed bounded checks | Solver telemetry stable at 72.479 MiB, renderer context count remained 1, CUDA VBO and solver released at each stop; controlled renderer destruction empties its resource dictionary. This is not a total-device-VRAM leak soak. |
| Independent source review | PASS after one follow-up repair | Two read-only review agents inspected the three fixes. One identified array-buffer restoration; it was repaired and re-reviewed. Reviewers read runtime receipts; they did not independently rerun GPU tests. |

Actual GUI images: [perspective liquid](ssfr-evidence/viewport-0-PERSP.png), [orthographic after resize](ssfr-evidence/viewport-1-ORTHO.png).

## Performance observations and limits

The final installed run recorded liquid callback medians of **2.13–9.14 ms**, spheres **0.40–1.07 ms**, and points **0.13–0.29 ms**. These are CPU wall times around the draw callback, not GPU timer-query measurements or complete frame rates. Solver telemetry spans **8.66–160.61 ms**, including startup/warm-up variation. Raw per-case measurements are retained; these short, changing-view runs do not establish a matched before/after speedup or regression.

The additional full-resolution scene depth costs one texture. Matching Blender's float depth also increases the pre-existing depth texture storage relative to the original assumed D24/S8. No particle count, substep count, physical parameter, smoothing setting, or solver algorithm was reduced to obtain a pass. No per-frame particle readback was added.

Untested lanes: real default framebuffer (ID 0) depth-format discovery, other GPUs/drivers/OSes, Vulkan/Metal, sustained stress/VRAM qualification, multi-context/window teardown, off-center projection quality, perspective particle-diameter accuracy and broader orthographic lighting/filter quality. These are not certified by the passing smoke tests.

Early attempts are retained locally: isolated bootstrap missing Warp, automation transport loss, verifier Windows GL 1.1 symbol-loader error, and actual viewport format rejection. None count as passing evidence. Final command-runner controls and workflow both exited 0 and wrote explicit PASS results.

## Rerun

Run the fast development lane from the repository root:

```powershell
python -m unittest discover -s tests -v
```

For the real GUI lane, use a disposable profile and an extension ZIP built from the candidate's tracked top-level Python files plus `blender_manifest.toml`. Supply the path to a Python package directory containing the compatible Warp installation; the probe records the loaded package and actual CUDA execution. For example, after assigning `$ssfrBlender`, `$ssfrZip`, and `$ssfrWarpPackages` to the target machine's paths:

```powershell
$ssfrProfile = Join-Path (Get-Location) 'evidence\isolated-profile'
$env:BLENDER_USER_RESOURCES = $ssfrProfile
$env:BLENDER_USER_CONFIG = Join-Path $ssfrProfile 'config'
$env:BLENDER_USER_SCRIPTS = Join-Path $ssfrProfile 'scripts'
$env:BLENDER_USER_DATAFILES = Join-Path $ssfrProfile 'datafiles'
& $ssfrBlender --background --factory-startup --command extension install-file -r user_default $ssfrZip
& $ssfrBlender --factory-startup --no-window-focus --gpu-backend opengl --python tests/run_ssfr_gui.py -- --case controls --warp-path $ssfrWarpPackages --evidence-dir evidence\rerun
& $ssfrBlender --factory-startup --no-window-focus --gpu-backend opengl --python tests/run_ssfr_gui.py -- --case workflow --warp-path $ssfrWarpPackages --evidence-dir evidence\rerun
```

Require explicit PASS plus empty errors in both output JSON files, a successful process exit, and matching source hashes. The runner rejects a candidate or configuration loaded outside the disposable profile, supplies a deadline, and quits only its own Blender instance. Do not use `--background` for the two GUI probe commands.
