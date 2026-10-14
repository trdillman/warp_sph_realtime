# Contributing to WarpSPH Realtime (`CONTRIBUTING.md`)

Thank you for contributing to **WarpSPH Realtime**! Because this extension bridges **Blender 5.2's OpenGL Core Profile** and **NVIDIA Warp 1.16+ CUDA Graphs**, please follow the setup and verification guidelines below.

---

## 1. Developer Environment Setup

1. **Prerequisites:**
   * Blender `5.2.0 LTS` or newer ([Blender Developer Handbook](https://developer.blender.org/docs/handbook/extensions/addon_dev_setup/)).
   * NVIDIA GPU with CUDA 12.x or 13.x drivers.
   * Python `3.13` (matches Blender 5.2's internal Python interpreter).

2. **Install Dependencies:**
   ```bash
   pip install "warp-lang>=1.16.0" numpy pytest
   ```

3. **Symlink for Live Reloading in Blender 5.2:**
   Link the repository into your Blender user extensions directory so edits can be reloaded via `Blender > System > Reload Scripts`:
   * **Windows:** `%APPDATA%\Blender Foundation\Blender\5.2\extensions\user_default\warp_sph_realtime`
   * **Linux:** `~/.config/blender/5.2/extensions/user_default/warp_sph_realtime`

---

## 2. Coding Standards & Guidelines

* **Python Style:** Follow PEP 8 and the [Blender Python Style Guide](https://developer.blender.org/docs/handbook/guidelines/python/).
* **Warp Kernel Rules (`@wp.kernel` / `@wp.func`):**
  * Explicitly type all kernel arguments (`wp.array(dtype=wp.vec3)`, `float`, `int`, `wp.uint64`).
  * Guard all division operations against zero (`if r > 1e-6:`) to prevent NaNs from propagating through neighbor sums.
  * Keep all physics kernels compatible with `wp.ScopedCapture` (no host callbacks or dynamic allocations inside kernels).
* **Blender UI & Modal Safety:**
  * Always release CUDA graphics resources (`ZeroCopyGLBuffer52.destroy()`) and remove `SpaceView3D.draw_handler_remove` inside `cleanup()` on `ESC` or script reload to prevent VRAM leaks or stale OpenGL VAO handles.

---

## 3. Running Automated Tests

All pull requests must pass the automated test suite before merge. Our test suite is designed to run **both** inside a standard Python/CUDA terminal (mocking `bpy` and `gpu` where needed) and inside `blender --background --python`:

```bash
# Run all unit and integration tests via pytest
pytest tests/ -v

# Or via standard library unittest
python -m unittest discover -s tests -v
```

### What the Test Suite Covers
* **`tests/test_sim_core.py`:** Cubic spline kernel accuracy, gradient symmetry, $O(n)$ spatial hash + SoA reordering integrity, domain boundary clamping, NaN-free multi-step integration, and exact numerical equivalence between `use_graph=True` and `use_graph=False`.
* **`tests/test_submodules.py`:** IPBF variational prediction & VBD Hessian contraction, SDF sphere penetration resolution, whitewater spray/foam/bubble classification by neighbor count ($n_f$), surface particle splitting flags, DLPack bridge, and `.xyz` export.
* **`tests/test_blender_mock.py`:** Demo 1 & Demo 2 GPU particle initializers, regime presets (`WATER`, `MILK`, `SHOCKWAVE`), and `blender_manifest.toml` schema validation.

---

## 4. Pull Request Checklist
- [ ] New kernels maintain Structure of Arrays (SoA) layout and zero CPU readback.
- [ ] `pytest tests/ -v` passes with 0 errors.
- [ ]Tested interactively in Blender 5.2+ (`OpenGL` backend) at $\ge 1,000,000$ particles.
- [ ] `ARCHITECTURE.md` or `SPEC_AND_PLAN.md` updated if module interfaces changed.
