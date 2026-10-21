# Example scenes

Small Python scripts that build a Custom WarpSPH scene in Blender with the add-on's own properties and operators
(roles on objects, `scene.warp_sph` settings, the **Fluid Preset** operator). They need Blender 5.2 with the add-on
enabled and Warp installed (see [the manual](../docs/manual/install.md)).

| Script | Scene |
|---|---|
| [`pour_into_box.py`](pour_into_box.py) | An emitter plane pours a stream into an open box made of four collider walls. |
| [`dam_break.py`](dam_break.py) | A water column collapses across a tank and floods past an obstacle. |
| [`moving_paddle.py`](moving_paddle.py) | A keyframed collider sweeps through a shallow pool. |
| [`viscous_drop.py`](viscous_drop.py) | A block of thick fluid (Viscous preset) falls into a pool. |

## Running one

* From the Text editor: open the script in Blender's Text editor and press Run Script. Then open the sidebar
  (`N`), tab **WarpSPH**, and press **Start Realtime Sim**.
* From the command line: `blender --python examples/dam_break.py`. Add `WARPSPH_EXAMPLE_START=1` to the
  environment to start the simulation automatically a second after the scene is built (it needs a window, so not
  with `-b`).

Each script deletes the objects named `Example ...` that an earlier run created, builds the scene in the current
file, sets **Scene Source** to **Custom**, and applies the **Water** preset (the viscous drop applies **Viscous**).
Change the constants at the top (`SPACING`, `PRESET`, frame numbers) to experiment. Smaller spacing means many more
particles; the panel warns when a scene may not fit in GPU memory. Edits to objects take effect on the next
**Stop** + **Start**.

`tests/test_examples.py` checks that every setting, operator and enum value the scripts use exists in the add-on.
