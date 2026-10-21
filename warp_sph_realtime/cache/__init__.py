"""Frame cache (P1.6 record and playback, P2.2 bake).

* ``disk``: the on-disk format, consistency rules, size estimate and default folder (NumPy only).
* ``gpu``: the VRAM ring and the solver adapter (Warp is passed in; never imported here).
* ``frame_store``: ring + disk behind a writer thread.
* ``bake``: bake planning (fresh / resume) and the chunked ``BakeJob``.
* ``blender_adapter``: the only ``bpy`` module: the playback point cloud and its timeline handler.

Everything except ``blender_adapter`` imports without Blender or Warp. Nothing is imported eagerly here.
"""
