"""Frame cache (P1.6 record and playback, P2.2 bake).

* ``disk``: the on-disk format, consistency rules, size estimate and default folder (NumPy only).
* ``gpu``: the VRAM ring and the solver adapter (Warp is passed in; never imported here).
* ``frame_store``: ring + disk behind a writer thread.
* ``bake``: bake planning (fresh / resume) and the chunked ``BakeJob``.
* ``age``: particle age (first cached frame of each id) without changing the file format (P2.3).
* ``output``: meshing defaults and export planning (pure, P2.3).
* ``blender_adapter``: the playback point cloud (with ``age``) and its timeline handler.
* ``blender_output``: the Geometry Nodes meshing preset and the USD / Alembic export (P2.3).

Everything except ``blender_adapter`` and ``blender_output`` imports without Blender or Warp. Nothing is imported eagerly here.
"""
