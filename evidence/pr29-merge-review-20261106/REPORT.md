# PR29 merge review repairs — 6 November 2026

Starting head440318b8452a2c832b2d53888478920202f4b67a passed existing CI and
retains the earlier bounded installed Windows SDK evidence. Independent review
found cache-read integrity and native-output memory-budget gaps.

The reader now takes one bounded immutable snapshot for hash, ZIP inspection and
NumPy parsing. NPY headers must match exact field names, dtype, shape, C-order,
manifest count, payload length and budgets before deserialization. Duplicate ZIP
members and unsupported header versions reject. Manifest reads are bounded too.
A tiny valid-hash archive declaring a terabyte metadata array rejects before
NumPy allocation; replacement/growth regressions preserve the read bound.

Manual Snapshot and Load propagate the configured native budget. All snapshots
charge generated Geometry Nodes expansion; playback persists its admission cap,
also respects the current configured cap, and checks every population before
reading/replacing native geometry. Quickstart now describes Live and Bake.

Windows CPython3.14.3, Warp1.17.0, CUDA hidden:667tests passed,64skipped,
72subtests passed in55.33seconds. Ruff passes. Native Blender5.2.0 build
fbe6228777e7, isolated factory-startup profile, no GPU:actual mesh publication
passes; an oversized later frame rejects, clears stale geometry and hides it;
immutable cache roundtrip passes. See source-tests.log and native-cpu.json.

These new results qualify source and native CPU adapter changes. They do not
extend the earlier pinned Windows SDK evidence to Linux GPU, real-time speed,
arbitrary artist node graphs, long animations, motion blur or production use.
No existing user Blender or production installation was modified.
