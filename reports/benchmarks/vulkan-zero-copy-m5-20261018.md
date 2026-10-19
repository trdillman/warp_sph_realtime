# Vulkan zero-copy M5 benchmarks (2026-10-18)

Source: evidence/vulkan-zero-copy/20261018-m2plus/runs/. Readback is a CPU copy, never zero-copy. Upload GPU time and Readback completion latency are unmeasured (no instrumentation); see M5_PROTOCOL.md.

| lane | kind | count | status | samples | drops | completed p50/p95/max ms | publish→draw p95 ms | pack GPU p95 ms | CPU submit p95 ms | frame p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vulkan | interop | 500000 | COMPLETE | 3000 | 0 | 3.095/3.864/7.699 | 2.633 | 0.216 | 0.330 | 3.730 |
| vulkan | interop | 1000000 | COMPLETE | 3000 | 0 | 3.119/4.175/42.523 | 2.765 | 0.219 | 0.307 | 3.679 |
| vulkan | interop | 2000000 | COMPLETE | 3000 | 0 | 3.494/4.447/38.083 | 2.782 | 0.051 | 0.286 | 3.793 |
| vulkan | interop | 4194304 | COMPLETE | 3000 | 0 | 9.719/11.030/12.511 | 3.640 | 0.185 | 0.281 | 4.048 |
| vulkan | readback | 500000 | COMPLETE | 3000 | 0 | unmeasured | 10.935 | n/a | 8.095 | 10.003 |
| vulkan | readback | 1000000 | COMPLETE | 3000 | 0 | unmeasured | 16.367 | n/a | 13.031 | 16.498 |
| vulkan | readback | 2000000 | COMPLETE | 3000 | 0 | unmeasured | 27.733 | n/a | 23.619 | 36.757 |
| vulkan | readback | 4194304 | COMPLETE | 3000 | 0 | unmeasured | 49.878 | n/a | 45.592 | 60.564 |
| opengl | readback | 500000 | COMPLETE | 3000 | 0 | unmeasured | 15.973 | n/a | 11.405 | 16.641 |
| opengl | readback | 1000000 | COMPLETE | 3000 | 0 | unmeasured | 20.840 | n/a | 16.198 | 18.936 |
| opengl | readback | 2000000 | COMPLETE | 3000 | 0 | unmeasured | 28.386 | n/a | 23.631 | 29.920 |
| opengl | readback | 4194304 | COMPLETE | 3000 | 0 | unmeasured | 46.587 | n/a | 40.105 | 54.681 |
