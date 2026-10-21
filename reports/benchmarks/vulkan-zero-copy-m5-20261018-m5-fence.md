# Vulkan zero-copy M5 benchmarks (20261018-m5-fence)

Source: evidence/vulkan-zero-copy/20261018-m5-fence/runs/. Readback is a CPU copy, never zero-copy. Upload GPU time and Readback completion latency are unmeasured (no instrumentation); see M5_PROTOCOL.md.

| lane | kind | count | status | samples | drops | completed p50/p95/max ms | publish→draw p95 ms | pack GPU p95 ms | CPU submit p95 ms | frame p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vulkan | interop | 500000 | COMPLETE | 3000 | 0 | 0.906/1.160/2.768 | 2.273 | 0.189 | 0.659 | 3.305 |
| vulkan | interop | 1000000 | COMPLETE | 3000 | 0 | 1.154/1.446/2.817 | 2.457 | 0.074 | 0.938 | 3.456 |
| vulkan | interop | 2000000 | COMPLETE | 3000 | 0 | 1.993/2.275/3.535 | 2.792 | 0.045 | 1.058 | 3.736 |
| vulkan | interop | 4194304 | COMPLETE | 3000 | 0 | 10.941/11.664/14.633 | 3.823 | 0.185 | 1.491 | 4.420 |
| vulkan | readback | 500000 | COMPLETE | 3000 | 0 | unmeasured | 7.178 | n/a | 5.188 | 7.912 |
| vulkan | readback | 1000000 | COMPLETE | 3000 | 0 | unmeasured | 12.483 | n/a | 9.789 | 14.594 |
| vulkan | readback | 2000000 | COMPLETE | 3000 | 0 | unmeasured | 23.166 | n/a | 18.982 | 34.483 |
| vulkan | readback | 4194304 | COMPLETE | 3000 | 0 | unmeasured | 44.470 | n/a | 40.238 | 57.220 |
| opengl | readback | 500000 | COMPLETE | 3000 | 0 | unmeasured | 14.419 | n/a | 9.593 | 16.641 |
| opengl | readback | 1000000 | COMPLETE | 3000 | 0 | unmeasured | 17.992 | n/a | 13.017 | 20.161 |
| opengl | readback | 2000000 | COMPLETE | 3000 | 0 | unmeasured | 26.741 | n/a | 21.321 | 31.303 |
| opengl | readback | 4194304 | COMPLETE | 3000 | 0 | unmeasured | 44.020 | n/a | 37.860 | 55.267 |
