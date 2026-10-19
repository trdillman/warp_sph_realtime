# Vulkan zero-copy M5 benchmarks (20261019-latency-fix)

Source: evidence/vulkan-zero-copy/20261019-latency-fix/runs/. Readback is a CPU copy, never zero-copy. Upload GPU time and Readback completion latency are unmeasured (no instrumentation); see M5_PROTOCOL.md.

| lane | kind | count | status | samples | drops | completed p50/p95/max ms | publish→draw p95 ms | pack GPU p95 ms | CPU submit p95 ms | frame p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vulkan | interop | 500000 | COMPLETE | 3000 | 0 | 1.050/1.496/2.594 | 3.789 | 0.230 | 0.886 | 4.963 |
| vulkan | interop | 1000000 | COMPLETE | 3000 | 0 | 1.190/1.603/3.221 | 3.225 | 0.214 | 0.863 | 4.386 |
| vulkan | interop | 2000000 | COMPLETE | 3000 | 0 | 1.865/2.219/4.052 | 3.554 | 0.056 | 1.200 | 4.439 |
| vulkan | interop | 4194304 | COMPLETE | 3000 | 0 | 3.798/5.203/13.199 | 4.953 | 0.186 | 1.945 | 6.163 |
| vulkan | readback | 500000 | COMPLETE | 3000 | 0 | unmeasured | 10.525 | n/a | 7.718 | 10.636 |
| vulkan | readback | 1000000 | COMPLETE | 3000 | 0 | unmeasured | 20.806 | n/a | 14.451 | 28.930 |
| vulkan | readback | 2000000 | COMPLETE | 3000 | 0 | unmeasured | 35.075 | n/a | 28.773 | 40.664 |
| vulkan | readback | 4194304 | INCOMPLETE_SAMPLES | 2785 | 0 | unmeasured | 72.548 | n/a | 65.789 | 71.303 |
| opengl | readback | 500000 | COMPLETE | 3000 | 0 | unmeasured | 13.144 | n/a | 6.903 | 17.165 |
| opengl | readback | 1000000 | COMPLETE | 3000 | 0 | unmeasured | 22.028 | n/a | 14.734 | 24.029 |
| opengl | readback | 2000000 | COMPLETE | 3000 | 0 | unmeasured | 38.228 | n/a | 30.762 | 41.468 |
| opengl | readback | 4194304 | INCOMPLETE_SAMPLES | 2998 | 0 | unmeasured | 68.534 | n/a | 59.189 | 69.252 |
