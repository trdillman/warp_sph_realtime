# Vulkan zero-copy M5 benchmarks (20261126-requal)

Source: evidence/vulkan-zero-copy/20261126-requal/runs/. Readback is a CPU copy, never zero-copy. Upload GPU time and Readback completion latency are unmeasured (no instrumentation); see M5_PROTOCOL.md.

| lane | kind | count | status | samples | drops | completed p50/p95/max ms | publish→draw p95 ms | pack GPU p95 ms | CPU submit p95 ms | frame p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| vulkan | interop | 500000 | COMPLETE | 3000 | 0 | 1.140/1.606/2.794 | 3.715 | 0.429 | 1.036 | 4.333 |
| vulkan | interop | 1000000 | COMPLETE | 3000 | 0 | 1.285/1.726/2.633 | 3.319 | 0.592 | 1.018 | 4.094 |
| vulkan | interop | 2000000 | COMPLETE | 3000 | 0 | 2.096/2.530/3.448 | 3.532 | 0.243 | 1.054 | 4.256 |
| vulkan | interop | 4194304 | COMPLETE | 3000 | 0 | 7.810/9.898/11.164 | 3.697 | 0.187 | 1.194 | 4.486 |
| vulkan | readback | 500000 | COMPLETE | 3000 | 0 | unmeasured | 10.264 | n/a | 7.431 | 10.246 |
| vulkan | readback | 1000000 | COMPLETE | 3000 | 0 | unmeasured | 15.380 | n/a | 11.038 | 21.585 |
| vulkan | readback | 2000000 | COMPLETE | 3000 | 0 | unmeasured | 25.713 | n/a | 21.173 | 36.213 |
| vulkan | readback | 4194304 | COMPLETE | 3000 | 0 | unmeasured | 46.619 | n/a | 41.763 | 59.739 |
| opengl | readback | 500000 | COMPLETE | 3000 | 0 | unmeasured | 9.356 | n/a | 5.335 | 16.676 |
| opengl | readback | 1000000 | COMPLETE | 3000 | 0 | unmeasured | 15.328 | n/a | 11.176 | 17.434 |
| opengl | readback | 2000000 | COMPLETE | 3000 | 0 | unmeasured | 23.688 | n/a | 19.750 | 28.031 |
| opengl | readback | 4194304 | COMPLETE | 3000 | 0 | unmeasured | 53.072 | n/a | 45.554 | 57.534 |
