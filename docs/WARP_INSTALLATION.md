# Managed Warp installation and compatibility

Chosen install target: NVIDIA Warp 1.17.0 PyPI, CUDA Toolkit 12.9.
Platform wheel hashes and sizes are immutable in warp_wheels.lock.json.

| Version | CUDA | Classification |
| --- | --- | --- |
| 1.17.0 PyPI | 12.9 | Locked candidate; hardware solver and Blender GUI proof separate |
| 1.16.x | unspecified | Uncertified; refuse automatic loading |
| 1.17.0+cu13 | 13.0 | Uncertified; refuse automatic loading |
| 1.18.0 PyPI | 13.4 | Uncertified; NVIDIA driver >=580 required |

Runtime gate loads only this extension's ownership-receipted Warp files.
CUDA 12.9 dynamic runtime DLL/so is separately required for OpenGL buffer
registration. NVIDIA Warp's built-in CUDA libraries alone do not prove GL interop.

Safety policy: explicit user download; respect Blender offline mode; wheel
SHA-256, size, metadata and archive member verification; no global pip,
no modification of other extension storage; preserve unmanaged files; never
replace loaded native binaries; repair staged and reverified on fresh startup;
rollback prior managed content on filesystem promotion failure.

References:
https://nvidia.github.io/warp/v1.17/user_guide/installation.html
https://nvidia.github.io/warp/v1.17/user_guide/runtime.html
https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__OPENGL.html
https://github.com/trdillman/liquid_warp_blender51_port/pull/19

Qualification 2026-10-14: Blender 5.2.0 LTS fbe6228777e7 Windows, driver 610.88, RTX 4080 SUPER, Warp 1.17.0 PyPI CUDA 12.9. Locked wheel SHA matched and native CUDA-graph/eager solver steps, direct pointer alias, CUDA/OpenGL GL buffer register/map/upload/unmap/destroy passed in a Blender GPU-initialized OpenGL context. Other CUDA minor versions, Linux platform native integration and different GPU hardware require separate certification.
