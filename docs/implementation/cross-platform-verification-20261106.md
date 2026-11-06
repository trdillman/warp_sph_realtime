# Linux then Windows verification checkpoint

Base primary6a79892 (runtimepayloadf0d60a5). Linux exact30tests passed;2additionalfailurecleanupchecks found stale.writer.lock after rejectedJSONsignature. Only productionchange moves signaturevalidation beforefilesystemsideeffects. Exact correctednative_sequence.py SHA d61e7e2573c1776ada52a403b26afa059c3fcd26eeaf0cfa0d2c85dbc22d7fc0 passes32Linuxchecks and26focusedWindowschecks. No physics/liveGPU/API/schema changes.

Linuxruntime lacks Blender/NVIDIAdevice; official download/DNS unavailable. NativeLinuxGN/render/CUDA remainunrun and explicitlynotqualified. Windowsuses exact canonicalBlender5.2fbe6228777e7/Warp1.17/RTX4080SUPER/Vulkan, isolatedprofile/installedextension, sharedGPUlease.

Complete acceptance: actualLiveGN+8frameprivatebake+cancel/live-statepreservation; filecorruption+last-goodbehavior; standalonewithoutSDK/Warp/originalcache; independentall-frameattributes/IDs/centroids/radii; actualfullanimationrender; cleanedownedprocesses anddurablehash-boundcheckpoint. Mainunchanged, no push or productioninstall. Primaryqualificationrun andours are serialized, not duplicate writerownership.
