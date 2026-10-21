@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /c /std:c++17 /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" "research_notes\Vulkan zero copy\m0_cuda_abi.cpp" /Fo"evidence\vulkan-zero-copy\20261017-contained-m0\cuda-abi.obj"
exit /b %errorlevel%
