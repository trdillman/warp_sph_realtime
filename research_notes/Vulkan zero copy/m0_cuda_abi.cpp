#include <cuda.h>
#include <stddef.h>
static_assert(sizeof(void*) == 8);
static_assert(sizeof(CUDA_EXTERNAL_MEMORY_HANDLE_DESC) == 104);
static_assert(sizeof(CUDA_EXTERNAL_MEMORY_BUFFER_DESC) == 88);
static_assert(offsetof(CUDA_EXTERNAL_MEMORY_HANDLE_DESC, reserved) == 36);
static_assert(offsetof(CUDA_EXTERNAL_MEMORY_BUFFER_DESC, reserved) == 20);
