#pragma once
#include "platform.h"
#include <cuda.h>
#include <nvrtc.h>

namespace lm {
// Bind only when the GPU service starts. The executable has no CUDA imports.
class GpuApi {
#ifdef _WIN32
    SharedLibrary driver{"nvcuda.dll"};
    SharedLibrary builtins{executable_path().parent_path() / "nvrtc-builtins64_129.dll"};
    SharedLibrary compiler{executable_path().parent_path() / "nvrtc64_120_0.dll"};
#else
    SharedLibrary driver{"libcuda.so.1"};
    SharedLibrary builtins{executable_path().parent_path() / "libnvrtc-builtins.so.12.9"};
    SharedLibrary compiler{executable_path().parent_path() / "libnvrtc.so.12"};
#endif

  public:
    decltype(&::cuCtxSetCurrent) cuCtxSetCurrent = nullptr;
    decltype(&::cuCtxSetLimit) cuCtxSetLimit = nullptr;
    decltype(&::cuCtxSynchronize) cuCtxSynchronize = nullptr;
    decltype(&::cuDeviceGet) cuDeviceGet = nullptr;
    decltype(&::cuDeviceGetAttribute) cuDeviceGetAttribute = nullptr;
    decltype(&::cuDevicePrimaryCtxRetain) cuDevicePrimaryCtxRetain = nullptr;
    decltype(&::cuDevicePrimaryCtxSetFlags) cuDevicePrimaryCtxSetFlags = nullptr;
    decltype(&::cuFuncLoad) cuFuncLoad = nullptr;
    decltype(&::cuGetErrorString) cuGetErrorString = nullptr;
    decltype(&::cuInit) cuInit = nullptr;
    decltype(&::cuLaunchKernel) cuLaunchKernel = nullptr;
    decltype(&::cuMemFreeHost) cuMemFreeHost = nullptr;
    decltype(&::cuMemHostAlloc) cuMemHostAlloc = nullptr;
    decltype(&::cuMemHostGetDevicePointer) cuMemHostGetDevicePointer = nullptr;
    decltype(&::cuModuleGetFunction) cuModuleGetFunction = nullptr;
    decltype(&::cuModuleLoadData) cuModuleLoadData = nullptr;
    decltype(&::cuModuleUnload) cuModuleUnload = nullptr;
    decltype(&::cuStreamCreate) cuStreamCreate = nullptr;
    decltype(&::cuStreamDestroy) cuStreamDestroy = nullptr;
    decltype(&::cuStreamSynchronize) cuStreamSynchronize = nullptr;
    decltype(&::nvrtcCompileProgram) nvrtcCompileProgram = nullptr;
    decltype(&::nvrtcCreateProgram) nvrtcCreateProgram = nullptr;
    decltype(&::nvrtcDestroyProgram) nvrtcDestroyProgram = nullptr;
    decltype(&::nvrtcGetErrorString) nvrtcGetErrorString = nullptr;
    decltype(&::nvrtcGetPTX) nvrtcGetPTX = nullptr;
    decltype(&::nvrtcGetPTXSize) nvrtcGetPTXSize = nullptr;
    decltype(&::nvrtcGetProgramLog) nvrtcGetProgramLog = nullptr;
    decltype(&::nvrtcGetProgramLogSize) nvrtcGetProgramLogSize = nullptr;

    GpuApi() {
#define RB_STRING_IMPL(value) #value
#define RB_STRING(value) RB_STRING_IMPL(value)
#define RB_BIND(library, name) name = reinterpret_cast<decltype(name)>(library.symbol(RB_STRING(name)))
        RB_BIND(driver, cuCtxSetCurrent);
        RB_BIND(driver, cuCtxSetLimit);
        RB_BIND(driver, cuCtxSynchronize);
        RB_BIND(driver, cuDeviceGet);
        RB_BIND(driver, cuDeviceGetAttribute);
        RB_BIND(driver, cuDevicePrimaryCtxRetain);
        RB_BIND(driver, cuDevicePrimaryCtxSetFlags);
        RB_BIND(driver, cuFuncLoad);
        RB_BIND(driver, cuGetErrorString);
        RB_BIND(driver, cuInit);
        RB_BIND(driver, cuLaunchKernel);
        RB_BIND(driver, cuMemFreeHost);
        RB_BIND(driver, cuMemHostAlloc);
        RB_BIND(driver, cuMemHostGetDevicePointer);
        RB_BIND(driver, cuModuleGetFunction);
        RB_BIND(driver, cuModuleLoadData);
        RB_BIND(driver, cuModuleUnload);
        RB_BIND(driver, cuStreamCreate);
        RB_BIND(driver, cuStreamDestroy);
        RB_BIND(driver, cuStreamSynchronize);
        RB_BIND(compiler, nvrtcCompileProgram);
        RB_BIND(compiler, nvrtcCreateProgram);
        RB_BIND(compiler, nvrtcDestroyProgram);
        RB_BIND(compiler, nvrtcGetErrorString);
        RB_BIND(compiler, nvrtcGetPTX);
        RB_BIND(compiler, nvrtcGetPTXSize);
        RB_BIND(compiler, nvrtcGetProgramLog);
        RB_BIND(compiler, nvrtcGetProgramLogSize);
#undef RB_BIND
#undef RB_STRING
#undef RB_STRING_IMPL
    }
};
inline GpuApi &gpu_api() {
    static GpuApi api;
    return api;
}
} // namespace lm
