"""Process-local defaults for the official native Windows ROCm runtime."""

LEGACY_NATIVE_ROCM_OVERRIDES = (
    'HSA_ENABLE_DXG_DETECTION', 'HSA_OVERRIDE_GFX_VERSION', 'ROCM_USE_WSL',
    'HSA_ENABLE_SDMA', 'HSA_FORCE_FINE_GRAIN_PCIE', 'HSA_DISABLE_REPLAY',
    'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF', 'PYTORCH_ALLOC_CONF',
    'GPU_MAX_HEAP_SIZE', 'GPU_MAX_ALLOC_PERCENT', 'GPU_SINGLE_ALLOC_PERCENT',
    'MESA_NO_ERROR', 'vblank_mode', 'LD_LIBRARY_PATH',
)

def ltx_environment(environment, gpu, system):
    """Avoid WSL/device/allocator overrides on the supported Windows HIP stack.

    The original environment is untouched. Explicit device isolation, auth and
    managed tool paths are preserved. Linux ROCm and CUDA retain their settings.
    """
    result = environment.copy()
    removed = []
    if system == 'win32' and gpu == 'amd':
        for key in LEGACY_NATIVE_ROCM_OVERRIDES:
            if key in result:
                removed.append(key)
                result.pop(key)
    return result, removed
