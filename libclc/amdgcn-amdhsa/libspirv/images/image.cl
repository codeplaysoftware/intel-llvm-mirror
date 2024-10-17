#include <clc/clc.h>
#include <spirv/spirv.h>

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef cl_khr_3d_image_writes
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

#ifdef _WIN32
#define MANGLE_FUNC_IMG_HANDLE(namelength, name, prefix, postfix)              \
  _Z##namelength##name##prefix##y##postfix
#else
#define MANGLE_FUNC_IMG_HANDLE(namelength, name, prefix, postfix)              \
  _Z##namelength##name##prefix##m##postfix
#endif

#define _CLC_DEFINE_IMAGE_BINDLESS_FETCH_BUILTIN(                              \
    elem_t, dimension, elem_t_mangled, vec_size, coord_mangled, coord_input,   \
    ...)                                                                       \
  _CLC_DEF elem_t MANGLE_FUNC_IMG_HANDLE(                                      \
      18, __spirv_ImageFetch, I##elem_t_mangled,                               \
      coord_mangled##ET_T0_T1_)(ulong imageHandle, coord_input) {              \
    return __ockl_image_sample_1D((unsigned int *)imageHandle, NULL,           \
                                  __VA_ARGS__);                                \
  }

_CLC_DEFINE_IMAGE_BINDLESS_FETCH_BUILTIN(float4 /*elem_t*/, 1 /*dimension*/,
                                         Dv4_f /*elem_t_mangled*/,
                                         v4f32 /*vec_size*/,
                                         i /*coord_mangled*/,
                                         int x /*coord_input*/,
                                         /*__VA_ARGS__*/
                                         x * sizeof(float4))
