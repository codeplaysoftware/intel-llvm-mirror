//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/async/common.h>
#include <clc/clc.h>
#include <integer/popcount.h>
#include <spirv/spirv.h>

#define __CLC_BODY                                                             \
<../../../generic/libspirv/async/async_work_group_strided_copy.inc>
#define __CLC_GEN_VEC3
#include <clc/async/gentype.inc>

int __nvvm_reflect(const char __constant *);

#define __CLC_GROUP_CP_ASYNC_DST_GLOBAL(TYPE)                                  \
  _CLC_OVERLOAD _CLC_DEF event_t __spirv_GroupAsyncCopy(                       \
      unsigned int scope, __attribute__((address_space(1))) TYPE *dst,         \
      const __attribute__((address_space(3))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event) {                                          \
    STRIDED_COPY(__attribute__((address_space(1))),                            \
                 __attribute__((address_space(3))), stride, 1);                \
    return event;                                                              \
  }

__CLC_GROUP_CP_ASYNC_DST_GLOBAL(int);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(uint);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(float);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(long);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(ulong);
__CLC_GROUP_CP_ASYNC_DST_GLOBAL(double);

#define __CLC_GROUP_CP_ASYNC_MASKED_4(TYPE)                                    \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t __spirv_GroupAsyncCopyMasked( \
      unsigned int scope, __attribute__((address_space(3))) TYPE *dst,         \
      const __attribute__((address_space(1))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event, uint mask) {                               \
                                                                               \
    uint active = __nvvm_activemask();                                         \
    size_t size = __clc_native_popcount(active);                               \
    uint lane_mask_lt = __nvvm_read_ptx_sreg_lanemask_lt();                    \
    uint mask_id = lane_mask_lt & active;                                      \
    size_t id = __clc_native_popcount(mask_id);                                \
    size_t i;                                                                  \
    if (__nvvm_reflect("__CUDA_ARCH") >= 800) {                                \
      for (i = id; i < num_gentypes; i += size) {                              \
        __nvvm_cp_async_ca_shared_global_4(dst + i, src + i * stride);         \
      }                                                                        \
      __nvvm_cp_async_commit_group();                                          \
    } else {                                                                   \
      for (i = id; i < num_gentypes; i += size) {                              \
        dst[i * stride] = src[i];                                              \
      }                                                                        \
    }                                                                          \
    return event;                                                              \
  }

__CLC_GROUP_CP_ASYNC_MASKED_4(int);
__CLC_GROUP_CP_ASYNC_MASKED_4(uint);
__CLC_GROUP_CP_ASYNC_MASKED_4(float);

#define __CLC_GROUP_CP_ASYNC_MASKED_GLOBAL_DST_4(TYPE)                         \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t __spirv_GroupAsyncCopyMasked( \
      unsigned int scope, __attribute__((address_space(1))) TYPE *dst,         \
      const __attribute__((address_space(3))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event, uint mask) {                               \
    uint active = __nvvm_activemask();                                         \
    size_t size = __clc_native_popcount(active);                               \
    uint lane_mask_lt = __nvvm_read_ptx_sreg_lanemask_lt();                    \
    uint mask_id = lane_mask_lt & active;                                      \
    size_t id = __clc_native_popcount(mask_id);                                \
    size_t i;                                                                  \
    for (i = id; i < num_gentypes; i += size) {                                \
      dst[i * stride] = src[i];                                                \
    }                                                                          \
    return event;                                                              \

__CLC_GROUP_CP_ASYNC_MASKED_GLOBAL_DST_4(int);
__CLC_GROUP_CP_ASYNC_MASKED_GLOBAL_DST_4(uint);
__CLC_GROUP_CP_ASYNC_MASKED_GLOBAL_DST_4(float);

#define __CLC_GROUP_CP_ASYNC_4(TYPE)                                           \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t __spirv_GroupAsyncCopy(       \
      unsigned int scope, __attribute__((address_space(3))) TYPE *dst,         \
      const __attribute__((address_space(1))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event) {                                          \
    if (__nvvm_reflect("__CUDA_ARCH") >= 800) {                                \
      size_t id, size;                                                         \
      if (scope == Workgroup) {                                                \
        id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *          \
              __spirv_LocalInvocationId_z()) +                                 \
             (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +     \
             __spirv_LocalInvocationId_x();                                    \
        size = __spirv_WorkgroupSize_x() * __spirv_WorkgroupSize_y() *         \
               __spirv_WorkgroupSize_z();                                      \
      } else {                                                                 \
        size = __spirv_SubgroupSize();                                         \
        id = __spirv_SubgroupLocalInvocationId();                              \
      }                                                                        \
      size_t i;                                                                \
      for (i = id; i < num_gentypes; i += size) {                              \
        __nvvm_cp_async_ca_shared_global_4(dst + i, src + i * stride);         \
      }                                                                        \
      __nvvm_cp_async_commit_group();                                          \
    } else {                                                                   \
      STRIDED_COPY(__attribute__((address_space(3))),                          \
                   __attribute__((address_space(1))), 1, stride);              \
    }                                                                          \
    return event;                                                              \
  }

__CLC_GROUP_CP_ASYNC_4(int);
__CLC_GROUP_CP_ASYNC_4(uint);
__CLC_GROUP_CP_ASYNC_4(float);

#undef __CLC_GROUP_CP_ASYNC_4

#define __CLC_GROUP_CP_ASYNC_8(TYPE)                                           \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t __spirv_GroupAsyncCopy(       \
      unsigned int scope, __attribute__((address_space(3))) TYPE *dst,         \
      const __attribute__((address_space(1))) TYPE *src, size_t num_gentypes,  \
      size_t stride, event_t event) {                                          \
    if (__nvvm_reflect("__CUDA_ARCH") >= 800) {                                \
      size_t id, size;                                                         \
      if (scope == Workgroup) {                                                \
        id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *          \
              __spirv_LocalInvocationId_z()) +                                 \
             (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +     \
             __spirv_LocalInvocationId_x();                                    \
        size = __spirv_WorkgroupSize_x() * __spirv_WorkgroupSize_y() *         \
               __spirv_WorkgroupSize_z();                                      \
      } else {                                                                 \
        size = __spirv_SubgroupSize();                                         \
        id = __spirv_SubgroupLocalInvocationId();                              \
      }                                                                        \
      size_t i;                                                                \
      for (i = id; i < num_gentypes; i += size) {                              \
        __nvvm_cp_async_ca_shared_global_8(dst + i, src + i * stride);         \
      }                                                                        \
      __nvvm_cp_async_commit_group();                                          \
    } else {                                                                   \
      STRIDED_COPY(__attribute__((address_space(3))),                          \
                   __attribute__((address_space(1))), 1, stride);              \
    }                                                                          \
    return event;                                                              \
  }

__CLC_GROUP_CP_ASYNC_8(long);
__CLC_GROUP_CP_ASYNC_8(ulong);
__CLC_GROUP_CP_ASYNC_8(double);

#undef __CLC_GROUP_ASYNCCOPY_8
