//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include <spirv/spirv.h>
#include <spirv/spirv_types.h>
#include <async.h>

int __nvvm_reflect(const char __constant *);

#define __CLC_GROUP_CP_ASYNC_4(TYPE) \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t                  \
      __spirv_GroupAsyncCopy(unsigned int scope, __attribute__((address_space(3))) TYPE *dst, \
                       const __attribute__((address_space(1))) TYPE *src, size_t num_gentypes,\
                       size_t stride, event_t event) {\
      if (__nvvm_reflect("__CUDA_ARCH") >= 800) {\
    size_t id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *         \
               __spirv_LocalInvocationId_z()) +                                \
              (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +    \
              __spirv_LocalInvocationId_x();\
    __nvvm_cp_async_ca_shared_global_4(dst + id, src + id);\
    __nvvm_cp_async_commit_group();\
    }\
    else{\
      STRIDED_COPY( __attribute__((address_space(3))), __attribute__((address_space(1))), 1, stride);\
    }\
  return event;\
}

__CLC_GROUP_CP_ASYNC_4(int);
__CLC_GROUP_CP_ASYNC_4(float);

#undef __CLC_GROUP_CP_ASYNC_4

#define __CLC_GROUP_CP_ASYNC_8(TYPE) \
  _CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t                  \
      __spirv_GroupAsyncCopy(unsigned int scope, __attribute__((address_space(3))) TYPE *dst, \
                       const __attribute__((address_space(1))) TYPE *src, size_t num_gentypes,\
                       size_t stride, event_t event) {\
      if (__nvvm_reflect("__CUDA_ARCH") >= 800) {\
    size_t id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *         \
               __spirv_LocalInvocationId_z()) +                                \
              (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +    \
              __spirv_LocalInvocationId_x();\
    __nvvm_cp_async_ca_shared_global_8(dst + id, src + id);\
    __nvvm_cp_async_commit_group();\
    }\
    else{\
      STRIDED_COPY( __attribute__((address_space(3))), __attribute__((address_space(1))), 1, stride);\
    }\
  return event;\
}

__CLC_GROUP_CP_ASYNC_8(long);
__CLC_GROUP_CP_ASYNC_8(double);

#undef __CLC_GROUP_ASYNCCOPY_8
