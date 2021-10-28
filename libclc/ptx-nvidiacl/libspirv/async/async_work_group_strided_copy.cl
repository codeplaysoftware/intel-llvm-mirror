//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include <spirv/spirv.h>
//todo: what is this include for?
#include <spirv/spirv_types.h>
#include <async.h>

int __nvvm_reflect(const char __constant *);

// 4 byte example case
_CLC_OVERLOAD _CLC_DEF event_t //todo do these ints originate from a gentype?
__spirv_GroupAsyncCopy(unsigned int scope, __attribute__((address_space(3))) int *dst,
                       const __attribute__((address_space(1))) int *src, size_t num_gentypes,
                       size_t stride, event_t event) {
      if (__nvvm_reflect("__CUDA_ARCH") >= 800) {

    size_t id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *         \
               __spirv_LocalInvocationId_z()) +                                \
              (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +    \
              __spirv_LocalInvocationId_x();
    // "ca" indicates we do caching at all levels rather than "cg" - global level caching only.
    __nvvm_cp_async_ca_shared_global_4(dst + id, src + id);
    __nvvm_cp_async_commit_group();
    }
    else{
      STRIDED_COPY( __attribute__((address_space(3))), __attribute__((address_space(1))), 1, stride);
    }
  return event;
}

// 8 byte example case
_CLC_OVERLOAD _CLC_DEF event_t
__spirv_GroupAsyncCopy(unsigned int scope, __attribute__((address_space(3))) double *dst,
                       const __attribute__((address_space(1))) double *src, size_t num_gentypes,
                       size_t stride, event_t event) {
      if (__nvvm_reflect("__CUDA_ARCH") >= 800) {

    size_t id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *         \
               __spirv_LocalInvocationId_z()) +                                \
              (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +    \
              __spirv_LocalInvocationId_x();

    __nvvm_cp_async_ca_shared_global_8(dst + id, src + id);
    __nvvm_cp_async_commit_group();
    }
    else{
      STRIDED_COPY( __attribute__((address_space(3))), __attribute__((address_space(1))), 1, stride);
      //STRIDED_COPY(1, stride);
    }
  return event;
}