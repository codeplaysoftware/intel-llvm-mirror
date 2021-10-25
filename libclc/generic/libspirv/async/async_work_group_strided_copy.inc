//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define STRIDED_COPY(DST_STRIDE, SRC_STRIDE)                                   \
  size_t size, id;                                                             \
  if (scope == Workgroup) {                                                    \
    size = __spirv_WorkgroupSize_x() * __spirv_WorkgroupSize_y() *             \
           __spirv_WorkgroupSize_z();                                          \
    id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *              \
          __spirv_LocalInvocationId_z()) +                                     \
         (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +         \
         __spirv_LocalInvocationId_x();                                        \
  } else {                                                                     \
    size = __spirv_SubgroupSize();                                             \
    id = __spirv_SubgroupLocalInvocationId();                                  \
  }                                                                            \
  size_t i;                                                                    \
                                                                               \
  for (i = id; i < num_gentypes; i += size) {                                  \
    dst[i * DST_STRIDE] = src[i * SRC_STRIDE];                                 \
  }

_CLC_OVERLOAD _CLC_DEF event_t
__spirv_GroupAsyncCopy(unsigned int scope, global __CLC_GENTYPE *dst,
                       const local __CLC_GENTYPE *src, size_t num_gentypes,
                       size_t stride, event_t event) {
  STRIDED_COPY(stride, 1);
  return event;
}

_CLC_OVERLOAD _CLC_DEF event_t
__spirv_GroupAsyncCopy(unsigned int scope, local __CLC_GENTYPE *dst,
                       const global __CLC_GENTYPE *src, size_t num_gentypes,
                       size_t stride, event_t event) {
  STRIDED_COPY(1, stride);
  return event;
}
