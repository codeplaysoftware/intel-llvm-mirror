//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLC_ASYNC
#define CLC_ASYNC
// Macro used by all data types / generic and nvidia, for async copy when arch < sm80

#define STRIDED_COPY(DST_AS, SRC_AS, DST_STRIDE, SRC_STRIDE)                   \
  size_t size = __spirv_WorkgroupSize_x() * __spirv_WorkgroupSize_y() *        \
                __spirv_WorkgroupSize_z();                                     \
  size_t id = (__spirv_WorkgroupSize_y() * __spirv_WorkgroupSize_x() *         \
               __spirv_LocalInvocationId_z()) +                                \
              (__spirv_WorkgroupSize_x() * __spirv_LocalInvocationId_y()) +    \
              __spirv_LocalInvocationId_x();                                   \
  size_t i;                                                                    \
                                                                               \
  for (i = id; i < num_gentypes; i += size) {                                  \
    dst[i * DST_STRIDE] = src[i * SRC_STRIDE];                                 \
  }

  #endif // CLC_ASYNC
