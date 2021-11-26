//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// TODO: could adsorb this into a common.h depending on future usage etc
#define STRIDED_COPY_SUBGROUP_MASKED()                                         \
if (__nvvm_read_ptx_sreg_lanemask_eq() & Mask){                                \
  size_t size = __clc_native_popcount(Mask);                                   \
  size_t id =                                                                  \
      __clc_native_popcount(__nvvm_read_ptx_sreg_lanemask_lt() & Mask);        \
  size_t i;                                                                    \
  for (i = id; i < num_gentypes; i += size) {                                  \
    dst[i * stride] = src[i];                                                  \
  }                                                                            \
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONVERGENT event_t __spirv_GroupAsyncCopyMasked(
    unsigned int scope, __attribute__((address_space(1))) __CLC_GENTYPE *dst,
    const __attribute__((address_space(3))) __CLC_GENTYPE *src,
    size_t num_gentypes, size_t stride, event_t event, uint Mask) {
  STRIDED_COPY_SUBGROUP_MASKED();
  return event;
}

#undef STRIDED_COPY_SUBGROUP_MASKED

