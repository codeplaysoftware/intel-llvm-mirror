//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>
#include <atomic_helpers.h>

__CLC_NVVM_ATOMIC(int, i, int, i, sub, _Z21__spirv_AtomicISubEXT)
__CLC_NVVM_ATOMIC(long, l, long, l, sub, _Z21__spirv_AtomicISubEXT)

#undef __CLC_NVVM_ATOMIC_TYPES
#undef __CLC_NVVM_ATOMIC
#undef __CLC_NVVM_ATOMIC_IMPL
