//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

#define __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE_NV, TYPE_MANGLED_NV, OP, ADDR_SPACE, ORDER) \
switch(scope){ \
	case Subgroup: \
	case Workgroup: \
		  return __nvvm_atom_cta##ORDER##_##OP##_gen_##TYPE_MANGLED_NV((ADDR_SPACE TYPE_NV*)pointer, (TYPE_NV)value);  \
	case Device: \
		  return __nvvm_atom##ORDER##_##OP##_gen_##TYPE_MANGLED_NV((ADDR_SPACE TYPE_NV*)pointer, (TYPE_NV)value);  \
	case CrossDevice: \
	default: \
		  return __nvvm_atom_sys##ORDER##_##OP##_gen_##TYPE_MANGLED_NV((ADDR_SPACE TYPE_NV*)pointer, (TYPE_NV)value);  \
}

#define __CLC_NVVM_ATOMIC_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, NAME_MANGLED, ADDR_SPACE, ADDR_SPACE_MANGLED) \
_CLC_DECL TYPE NAME_MANGLED##PU3##ADDR_SPACE_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS1_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
    volatile ADDR_SPACE TYPE *pointer, enum Scope scope, enum MemorySemanticsMask semantics, \
    TYPE value) { \
		/* Semantics mask may include memory order, storage class and other info   \
          Memory order is stored in the lowest 5 bits */                          \
        unsigned int order = semantics & 0x1F; \
		switch (order) {                                                           \
			case None:                                                                 \
			  __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE_NV, TYPE_MANGLED_NV, OP, ADDR_SPACE, )  \
			case Acquire:                                                              \
			  __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE_NV, TYPE_MANGLED_NV, OP, ADDR_SPACE, _acquire)  \
			case Release:                                                              \
			  __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE_NV, TYPE_MANGLED_NV, OP, ADDR_SPACE, _release)  \
			default: \
			case AcquireRelease:                                                       \
			  __CLC_NVVM_ATOMIC_IMPL_ORDER(TYPE_NV, TYPE_MANGLED_NV, OP, ADDR_SPACE, _acq_rel)  \
			}                                                                          \
}

#define __CLC_NVVM_ATOMIC(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, NAME_MANGLED) \
__CLC_NVVM_ATOMIC_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, NAME_MANGLED, __global, AS1) \
__CLC_NVVM_ATOMIC_IMPL(TYPE, TYPE_MANGLED, TYPE_NV, TYPE_MANGLED_NV, OP, NAME_MANGLED, __local, AS3)
