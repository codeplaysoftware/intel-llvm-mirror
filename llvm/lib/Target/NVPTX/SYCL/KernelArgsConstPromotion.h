//===- KernelArgsConstPromotion.cpp - TODO: JKB: --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass operates on SYCL kernels being compiled to CUDA. It rewrites the
// signature of the kernel, removing all the arguments, instead the arguments
// are used to construct a struct, where each kernel argument is a struct
// member. The pass then constructs a global variable of the struct type (in
// constant address space) and replaces the use of all the arguments with the
// corresponding members. The SYCL runtime is expected to fill in the struct
// with the data taken from appropriate kernel arguments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_KERNELARGSCONSTPROMOTION_H
#define LLVM_SYCL_KERNELARGSCONSTPROMOTION_H

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

ModulePass *createKernelArgsConstPromotionPass();

} // end namespace llvm

#endif // LLVM_SYCL_KERNELARGSCONSTPROMOTION_H
