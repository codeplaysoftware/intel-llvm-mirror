//===- StructurizeCFG.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_AGGPEEL_H
#define LLVM_TRANSFORMS_SCALAR_AGGPEEL_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct AggPeelPass : PassInfoMixin<AggPeelPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_AGGPEEL_H
