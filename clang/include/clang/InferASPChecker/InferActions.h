//=== InferActions.h - Address Space Inference Frontend Actions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INFERASPCHECKER_INFERACTIONS_H
#define LLVM_CLANG_INFERASPCHECKER_INFERACTIONS_H

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include <memory>

namespace clang {

/// Action adapter to perform address space inference
class InferASPAction : public WrapperFrontendAction {
protected:
  bool BeginInvocation(CompilerInstance &CI) override;

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

public:
  InferASPAction(std::unique_ptr<FrontendAction> WrappedAction);
};

} // namespace clang

#endif // LLVM_CLANG_INFERASPCHECKER_INFERACTIONS_H
