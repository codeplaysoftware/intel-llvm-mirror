//=== TypePrinter.h - Print Unifier type                   ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INFERASPCHECKER_TYPEPRINTER_H
#define LLVM_CLANG_INFERASPCHECKER_TYPEPRINTER_H

#include "clang/InferASPChecker/UnifyType.h"

namespace llvm {
class raw_ostream;
}

namespace unify {

/// @brief  Pretty printer for Unifier type.
class TypePrinter : public UnifyTypeVisitor<TypePrinter> {
  using Base = UnifyTypeVisitor<TypePrinter>;

  friend Base;
  using Base::visit;

  llvm::raw_ostream &OS;

  void visit(GLorPRValue);
  void visitAddressSpace(AddressSpace *);
  void visitFundamentalType(FundamentalType *);
  void visitAddressType(AddressType *);
  void visitArrayOrVectorType(ArrayOrVectorType *);
  void visitFunctionType(FunctionType *);
  void visitRecordType(RecordType *);

  TypePrinter(llvm::raw_ostream &OS) : OS{OS} {}

public:
  static void print(llvm::raw_ostream &OS, GLorPRValue Ty) {
    TypePrinter{OS}.visit(Ty);
    OS << "\n";
  }
  static void printNoEOL(llvm::raw_ostream &OS, GLorPRValue Ty) {
    TypePrinter{OS}.visit(Ty);
  }
};

} // namespace unify

#endif // LLVM_CLANG_INFERASPCHECKER_TYPEPRINTER_H
