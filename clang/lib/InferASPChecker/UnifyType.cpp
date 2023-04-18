//=== UnifyType.cpp - Type system for the unifier             ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"

#include "clang/InferASPChecker/TypePrinter.h"
#include "clang/InferASPChecker/UnifyType.h"

#include "llvm/Support/raw_ostream.h"

using namespace unify;

namespace {
class TypeProfile : public UnifyTypeVisitor<TypeProfile> {
public:
  using Base = UnifyTypeVisitor<TypeProfile>;

  using Base::visit;

  llvm::FoldingSetNodeID &ID;

  TypeProfile(llvm::FoldingSetNodeID &ID) : ID{ID} {}

  void visitType(Type *Ty) {
    ID.AddInteger(static_cast<uint32_t>(Ty->getClass()));
  }

  void visitAddressSpace(AddressSpace *Ty) {
    visitType(Ty);
    if (Ty->isSolved())
      ID.AddInteger(static_cast<uint32_t>(Ty->getAddressSpace()));
    else
      ID.AddPointer(this);
  }
  void visitClangType(ClangType *Ty) {
    visitType(Ty);
    ID.AddPointer(Ty->getSourceType());
  }
  void visitAddressType(AddressType *Ty) {
    visitClangType(Ty);
    visit(Ty->getAddressSpace());
    visit(Ty->getElementType());
  }
  void visitArrayOrVectorType(ArrayOrVectorType *Ty) {
    visitClangType(Ty);
    visit(Ty->getElementType());
  }
  void visitFunctionType(FunctionType *Ty) {
    visitClangType(Ty);
    visit(Ty->getReturnType());
    if (Ty->getThisType())
      visit(Ty->getThisType());
    llvm::for_each(Ty->getArguments(), [&](Type *A) { visit(A); });
  }
  void visitRecordType(RecordType *Ty) {
    visitClangType(Ty);
    llvm::for_each(Ty->getFields(), [&](auto &F) { visit(F.second); });
    llvm::for_each(Ty->getBases(), [&](auto &F) {
      ID.AddBoolean(F.first->isVirtual());
      visit(F.second);
    });
  }
};

} // namespace

void Type::print() const { print(llvm::outs()); }

void Type::print(llvm::raw_ostream &OS) const {
  TypePrinter::print(OS, const_cast<Type *>(this));
}

void Type::Profile(llvm::FoldingSetNodeID &ID) const {
  TypeProfile{ID}.visit(const_cast<Type *>(this));
}

void GLorPRValue::print() const { print(llvm::outs()); }

void GLorPRValue::print(llvm::raw_ostream &OS) const {
  TypePrinter::print(OS, *this);
}
