//=== UnifyContext.h - Context for the unifier                  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INFERASPCHECKER_UNIFYCONTEXT_H
#define LLVM_CLANG_INFERASPCHECKER_UNIFYCONTEXT_H

#include "Unifier.h"
#include "UnifyType.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"

#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"

namespace clang {
class DiagnosticsEngine;
}

namespace unify {

class Unifier;

/// @brief Unification context. Manage memory and ensure 2 equal types have a
/// unique memory address. This is achieved using the BumpPtrAllocator and
/// FoldingSet.
class UnifyContext {
  static constexpr unsigned AddressSpaceCount =
      static_cast<unsigned>(clang::LangAS::FirstTargetAddressSpace);
  Unifier RootUnifier;
  clang::ASTContext &ASTCtx;
  clang::DiagnosticsEngine &Diag;

  std::array<AddressSpace, AddressSpaceCount> AddressSpaceMap;
  llvm::FoldingSet<Type> FoldSet;
  llvm::BumpPtrAllocator PoolAllocation;

  template <typename U, typename... TS> U *PoolAllocate(TS... args) {
    return new (PoolAllocation) U(args...);
  }

  template <typename U, typename... TS> U *Internalize(TS... args) {
    llvm::FoldingSetNodeID Id;
    U Ty{args...};
    Ty.Profile(Id);

    void *InsertPos = nullptr;
    Type *InternalTy = FoldSet.FindNodeOrInsertPos(Id, InsertPos);
    if (InternalTy)
      return llvm::cast<U>(InternalTy);

    Type *Wrapped = PoolAllocate<U>(args...);

    FoldSet.InsertNode(Wrapped, InsertPos);

    return llvm::cast<U>(Wrapped);
  }

public:
  UnifyContext(clang::ASTContext &ASTCtx, clang::DiagnosticsEngine &Diag);

  Unifier *getUnifier() { return &RootUnifier; }

  clang::ASTContext &getASTContext() const { return ASTCtx; }
  clang::DiagnosticsEngine &getDiag() const { return Diag; }

  /// Build the same type as the input but with new slots.
  /// @param Ty The type to derive
  /// @param SubMap Substitution map
  /// @return the derived type
  Type *getDerivedType(Type *Ty,
                       llvm::DenseMap<Type *, Type *> *SubMap = nullptr);

  AddressSpace *getSlot();
  AddressSpace *getAddressSpaceType(clang::LangAS);
  AddressType *getAddressLikeType(const clang::Type *, AddressSpace *, Type *);
  AddressType *getAddressLikeType(clang::QualType QT, AddressSpace *A,
                                  Type *T) {
    return getAddressLikeType(QT.getTypePtr(), A, T);
  }
  ArrayOrVectorType *getArrayOrVectorType(const clang::Type *, Type *);
  ArrayOrVectorType *getArrayOrVectorType(clang::QualType QT, Type *T) {
    return getArrayOrVectorType(QT.getTypePtr(), T);
  }
  FundamentalType *getFundamentalType(const clang::Type *);
  FundamentalType *getFundamentalType(clang::QualType QT) {
    return getFundamentalType(QT.getTypePtr());
  }
  FunctionType *getFunctionType(const clang::Type *, Type *, Type *,
                                llvm::ArrayRef<Type *>);
  FunctionType *getFunctionType(clang::QualType QT, Type *ThisType, Type *R,
                                llvm::ArrayRef<Type *> Args) {
    return getFunctionType(QT.getTypePtr(), ThisType, R, Args);
  }
  RecordType *getRecordType(const clang::Type *,
                            llvm::ArrayRef<RecordType::Field>,
                            llvm::ArrayRef<RecordType::Base> Bases);
  RecordType *getRecordType(clang::QualType QT,
                            llvm::ArrayRef<RecordType::Field> Fields,
                            llvm::ArrayRef<RecordType::Base> Bases) {
    return getRecordType(QT.getTypePtr(), Fields, Bases);
  }
};
} // namespace unify

#endif
