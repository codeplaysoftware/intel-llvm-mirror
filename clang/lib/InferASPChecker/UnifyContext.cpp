//=== UnifyContext.cpp - Context for the unifier                -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InferASPChecker/UnifyContext.h"

using namespace unify;

namespace {
class TypeDerivation : public UnifyTypeVisitor<TypeDerivation, Type *> {
public:
  using Base = UnifyTypeVisitor<TypeDerivation, Type *>;

  friend Base;
  using Base::visit;

  UnifyContext &Ctx;
  llvm::DenseMap<Type *, Type *> &SubMap;

  AddressSpace *visitAddressSpace(AddressSpace *Ty) {
    if (Ty->isSolved())
      return Ty;
    if (auto it = SubMap.find(Ty); it != SubMap.end())
      return llvm::cast<AddressSpace>(it->second);
    return llvm::cast<AddressSpace>(SubMap[Ty] = Ctx.getSlot());
  }

  Type *visitFundamentalType(FundamentalType *Ty) { return Ty; }

  Type *visitAddressType(AddressType *Ty) {
    AddressSpace *ASP = visitAddressSpace(Ty->getAddressSpace());
    Type *ElementType = visit(Ty->getElementType());
    if (ASP == Ty->getAddressSpace() && ElementType == Ty->getElementType())
      return Ty;
    return Ctx.getAddressLikeType(Ty->getSourceType(), ASP, ElementType);
  }

  Type *visitArrayOrVectorType(ArrayOrVectorType *Ty) {
    Type *ElementType = visit(Ty->getElementType());
    if (ElementType == Ty->getElementType())
      return Ty;
    return Ctx.getArrayOrVectorType(Ty->getSourceType(), ElementType);
  }

  Type *visitFunctionType(FunctionType *Ty) {
    Type *RetType = visit(Ty->getReturnType());
    llvm::SmallVector<Type *, 8> Args;
    Type *ThisType = nullptr;
    if (Ty->getThisType())
      ThisType = visit(Ty->getThisType());
    bool AllUnchanged =
        RetType == Ty->getReturnType() && ThisType == Ty->getThisType();
    llvm::for_each(Ty->getArguments(), [&](Type *A) {
      Args.push_back(visit(A));
      AllUnchanged = AllUnchanged && Args.back() == A;
    });
    if (AllUnchanged)
      return Ty;

    return Ctx.getFunctionType(Ty->getSourceType(), ThisType, RetType, Args);
  }

  Type *visitRecordType(RecordType *Ty) {
    bool AllUnchanged = true;
    RecordType::BaseList Bases;
    llvm::for_each(Ty->getBases(), [&](const RecordType::Base &A) {
      Bases.emplace_back(A.first, llvm::cast<RecordType>(visit(A.second)));
      AllUnchanged = AllUnchanged && Bases.back().second == A.second;
    });
    RecordType::FieldList Fields;
    llvm::for_each(Ty->getFields(), [&](const RecordType::Field &A) {
      Fields.emplace_back(A.first, visit(A.second));
      AllUnchanged = AllUnchanged && Fields.back().second == A.second;
    });
    if (AllUnchanged)
      return Ty;

    return Ctx.getRecordType(Ty->getSourceType(), Fields, Bases);
  }

  TypeDerivation(UnifyContext &Ctx, llvm::DenseMap<Type *, Type *> &SubMap)
      : Ctx{Ctx}, SubMap{SubMap} {}
};

} // namespace

UnifyContext::UnifyContext(clang::ASTContext &ASTCtx,
                           clang::DiagnosticsEngine &Diag)
    : RootUnifier{this}, ASTCtx{ASTCtx}, Diag{Diag} {
  for (size_t i = 0; i < AddressSpaceMap.size(); i++) {
    AddressSpaceMap[i] = AddressSpace{static_cast<clang::LangAS>(i)};
  }
}

Type *UnifyContext::getDerivedType(Type *Ty,
                                   llvm::DenseMap<Type *, Type *> *SubMap) {
  llvm::DenseMap<Type *, Type *> SubMapLocal;
  return TypeDerivation{*this, SubMap ? *SubMap : SubMapLocal}.visit(Ty);
}

AddressSpace *UnifyContext::getSlot() { return PoolAllocate<AddressSpace>(); }

AddressSpace *UnifyContext::getAddressSpaceType(clang::LangAS AS) {
  assert(AS != clang::LangAS::Default);
  return &AddressSpaceMap[static_cast<unsigned>(AS)];
}

AddressType *UnifyContext::getAddressLikeType(const clang::Type *ClangType,
                                              AddressSpace *AS,
                                              Type *ElementType) {
  return Internalize<AddressType>(ClangType, AS, ElementType);
}

ArrayOrVectorType *
UnifyContext::getArrayOrVectorType(const clang::Type *ClangType,
                                   Type *ElementType) {
  return Internalize<ArrayOrVectorType>(ClangType, ElementType);
}

FundamentalType *UnifyContext::getFundamentalType(const clang::Type *Ty) {
  return Internalize<FundamentalType>(Ty);
}

FunctionType *UnifyContext::getFunctionType(const clang::Type *ClangType,
                                            Type *ThisType, Type *Ret,
                                            llvm::ArrayRef<Type *> Args) {
  return Internalize<FunctionType>(ClangType, ThisType, Ret, Args);
}

RecordType *
UnifyContext::getRecordType(const clang::Type *ClangType,
                            llvm::ArrayRef<RecordType::Field> Fields,
                            llvm::ArrayRef<RecordType::Base> Bases) {
  return Internalize<RecordType>(ClangType, Fields, Bases);
}
