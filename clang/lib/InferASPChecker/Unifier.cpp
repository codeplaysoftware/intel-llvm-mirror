//=== Unifier.cpp - Unifier engine                              -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InferASPChecker/Unifier.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/InferASPChecker/UnifyContext.h"

#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/raw_ostream.h"

using namespace unify;

namespace {

struct AssertSameType : public PairUnifyTypeVisitor<AssertSameType> {
  void visitType(Type *TyLHS, Type *TyRHS) {
    assert(TyLHS->getClass() == TyRHS->getClass());
  }

  void visitAddressType(AddressType *TyLHS, AddressType *TyRHS) {
    visit(TyLHS->getElementType(), TyRHS->getElementType());
  }
  void visitArrayOrVectorType(ArrayOrVectorType *TyLHS,
                              ArrayOrVectorType *TyRHS) {
    visit(TyLHS->getElementType(), TyRHS->getElementType());
  }

  void visitFunctionType(FunctionType *TyLHS, FunctionType *TyRHS) {
    visit(TyLHS->getReturnType(), TyRHS->getReturnType());

    for (auto [LParamTy, RParamTy] :
         llvm::zip_equal(TyLHS->getArguments(), TyRHS->getArguments())) {
      visit(LParamTy, RParamTy);
    }
  }

  void visitRecordType(RecordType *TyLHS, RecordType *TyRHS) {
    for (auto [LTy, RTy] : llvm::zip(TyLHS->getBases(), TyRHS->getBases())) {
      visit(LTy.second, RTy.second);
    }
    for (auto [LTy, RTy] : llvm::zip(TyLHS->getFields(), TyRHS->getFields())) {
      visit(LTy.second, RTy.second);
    }
  }
};

struct CollectSubConstraints {
  Unifier &UnifierEngine;
  Constraint C;

  explicit CollectSubConstraints(Unifier &UnifierEngine, Constraint C)
      : UnifierEngine{UnifierEngine}, C{C} {}

  void visit(Type *TyLHS, Type *TyRHS) {
    assert(TyLHS->getClass() == TyRHS->getClass());

    switch (TyLHS->getClass()) {
    case Type::TypeClass::AddressSpace:
      visit(llvm::cast<AddressSpace>(TyLHS), llvm::cast<AddressSpace>(TyRHS));
      break;
    case Type::TypeClass::Fundamental:
      visit(llvm::cast<FundamentalType>(TyLHS),
            llvm::cast<FundamentalType>(TyRHS));
      break;
    case Type::TypeClass::Address:
      visit(llvm::cast<AddressType>(TyLHS), llvm::cast<AddressType>(TyRHS));
      break;
    case Type::TypeClass::ArrayVector:
      visit(llvm::cast<ArrayOrVectorType>(TyLHS),
            llvm::cast<ArrayOrVectorType>(TyRHS));
      break;
    case Type::TypeClass::Function:
      visit(llvm::cast<FunctionType>(TyLHS), llvm::cast<FunctionType>(TyRHS));
      break;
    case Type::TypeClass::Record:
      visit(llvm::cast<RecordType>(TyLHS), llvm::cast<RecordType>(TyRHS));
      break;
    }
  }

  void visit(AddressSpace *TyLHS, AddressSpace *TyRHS) {
    UnifierEngine.addConstraint(C.getLHSCtx(), TyLHS, C.getRHSCtx(), TyRHS);
  }

  void visit(FundamentalType *TyLHS, FundamentalType *TyRHS) {}

  void visit(AddressType *TyLHS, AddressType *TyRHS) {
    UnifierEngine.addConstraint(C.getLHSCtx(), TyLHS->getAddressSpace(),
                                C.getRHSCtx(), TyRHS->getAddressSpace());
    UnifierEngine.addConstraint(C.getLHSCtx(), TyLHS->getElementType(),
                                C.getRHSCtx(), TyRHS->getElementType());
  }
  void visit(ArrayOrVectorType *TyLHS, ArrayOrVectorType *TyRHS) {
    UnifierEngine.addConstraint(C.getLHSCtx(), TyLHS->getElementType(),
                                C.getRHSCtx(), TyRHS->getElementType());
  }

  void visit(FunctionType *TyLHS, FunctionType *TyRHS) {
    UnifierEngine.addConstraint(C.getLHSCtx(), TyLHS->getReturnType(),
                                C.getRHSCtx(), TyRHS->getReturnType());

    for (auto [LParamTy, RParamTy] :
         llvm::zip_equal(TyLHS->getArguments(), TyRHS->getArguments())) {
      UnifierEngine.addConstraint(C.getLHSCtx(), LParamTy, C.getRHSCtx(),
                                  RParamTy);
    }
  }

  void visit(RecordType *TyLHS, RecordType *TyRHS) {
    for (auto [LTy, RTy] : llvm::zip(TyLHS->getBases(), TyRHS->getBases())) {
      UnifierEngine.addConstraint(C.getLHSCtx(), LTy.second, C.getRHSCtx(),
                                  RTy.second);
    }
    for (auto [LTy, RTy] : llvm::zip(TyLHS->getFields(), TyRHS->getFields())) {
      UnifierEngine.addConstraint(C.getLHSCtx(), LTy.second, C.getRHSCtx(),
                                  RTy.second);
    }
  }
};

struct ApplySubstitution : public UnifyTypeVisitor<ApplySubstitution, Type *> {
  using Base = UnifyTypeVisitor<ApplySubstitution, Type *>;
  UnifyContext *Ctx;
  const Unifier::SubstitutionType &Substitution;

  explicit ApplySubstitution(Unifier &UnifierEngine)
      : Ctx{UnifierEngine.getContext()},
        Substitution{UnifierEngine.getSubstitutions()} {}

  AddressSpace *visitAddressSpace(AddressSpace *Ty) {
    if (Ty->isSolved()) {
      return Ty;
    }
    Unifier::SubstitutionType::const_iterator ASSubIt = Substitution.find(Ty);
    while (ASSubIt != Substitution.end()) {
      Ty = llvm::cast<AddressSpace>(ASSubIt->second);
      ASSubIt = Substitution.find(Ty);
    }
    return Ty;
  }

  FundamentalType *visitFundamentalType(FundamentalType *Ty) { return Ty; }

  AddressType *visitAddressType(AddressType *Ty) {
    return Ctx->getAddressLikeType(Ty->getSourceType(),
                                   visitAddressSpace(Ty->getAddressSpace()),
                                   visit(Ty->getElementType()));
  }

  ArrayOrVectorType *visitArrayOrVectorType(ArrayOrVectorType *Ty) {
    return Ctx->getArrayOrVectorType(Ty->getSourceType(),
                                     visit(Ty->getElementType()));
  }

  FunctionType *visitFunctionType(FunctionType *Ty) {
    llvm::SmallVector<Type *, 8> Args;
    llvm::for_each(Ty->getArguments(),
                   [&](Type *TArg) { Args.push_back(visit(TArg)); });
    Type *ThisType = nullptr;
    if (Ty->getThisType())
      ThisType = visit(Ty->getThisType());
    return Ctx->getFunctionType(Ty->getSourceType(), ThisType,
                                visit(Ty->getReturnType()), Args);
  }

  RecordType *visitRecordType(RecordType *Ty) {
    RecordType::BaseList Bases;
    llvm::for_each(Ty->getBases(), [&](auto &TArg) {
      Bases.emplace_back(TArg.first,
                         llvm::cast<RecordType>(visit(TArg.second)));
    });
    RecordType::FieldList Fields;
    llvm::for_each(Ty->getFields(), [&](auto &TArg) {
      Fields.emplace_back(TArg.first, visit(TArg.second));
    });
    return Ctx->getRecordType(Ty->getSourceType(), Fields, Bases);
  }
};

} // namespace

clang::DiagnosticBuilder Unifier::Diag(clang::SourceLocation Loc,
                                       unsigned DiagID) const {
  return getDiag().Report(Loc, DiagID);
}
clang::DiagnosticBuilder Unifier::Diag(unsigned DiagID) const {
  return getDiag().Report(clang::SourceLocation{}, DiagID);
}

clang::DiagnosticsEngine &Unifier::getDiag() const { return Ctx->getDiag(); }

void Unifier::checkConstraints(Type *LHS, Type *RHS) const {
  AssertSameType{}.visit(LHS, RHS);
}

Type *Unifier::getCannonImpl(Type *Ty) {
  return ApplySubstitution{*this}.visit(Ty);
}

bool Unifier::unifyVariableImpl(Constraint::TypeCtx ASCtx, AddressSpace *AS,
                                Constraint::TypeCtx TCtx, Type *T) {
  auto ASSubIt = Substitution.find(AS);
  if (ASSubIt != Substitution.end()) {
    addConstraint(ASCtx, ASSubIt->second, TCtx, T);
    return true;
  }
  if (T->isTypeVariable()) {
    ASSubIt = Substitution.find(T);
    if (ASSubIt != Substitution.end()) {
      addConstraint(ASCtx, AS, TCtx, ASSubIt->second);
      return true;
    }
  }
  // OccursCheck(T).check(S) meaningful ?
  Substitution[AS] = T;
  return true;
}

bool Unifier::unifyImpl(Constraint &C) {
  Type *L = C.getLHS();
  Type *R = C.getRHS();

  if (L->isTypeVariable()) {
    return unifyVariableImpl(C.getLHSCtx(), llvm::dyn_cast<AddressSpace>(L),
                             C.getRHSCtx(), R);
  }
  if (R->isTypeVariable()) {
    return unifyVariableImpl(C.getRHSCtx(), llvm::dyn_cast<AddressSpace>(R),
                             C.getLHSCtx(), L);
  }
  assert(L->getClass() == R->getClass());
  if (L->getClass() == Type::TypeClass::AddressSpace) {
    AddressSpace *LAS = llvm::cast<AddressSpace>(L);
    AddressSpace *RAS = llvm::cast<AddressSpace>(R);
    if (LAS->getAddressSpace() != RAS->getAddressSpace()) {
      Diag(clang::diag::err_asp_conflict)
          << LAS->getAddressSpace() << RAS->getAddressSpace();
      if (C.getLHSCtx())
        Diag(C.getLHSCtx().getLocation(), clang::diag::note_asp_conflict_ctx)
            << LAS->getAddressSpace();
      if (C.getRHSCtx())
        Diag(C.getRHSCtx().getLocation(), clang::diag::note_asp_conflict_ctx)
            << RAS->getAddressSpace();
      return false;
    }
    return true;
  }
  // Collect sub constraints from type.
  CollectSubConstraints{*this, C}.visit(L, R);
  return true;
}

bool Unifier::unify() {
  while (!Constraints.empty()) {
    auto ConstraintIt = Constraints.begin();
    Constraint C = *ConstraintIt;
    Constraints.erase(ConstraintIt);

    if (!unifyImpl(C))
      return false;
  }

  return true;
}

void Unifier::dump() const {
  llvm::for_each(Constraints, [&](const Constraint &C) {
    llvm::outs() << "-----------------------\n";
    if (C.getLHSCtx()) {
      C.getLHSCtx().print(llvm::outs(), Ctx->getASTContext());
    }
    llvm::outs() << "\t";
    C.getLHS()->print(llvm::outs());
    llvm::outs() << "=\n";
    if (C.getRHSCtx()) {
      C.getRHSCtx().print(llvm::outs(), Ctx->getASTContext());
    }
    llvm::outs() << "\t";
    C.getRHS()->print(llvm::outs());
    llvm::outs() << "-----------------------\n";
  });
}
