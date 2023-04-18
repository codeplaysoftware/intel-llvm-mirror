//=== ConstraintGatherCast.cpp - constraints for the unifier  ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This collects constraints applied by cast expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/InferASPChecker/ConstraintGather.h"
#include "clang/InferASPChecker/UnifyContext.h"

#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Basic/Builtins.h"
#include "clang/InferASPChecker/TypePrinter.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace unify;

namespace {

/// @brief Collect constraints we can out of cast.
/// The unifier doesn't handled non matching type other than address spaces
/// (on purpose). Instead the complexity of handling this is pushed to the
/// context in which they appear.
struct CollectConstraintsForCast
    : public PairUnifyTypeVisitor<CollectConstraintsForCast> {
  using Base = PairUnifyTypeVisitor<CollectConstraintsForCast>;

  Unifier &UnifierEngine;
  Constraint::TypeCtx LHSCtx;
  Constraint::TypeCtx RHSCtx;

  CollectConstraintsForCast(Unifier &UnifierEngine, Constraint::TypeCtx LHSCtx,
                            Constraint::TypeCtx RHSCtx)
      : UnifierEngine{UnifierEngine}, LHSCtx{LHSCtx}, RHSCtx{RHSCtx} {}

  void visitAddressSpace(AddressSpace *TyLHS, AddressSpace *TyRHS) {
    UnifierEngine.addConstraint(LHSCtx, TyLHS, RHSCtx, TyRHS);
  }

  void visitFundamentalType(FundamentalType *TyLHS, FundamentalType *TyRHS) {}

  void visitAddressType(AddressType *TyLHS, AddressType *TyRHS) {
    UnifierEngine.addConstraint(LHSCtx, TyLHS->getAddressSpace(), RHSCtx,
                                TyRHS->getAddressSpace());
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

// Build a Base To Derive or Derive To Base type.
struct BuildBaseToDeriveType
    : public PairUnifyTypeVisitor<BuildBaseToDeriveType, Type *> {
  using Base = PairUnifyTypeVisitor<BuildBaseToDeriveType, Type *>;

  using Base::visit;

  unify::UnifyContext *Ctx;
  clang::ASTContext &ASTCtx;
  clang::CastExpr *Expr;

  BuildBaseToDeriveType(unify::UnifyContext *Ctx, clang::ASTContext &ASTCtx,
                        clang::CastExpr *Expr)
      : Ctx{Ctx}, ASTCtx{ASTCtx}, Expr{Expr} {}

  GLorPRValue visit(GLorPRValue LHS, GLorPRValue RHS) {
    if (LHS.ASP && RHS.ASP)
      return GLorPRValue{llvm::cast<AddressSpace>(visit(LHS.ASP, RHS.ASP)),
                         visit(LHS.Ty, RHS.Ty)};
    return GLorPRValue{visit(LHS.Ty, RHS.Ty)};
  }

  Type *visitType(Type *LHS, Type *RHS) { return RHS; }

  Type *visitAddressType(AddressType *LHS, AddressType *RHS) {
    return Ctx->getAddressLikeType(
        LHS->getSourceType(), RHS->getAddressSpace(),
        visit(LHS->getElementType(), RHS->getElementType()));
  }

  Type *visitArrayOrVectorType(ArrayOrVectorType *LHS, ArrayOrVectorType *RHS) {
    return Ctx->getArrayOrVectorType(
        LHS->getSourceType(),
        visit(LHS->getElementType(), RHS->getElementType()));
  }

  RecordType *rebuildRecordForBaseToDerived(
      RecordType *LHS, RecordType *RHS,
      llvm::iterator_range<clang::CastExpr::path_iterator> CastPathIt) {
    assert(CastPathIt.begin() != CastPathIt.end());
    RecordType::BaseList Bases{LHS->getBases()};
    const clang::CXXBaseSpecifier *BS = *CastPathIt.begin();
    CastPathIt = {CastPathIt.begin() + 1, CastPathIt.end()};
    RecordType::BaseList::iterator it = llvm::find_if(
        Bases, [&](const RecordType::Base &B) { return B.first == BS; });
    assert(it != Bases.end() && "Base not found ?");
    if (CastPathIt.begin() == CastPathIt.end())
      it->second = RHS;
    else
      it->second = rebuildRecordForBaseToDerived(it->second, RHS, CastPathIt);
    return Ctx->getRecordType(LHS->getSourceType(), LHS->getFields(), Bases);
  }

  Type *visitRecordType(RecordType *LHS, RecordType *RHS) {
    if (Expr->getCastKind() == clang::CK_BaseToDerived)
      return rebuildRecordForBaseToDerived(LHS, RHS, Expr->path());
    RecordType *CurrentBase = RHS;
    for (const clang::CXXBaseSpecifier *BS : Expr->path()) {
      RecordType::BaseList::const_iterator it = llvm::find_if(
          CurrentBase->getBases(),
          [&](const RecordType::Base &B) { return B.first == BS; });
      assert(it != CurrentBase->getBases().end() && "Base not found ?");
      CurrentBase = it->second;
    }
    return CurrentBase;
  }
};
} // namespace

llvm::Optional<GLorPRValue>
ConstraintGather::VisitCastExpr(clang::CastExpr *S) {
  GLorPRValue OperandTy = *BaseStmt::Visit(S->getSubExpr());
  switch (S->getCastKind()) {
  case clang::CK_LValueToRValue: {
    return StmtToType[S] = OperandTy.Ty;
  }
  case clang::CK_ConstructorConversion:
    [[fallthrough]];
  case clang::CK_UserDefinedConversion:
    [[fallthrough]];
  case clang::CK_NoOp: {
    return StmtToType[S] = OperandTy;
  }
  case clang::CK_BitCast: {
    assert(S->isPRValue());
    GLorPRValue CastRes = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
    CollectConstraintsForCast{LocalUnifier, S, S->getSubExpr()}.visit(
        CastRes, OperandTy);
    return StmtToType[S] = CastRes;
  }
  case clang::CK_AddressSpaceConversion: {
    GLorPRValue ASPCast =
        OperandTy.ASP
            ? TypeBuilder::getGLValue(
                  Ctx, ASTCtx, OperandTy.ASP->getAddressSpace(), S->getType())
            : TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
    LocalUnifier.addConstraint(S, ASPCast, S->getSubExpr(), OperandTy);
    return StmtToType[S] = ASPCast;
  }
  case clang::CK_ArrayToPointerDecay: {
    ArrayOrVectorType *ArrTy = llvm::cast<ArrayOrVectorType>(OperandTy.Ty);
    return StmtToType[S] = {Ctx->getAddressLikeType(S->getType(), OperandTy.ASP,
                                                    ArrTy->getElementType())};
  }
  case clang::CK_BaseToDerived:
    [[fallthrough]];
  case clang::CK_DerivedToBase:
    [[fallthrough]];
  case clang::CK_UncheckedDerivedToBase: {
    GLorPRValue CastRes = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
    if (OperandTy.ASP)
      CastRes.ASP = OperandTy.ASP;
    return StmtToType[S] =
               BuildBaseToDeriveType{Ctx, ASTCtx, S}.visit(CastRes, OperandTy);
  }
  case clang::CK_NullToPointer:
    [[fallthrough]];
  case clang::CK_IntegralToPointer:
    [[fallthrough]];
  case clang::CK_PointerToIntegral:
    [[fallthrough]];
  case clang::CK_PointerToBoolean:
    [[fallthrough]];
  case clang::CK_ToVoid:
    [[fallthrough]];
  case clang::CK_IntegralCast:
    [[fallthrough]];
  case clang::CK_IntegralToBoolean:
    [[fallthrough]];
  case clang::CK_IntegralToFloating:
    [[fallthrough]];
  case clang::CK_FloatingToIntegral:
    [[fallthrough]];
  case clang::CK_FloatingToBoolean:
    [[fallthrough]];
  case clang::CK_BooleanToSignedIntegral:
    [[fallthrough]];
  case clang::CK_FloatingCast: {
    return StmtToType[S] = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
  }
  case clang::CK_VectorSplat: {
    return StmtToType[S] =
               Ctx->getArrayOrVectorType(S->getType(), OperandTy.Ty);
  }
  default: {
    S->dump();
    llvm_unreachable("Not Implemented yet");
    break;
  }
  }
  return {};
}
