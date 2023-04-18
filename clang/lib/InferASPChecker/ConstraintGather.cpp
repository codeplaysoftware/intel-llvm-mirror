//=== ConstraintGather.cpp - constraints collector for unifier ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This collects constraints a program applies to address spaces.
// The constraints are later unified by the unifier.
//
//===----------------------------------------------------------------------===//

#include "clang/InferASPChecker/ConstraintGather.h"
#include "clang/InferASPChecker/UnifyContext.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Basic/Builtins.h"
#include "clang/InferASPChecker/TypePrinter.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace unify;

void unify::StmtToHandle::print(llvm::raw_ostream &OS) const {
  for (clang::Stmt::StmtClass C : SeenStatments) {
    switch (C) {
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                                                    \
  case clang::Stmt::CLASS##Class: {                                            \
    OS << #CLASS << "\n";                                                      \
    break;                                                                     \
  }
#include "clang/AST/StmtNodes.inc"
    case clang::Stmt::NoStmtClass: {
      llvm_unreachable("Not a statement class, shouldn't hit this");
      break;
    }
    }
  }
}

/// Strip reference type if it is.
static GLorPRValue stripReference(GLorPRValue Ty) {
  if (AddressType *Ref = llvm::dyn_cast<AddressType>(Ty.Ty);
      Ref && Ref->isReference()) {
    return {Ref->getAddressSpace(), Ref->getElementType()};
  }
  return Ty;
}

clang::FunctionDecl *ConstraintGather::normalize(clang::FunctionDecl *FD) {
  clang::FunctionDecl *Def = FD->getDefinition();
  return Def ? Def : FD->getMostRecentDecl();
}

const clang::FunctionDecl *
ConstraintGather::normalize(const clang::FunctionDecl *FD) {
  return normalize(const_cast<clang::FunctionDecl *>(FD));
}

clang::Decl *ConstraintGather::normalize(clang::Decl *D) {
  D = D->getCanonicalDecl();
  if (clang::FunctionDecl *FD = llvm::dyn_cast<clang::FunctionDecl>(D)) {
    return normalize(FD);
  }
  return D->getMostRecentDecl();
}

const clang::Decl *ConstraintGather::normalize(const clang::Decl *D) {
  return normalize(const_cast<clang::Decl *>(D));
}

Type *ConstraintGather::ProcessBuiltin(clang::FunctionDecl *FD) {
  FunctionType *Fn = llvm::cast<FunctionType>(FnType);
  switch (FD->getBuiltinID()) {
  case clang::Builtin::BImemset:
  case clang::Builtin::BI__builtin_memset:
  case clang::Builtin::BI__builtin_assume: {
    // Nothing to do, can't infer anything
    break;
  }
  case clang::Builtin::BImemcpy:
  case clang::Builtin::BI__builtin_memcpy:
  case clang::Builtin::BImemmove:
  case clang::Builtin::BI__builtin_memmove:
  case clang::Builtin::BImempcpy:
  case clang::Builtin::BI__builtin_mempcpy: {
    // First and second argument are equal
    llvm::SmallVector<Type *, 3> Args{Fn->getArguments()};
    Args[1] = Args[0];
    FnType = Ctx->getFunctionType(Fn->getSourceType(), Fn->getThisType(),
                                  Fn->getReturnType(), Args);
    break;
  }
  case clang::Builtin::BIforward:
  case clang::Builtin::BImove:
  case clang::Builtin::BImove_if_noexcept: {
    // First and second argument are equals
    AddressType *RetAddr = llvm::cast<AddressType>(Fn->getReturnType());
    AddressType *Arg = llvm::cast<AddressType>(Fn->getArguments()[0]);
    FnType = Ctx->getFunctionType(
        Fn->getSourceType(), Fn->getThisType(),
        Ctx->getAddressLikeType(RetAddr->getSourceType(),
                                Arg->getAddressSpace(), Arg->getElementType()),
        Fn->getArguments());
    break;
  }
  default: {
    FD->dump();
    llvm_unreachable("Unhandled builtin");
  }
  }

  return FnType;
}

/// This routine sets implicit variable for the function:
/// If it is a method, it sets the this pointer type
/// if it is part of a lambda expression, it will pull capture variables type
/// from the parent context.
void ConstraintGather::FillImplicitVariable(clang::FunctionDecl *FD) {
  if (clang::CXXMethodDecl *MD = llvm::dyn_cast<clang::CXXMethodDecl>(FD)) {
    if (MD->isInstance())
      ThisType = TypeBuilder::getPRValue(Ctx, ASTCtx, MD->getThisType()).Ty;
    clang::CXXRecordDecl *RD = MD->getParent();
    if (RD->isLambda()) {
      AddressType *ThisTypeAsAddr = llvm::cast<AddressType>(ThisType);
      auto [ThisASP, ThisRecord] = ThisTypeAsAddr->get<RecordType>();
      RecordType::FieldList NewRecordFiledTypes{ThisRecord->getFields()};
      llvm::DenseMap<const clang::ValueDecl *, clang::FieldDecl *> Captures;
      clang::FieldDecl *ThisCapture;
      RD->getCaptureFields(Captures, ThisCapture);
      // Don't iterate using the DenseMap, this not stable for the tests...
      for (auto &C : RD->captures()) {
        if (C.capturesThis()) {
          ThisTypeExprType =
              TypeBuilder::getPRValue(Ctx, ASTCtx, ThisCapture->getType()).Ty;
          llvm::find_if(NewRecordFiledTypes, [&](const RecordType::Field &F) {
            return F.first == ThisCapture;
          })->second = ThisTypeExprType;
          continue;
        }
        auto [VD, FieldD] = *Captures.find(C.getCapturedVar());

        clang::LangAS VarAS = clang::LangAS::sycl_private;
        if (const clang::VarDecl *VarD = llvm::dyn_cast<clang::VarDecl>(VD))
          VarAS = VarD->isLocalVarDeclOrParm() && !VarD->isStaticLocal()
                      ? clang::LangAS::sycl_private
                      : clang::LangAS::opencl_constant;
        GLorPRValue T = {
            ThisASP, TypeBuilder::getPRValue(Ctx, ASTCtx, VD->getType()).Ty};
        if (VD->getType() != FieldD->getType()) {
          T.Ty = Ctx->getAddressLikeType(FieldD->getType(),
                                         Ctx->getAddressSpaceType(VarAS), T.Ty);
        }
        llvm::find_if(NewRecordFiledTypes, [&](const RecordType::Field &F) {
          return F.first == FieldD;
        })->second = T.Ty;

        DeclOrder.push_back(VD);
        pushDecl(VD, T);
      }
      ThisRecord =
          Ctx->getRecordType(ThisRecord->getSourceType(), NewRecordFiledTypes,
                             ThisRecord->getBases());
      ThisType = Ctx->getAddressLikeType(ThisTypeAsAddr->getSourceType(),
                                         ThisTypeAsAddr->getAddressSpace(),
                                         ThisRecord);
      pushDeclToRoot(RD, ThisRecord);
    }
    if (!ThisTypeExprType)
      ThisTypeExprType = ThisType;
  }
}

Type *ConstraintGather::ProcessFunctionImpl(clang::FunctionDecl *FD) {
  RetType = TypeBuilder::getPRValue(Ctx, ASTCtx, FD->getReturnType()).Ty;
  llvm::SmallVector<Type *, 8> Args;
  llvm::for_each(FD->parameters(), [&](clang::ParmVarDecl *PD) {
    Args.push_back(VisitVarDecl(PD)->Ty);
  });
  FnType =
      Ctx->getFunctionType(FD->getType().getTypePtr(), ThisType, RetType, Args);

  FillImplicitVariable(FD);

  if (FD->getBuiltinID()) {
    return ProcessBuiltin(FD);
  }

  if (FD->hasBody()) {
    BaseStmt::Visit(FD->getBody());
  }

  if (LocalUnifier.unify()) {
    RetType = LocalUnifier.getCannon(RetType);
    llvm::for_each(Args, [&](Type *&T) { T = LocalUnifier.getCannon(T); });
    if (ThisType)
      ThisType = LocalUnifier.getCannon(ThisType);
    FnType = Ctx->getFunctionType(FD->getType().getTypePtr(), ThisType, RetType,
                                  Args);
    pushDeclToRoot(FD, FnType);
    if (getASTContext().getLangOpts().SYCLDumpInferredAddressSpace) {
      FD->printQualifiedName(llvm::outs());
      llvm::outs() << " : ";
      TypePrinter::print(llvm::outs(), FnType);
      llvm::for_each(DeclOrder, [&](auto *D) {
        D->print(llvm::outs());
        llvm::outs() << " : ";
        TypePrinter::print(llvm::outs(), LocalUnifier.getCannon(*findType(D)));
      });
    }
    return FnType;
  }
  if (getASTContext().getLangOpts().SYCLDumpInferredAddressSpace) {
    llvm::outs() << FD->getNameAsString() << ": Inference failed\n";
  }
  return nullptr;
}

const ConstraintGather *
ConstraintGather::RecursiveProcessFunction(clang::FunctionDecl *FD) {
  FD = normalize(FD);
  ConstraintGather *PushScope = FuncLookup.allocateFunctionInfo(
      FD, *getUnifyContext(), getASTContext(), FuncLookup);
  PushScope->ProcessFunctionImpl(FD);
  return PushScope;
}

Type *ConstraintGather::ProcessFunction(clang::FunctionDecl *FD) {
  FD = normalize(FD);
  ConstraintGather *PushScope = FuncLookup.allocateFunctionInfo(FD, *this);
  return PushScope->ProcessFunctionImpl(FD);
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitCompoundStmt(clang::CompoundStmt *S) {
  for (auto *I : S->body())
    BaseStmt::Visit(I);
  return {};
}

llvm::Optional<GLorPRValue> ConstraintGather::VisitVarDecl(clang::VarDecl *D,
                                                           bool AssumeGV) {
  assert((!AssumeGV || !D->isLocalVarDeclOrParm()) && "Decl not a GV");
  InitContextRAII SetInitCtx{*this,
                             D->isLocalVarDeclOrParm() && !D->isStaticLocal()
                                 ? clang::LangAS::sycl_private
                                 : clang::LangAS::opencl_constant};
  GLorPRValue T =
      TypeBuilder::getGLValue(Ctx, ASTCtx, DeclContextASP, D->getType());

  if (!AssumeGV) {
    DeclOrder.push_back(D);
    pushDecl(D, T);
  } else {
    pushDeclToRoot(D, T);
  }
  if (!llvm::isa<clang::ParmVarDecl>(D) && D->hasInit()) {
    LocalUnifier.addConstraint(D, stripReference(T), D->getInit(),
                               *BaseStmt::Visit(D->getInit()));
  }
  return T;
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitEnumConstantDecl(clang::EnumConstantDecl *D) {
  if (clang::Expr *Init = D->getInitExpr()) {
    BaseStmt::Visit(Init);
  }
  return TypeBuilder::getPRValue(Ctx, ASTCtx, D->getType());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitDeclStmt(clang::DeclStmt *S) {
  for (auto *D : S->decls())
    BaseDecl::Visit(D);
  return {};
}

llvm::Optional<GLorPRValue> ConstraintGather::VisitorSYCLUniqueStableIdExpr(
    clang::SYCLUniqueStableIdExpr *S) {
  clang::QualType Res = S->getType();
  BaseStmt::Visit(S->getExpr());

  return Ctx->getAddressLikeType(
      ASTCtx.getPointerType(S->getType()),
      Ctx->getAddressSpaceType(clang::LangAS::opencl_constant),
      TypeBuilder::getPRValue(Ctx, ASTCtx, Res->getPointeeType()).Ty);
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitReturnStmt(clang::ReturnStmt *S) {
  if (S->getRetValue()) {
    GLorPRValue RetExpr = *BaseStmt::Visit(S->getRetValue());
    if (AddressType *AddressTy = llvm::dyn_cast<AddressType>(RetExpr.Ty)) {
      (void)AddressTy;
      assert(!AddressTy->isReference());
    }
    if (AddressType *AddrRetType = llvm::dyn_cast<AddressType>(RetType);
        AddrRetType && AddrRetType->isReference()) {
      RetExpr = Ctx->getAddressLikeType(AddrRetType->getSourceType(),
                                        RetExpr.ASP, RetExpr.Ty);
    }
    LocalUnifier.addConstraint(S->getRetValue(), RetType, S->getRetValue(),
                               RetExpr);
  }
  return RetType;
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitStringLiteral(clang::StringLiteral *S) {
  return StmtToType[S] = TypeBuilder::getGLValue(
             Ctx, ASTCtx, clang::LangAS::opencl_constant, S->getType());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitFloatingLiteral(clang::FloatingLiteral *S) {
  return StmtToType[S] = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitIntegerLiteral(clang::IntegerLiteral *S) {
  return StmtToType[S] = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *S) {
  return StmtToType[S] = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitConditionalOperator(clang::ConditionalOperator *S) {
  BaseStmt::Visit(S->getCond());
  GLorPRValue TrueTy = *BaseStmt::Visit(S->getTrueExpr());
  GLorPRValue FalseTy = *BaseStmt::Visit(S->getFalseExpr());
  LocalUnifier.addConstraint(S->getTrueExpr(), TrueTy, S->getFalseExpr(),
                             FalseTy);
  return StmtToType[S] = TrueTy;
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitBinaryOperator(clang::BinaryOperator *S) {
  switch (S->getOpcode()) {
  case clang::BO_PtrMemD:
  case clang::BO_PtrMemI:
    /* code */
    break;

  case clang::BO_And:
  case clang::BO_AndAssign:
  case clang::BO_Or:
  case clang::BO_OrAssign:
  case clang::BO_Xor:
  case clang::BO_XorAssign:
  case clang::BO_LAnd:
  case clang::BO_LOr:
  case clang::BO_Shl:
  case clang::BO_ShlAssign:
  case clang::BO_Shr:
  case clang::BO_ShrAssign:
  case clang::BO_Mul:
  case clang::BO_MulAssign:
  case clang::BO_Div:
  case clang::BO_DivAssign:
  case clang::BO_Rem:
  case clang::BO_RemAssign: {
    // Only takes fundamentals
    BaseStmt::Visit(S->getLHS());
    return StmtToType[S] = *BaseStmt::Visit(S->getRHS());
  }
  case clang::BO_Add:
  case clang::BO_AddAssign:
  case clang::BO_Sub:
  case clang::BO_SubAssign: {
    // Only takes fundamentals
    Type *TLHS = BaseStmt::Visit(S->getLHS())->Ty;
    Type *TRHS = BaseStmt::Visit(S->getRHS())->Ty;
    Type *Ret = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType()).Ty;
    StmtToType[S] = Ret;
    if (llvm::isa<FundamentalType>(TLHS) && llvm::isa<FundamentalType>(TRHS)) {
      return Ret;
    }
    if (llvm::isa<FundamentalType>(Ret)) {
      assert(S->getOpcode() == clang::BO_Sub && llvm::isa<AddressType>(TLHS) &&
             llvm::isa<AddressType>(TRHS));
      LocalUnifier.addConstraint(S->getLHS(), TLHS, S->getRHS(), TRHS);
      return Ret;
    }
    if (llvm::isa<AddressType>(TLHS)) {
      LocalUnifier.addConstraint(S->getLHS(), TLHS, S, Ret);
      return Ret;
    }
    LocalUnifier.addConstraint(S->getRHS(), TRHS, S, Ret);

    return Ret;
  }
  case clang::BO_LT:
  case clang::BO_GT:
  case clang::BO_LE:
  case clang::BO_GE:
  case clang::BO_EQ:
  case clang::BO_NE: {
    BaseStmt::Visit(S->getLHS());
    BaseStmt::Visit(S->getRHS());
    return StmtToType[S] = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
  }
  case clang::BO_Comma: {
    BaseStmt::Visit(S->getLHS());
    return StmtToType[S] = *BaseStmt::Visit(S->getRHS());
  }
  case clang::BO_Assign: {
    GLorPRValue TLHS = *BaseStmt::Visit(S->getLHS());
    GLorPRValue TRHS = *BaseStmt::Visit(S->getRHS());
    LocalUnifier.addConstraint(S->getLHS(), TLHS.Ty, S->getRHS(), TRHS.Ty);
    return StmtToType[S] = TLHS;
  }
  case clang::BO_Cmp:
    llvm_unreachable("Not implemented yet");
    break;
  default:
    break;
  }
  return {};
}

llvm::Optional<GLorPRValue> ConstraintGather::VisitCompoundAssignOperator(
    clang::CompoundAssignOperator *S) {
  return VisitBinaryOperator(S);
}

GLorPRValue ConstraintGather::ProcessCall(FunctionType *CalleeType,
                                          clang::CallExpr::arg_range Args) {
  for (auto [FnType, ArgExpr] :
       llvm::zip_equal(CalleeType->getArguments(), Args)) {
    LocalUnifier.addConstraint(nullptr, stripReference(FnType), ArgExpr,
                               *BaseStmt::Visit(ArgExpr));
  }

  return stripReference(CalleeType->getReturnType());
}

FunctionType *ConstraintGather::BindFunctionCall(clang::Decl *FD) {
  const ConstraintGather *FuncInfo = FuncLookup.lookupFunction(normalize(FD));
  if (!FuncInfo) {
    return llvm::cast<FunctionType>(
        RecursiveProcessFunction(llvm::cast<clang::FunctionDecl>(FD))->FnType);
  }
  return llvm::cast<FunctionType>(Ctx->getDerivedType(FuncInfo->FnType));
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitCXXConstructExpr(clang::CXXConstructExpr *S) {
  GLorPRValue CType = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());

  Type *This = Ctx->getAddressLikeType(ASTCtx.getPointerType(S->getType()),
                                       Ctx->getAddressSpaceType(DeclContextASP),
                                       CType.Ty);

  FunctionType *CtorFn = BindFunctionCall(S->getConstructor());
  LocalUnifier.addConstraint(nullptr, This, S, CtorFn->getThisType());

  ProcessCall(CtorFn, S->arguments());
  return StmtToType[S] = CType.Ty;
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *S) {
  return StmtToType[S] = *BaseStmt::Visit(S->getExpr());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *S) {
  return StmtToType[S] = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
}
llvm::Optional<GLorPRValue>
ConstraintGather::VisitCXXThisExpr(clang::CXXThisExpr *S) {
  return StmtToType[S] = ThisTypeExprType;
}
llvm::Optional<GLorPRValue>
ConstraintGather::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *S) {
  GLorPRValue BaseType = *BaseStmt::Visit(S->getBase());
  BaseStmt::Visit(S->getIdx());
  if (AddressType *BaseAddr = llvm::dyn_cast<AddressType>(BaseType.Ty))
    BaseType = {BaseAddr->getAddressSpace(), BaseAddr->getElementType()};
  else {
    if (ArrayOrVectorType *BaseAddr =
            llvm::dyn_cast<ArrayOrVectorType>(BaseType.Ty)) {
      BaseType = {BaseType.ASP, BaseAddr->getElementType()};
    } else {
      S->dump();
      llvm_unreachable("Unhandled yet");
    }
  }

  return StmtToType[S] = BaseType;
}
llvm::Optional<GLorPRValue>
ConstraintGather::VisitCallExpr(clang::CallExpr *S) {
  return StmtToType[S] =
             ProcessCall(BindFunctionCall(S->getCalleeDecl()), S->arguments());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *S) {
  FunctionType *Callee =
      llvm::cast<FunctionType>(BaseStmt::Visit(S->getCallee())->Ty);

  return StmtToType[S] = ProcessCall(Callee, S->arguments());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *S) {
  FunctionType *FnTy = BindFunctionCall(S->getCalleeDecl());
  clang::CallExpr::arg_range Args = S->arguments();
  // Member function operator are modeled as function calls with the this
  // pointer being the first argument. Bind this here.
  if (FnTy->getThisType()) {
    GLorPRValue ThisObj = *BaseStmt::Visit(*Args.begin());
    ThisObj = Ctx->getAddressLikeType(ASTCtx.getPointerType(S->getType()),
                                      ThisObj.ASP, ThisObj.Ty);
    LocalUnifier.addConstraint(S->getCalleeDecl(), FnTy->getThisType(),
                               *Args.begin(), ThisObj.Ty);
    Args = {++Args.begin(), Args.end()};
  }

  return StmtToType[S] = ProcessCall(FnTy, Args);
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitDeclRefExpr(clang::DeclRefExpr *S) {
  llvm::Optional<GLorPRValue> Ty;
  if (clang::VarDecl *GVD = llvm::dyn_cast<clang::VarDecl>(S->getDecl())) {
    Ty = findType(S);
    if (!Ty.has_value())
      Ty = VisitVarDecl(GVD, /*AssumeGV =*/true);
  } else {
    Ty = BaseDecl::Visit(S->getDecl());
  }

  return StmtToType[S] = stripReference(*Ty);
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitExtVectorElementExpr(clang::ExtVectorElementExpr *S) {
  GLorPRValue Ty = *BaseStmt::Visit(S->getBase());
  assert(!S->getType()->isPointerType());
  return S->getType()->isExtVectorType()
             ? Ty
             : GLorPRValue{
                   Ty.ASP,
                   llvm::cast<ArrayOrVectorType>(Ty.Ty)->getElementType()};
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitConstantExpr(clang::ConstantExpr *S) {
  return TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitExprWithCleanups(clang::ExprWithCleanups *S) {
  llvm::for_each(S->getObjects(),
                 [&](clang::ExprWithCleanups::CleanupObject Obj) {
                   if (Obj.is<clang::BlockDecl *>()) {
                     BaseDecl::Visit(Obj.get<clang::BlockDecl *>());
                   } else {
                     BaseStmt::Visit(Obj.get<clang::CompoundLiteralExpr *>());
                   }
                 });
  return StmtToType[S] = *BaseStmt::Visit(S->getSubExpr());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitInitListExpr(clang::InitListExpr *S) {
  if (S->isTransparent())
    return StmtToType[S] = *BaseStmt::Visit(S->getInit(0));
  const clang::Type *InitListExprTy =
      TypeBuilder::normalizeType(ASTCtx, S->getType());
  if (const clang::RecordType *R =
          llvm::dyn_cast<clang::RecordType>(InitListExprTy)) {
    const clang::CXXRecordDecl *RD =
        llvm::cast<clang::CXXRecordDecl>(R->getDecl());

    clang::ArrayRef<clang::Expr *> InitLists = S->inits();
    clang::ArrayRef<clang::Expr *>::iterator CurrentExpr = InitLists.begin();

    RecordType::BaseList Bases;
    llvm::for_each(RD->bases(), [&](const clang::CXXBaseSpecifier &D) {
      Bases.emplace_back(
          &D, llvm::cast<RecordType>(BaseStmt::Visit(*CurrentExpr)->Ty));
      CurrentExpr++;
    });
    RecordType::FieldList Fields;
    llvm::for_each(R->getDecl()->fields(), [&](clang::FieldDecl *D) {
      Fields.emplace_back(D, BaseStmt::Visit(*CurrentExpr)->Ty);
      CurrentExpr++;
    });

    return StmtToType[S] = Ctx->getRecordType(R, Fields, Bases);
  }
  if (llvm::isa<clang::ExtVectorType>(InitListExprTy) ||
      llvm::isa<clang::ArrayType>(InitListExprTy)) {
    GLorPRValue Res = TypeBuilder::getPRValue(Ctx, ASTCtx, S->getType());
    ArrayOrVectorType *VecTy = llvm::cast<ArrayOrVectorType>(Res.Ty);

    for (clang::Expr *E : S->inits()) {
      GLorPRValue Elt = *BaseStmt::Visit(E);
      LocalUnifier.addConstraint(S, VecTy->getElementType(), E, Elt.Ty);
    }
    return Res;
  }
  S->dump();
  llvm_unreachable("Unhandled init list form");
  return {};
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitLambdaExpr(clang::LambdaExpr *S) {
  llvm::Optional<GLorPRValue> Ty = findType(S->getLambdaClass());
  assert(Ty.has_value());

  RecordType *LambdaRecType = llvm::cast<RecordType>(Ty->Ty);

  llvm::DenseMap<const clang::ValueDecl *, clang::FieldDecl *> Captures;
  clang::FieldDecl *ThisCapture;
  S->getLambdaClass()->getCaptureFields(Captures, ThisCapture);
  for (auto [VD, FD] : Captures) {
    GLorPRValue VDTy = *findType(VD);
    if (VD->getType() != FD->getType()) {
      VDTy = Ctx->getAddressLikeType(FD->getType(), VDTy.ASP, VDTy.Ty);
    }
    LocalUnifier.addConstraint(VD, VDTy, FD,
                               LambdaRecType->getField(FD)->second);
  }

  return Ty->Ty;
}

llvm::Optional<GLorPRValue> ConstraintGather::VisitMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *S) {
  GLorPRValue T{Ctx->getAddressSpaceType(DeclContextASP),
                BaseStmt::Visit(S->getSubExpr())->Ty};
  return StmtToType[S] = T;
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitMemberExpr(clang::MemberExpr *S) {
  GLorPRValue CType = *BaseStmt::Visit(S->getBase());
  return llvm::TypeSwitch<clang::ValueDecl *, llvm::Optional<GLorPRValue>>(
             S->getMemberDecl())
      .Case<clang::FieldDecl>(
          [&](clang::FieldDecl *FD) -> llvm::Optional<GLorPRValue> {
            if (S->isArrow()) {
              AddressType *Ptr = llvm::cast<AddressType>(CType.Ty);
              CType = {Ptr->getAddressSpace(), Ptr->getElementType()};
            }
            RecordType *ClassType = llvm::cast<RecordType>(CType.Ty);
            Type *F = ClassType->findFieldType(FD);
            return StmtToType[S] = [&]() -> GLorPRValue {
              if (AddressType *FAddr = llvm::dyn_cast<AddressType>(F);
                  FAddr && FAddr->isReference()) {
                return {FAddr->getAddressSpace(), FAddr->getElementType()};
              }
              return {CType.ASP, F};
            }();
          })
      .Case<clang::CXXMethodDecl>([&](clang::CXXMethodDecl *MD) {
        FunctionType *MemberFunction = BindFunctionCall(MD);
        if (!S->isArrow()) {
          CType = Ctx->getAddressLikeType(ASTCtx.getPointerType(S->getType()),
                                          CType.ASP, CType.Ty);
        }
        assert(llvm::isa<AddressType>(CType.Ty));
        LocalUnifier.addConstraint(S, MemberFunction->getThisType(),
                                   S->getBase(), CType.Ty);
        return StmtToType[S] = MemberFunction;
      })
      .Default([&](const clang::ValueDecl *VD) -> llvm::Optional<GLorPRValue> {
        VD->dump();
        llvm_unreachable("Unknown knind");
        return {};
      });
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitParenExpr(clang::ParenExpr *S) {
  return StmtToType[S] = *BaseStmt::Visit(S->getSubExpr());
}

llvm::Optional<GLorPRValue> ConstraintGather::VisitSubstNonTypeTemplateParmExpr(
    clang::SubstNonTypeTemplateParmExpr *S) {
  return BaseStmt::Visit(S->getReplacement());
}

llvm::Optional<GLorPRValue>
ConstraintGather::VisitUnaryOperator(clang::UnaryOperator *S) {
  GLorPRValue Op = *BaseStmt::Visit(S->getSubExpr());

  llvm::Optional<GLorPRValue> Res;
  switch (S->getOpcode()) {
  case clang::UO_AddrOf: {
    assert(Op.ASP && "No ASP for an expected GLValue");
    Res = Ctx->getAddressLikeType(S->getType(), Op.ASP, Op.Ty);
    break;
  }
  case clang::UO_PostInc:
  case clang::UO_PostDec: {
    assert(Op.ASP && "No ASP for an expected GLValue");
    Res = Op.Ty;
    break;
  }
  case clang::UO_PreInc:
  case clang::UO_PreDec: {
    assert(Op.ASP && "No ASP for an expected GLValue");
    Res = Op;
    break;
  }
  case clang::UO_Deref: {
    assert(!Op.ASP && "ASP for an expected PRValue");
    AddressType *Addr = llvm::cast<AddressType>(Op.Ty);
    assert(Addr->getSourceType()->isPointerType() && "Deref a non pointer ?");
    Res = {Addr->getAddressSpace(), Addr->getElementType()};
    break;
  }
  case clang::UO_Plus:
  case clang::UO_Minus:
  case clang::UO_Not:
  case clang::UO_LNot: {
    Res = Op.Ty;
    break;
  }
  default:
    llvm_unreachable("Not handling extension right now");
    break;
  }
  return StmtToType[S] = *Res;
}

llvm::Optional<GLorPRValue> ConstraintGather::VisitIfStmt(clang::IfStmt *If) {
  if (llvm::Optional<clang::Stmt *> ActiveStmt =
          If->getNondiscardedCase(getASTContext())) {
    if (*ActiveStmt)
      BaseStmt::Visit(*ActiveStmt);
    return {};
  }

  return BaseStmt::VisitIfStmt(If);
}

llvm::Optional<GLorPRValue> ConstraintGather::VisitStmt(clang::Stmt *S) {
  for (clang::Stmt *SubStmt : S->children())
    if (SubStmt)
      BaseStmt::Visit(SubStmt);
  return {};
}
