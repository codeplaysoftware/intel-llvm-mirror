//=== ConstraintGather.h - constraints collector for unifier ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INFERASPCHECKER_CONSTRAINTGATHER_H
#define LLVM_CLANG_INFERASPCHECKER_CONSTRAINTGATHER_H

#include "UnifyContext.h"
#include "UnifyType.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeVisitor.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"

namespace llvm {
class raw_ostream;
}

namespace unify {

class StmtToHandle : public clang::RecursiveASTVisitor<StmtToHandle> {
  using Base = clang::RecursiveASTVisitor<StmtToHandle>;
  llvm::SmallSet<clang::Stmt::StmtClass, 16> SeenStatments;

public:
  void ProcessFunction(clang::FunctionDecl *FD) { Base::TraverseDecl(FD); }

  bool VisitStmt(clang::Stmt *S) {
    SeenStatments.insert(S->getStmtClass());
    return true;
  }

  void print(llvm::raw_ostream &OS) const;
};

/// @brief Small type visitor to transform a clang type into a Unifier type.
class TypeBuilder : public clang::TypeVisitor<TypeBuilder, unify::Type *> {
  using Base = clang::TypeVisitor<TypeBuilder, unify::Type *>;

  friend Base;

  unify::UnifyContext *Ctx;
  clang::ASTContext &ASTCtx;
  clang::QualType Ty;

  static unify::AddressSpace *GetAddressSpace(unify::UnifyContext *Ctx,
                                              clang::LangAS AS);
  unify::AddressSpace *GetAddressSpace();
  unify::Type *Visit(clang::QualType Ty);
  unify::Type *VisitArrayType(const clang::ArrayType *);
  unify::Type *VisitFunctionType(const clang::FunctionType *);
  unify::Type *VisitMemberPointerType(const clang::MemberPointerType *);
  unify::Type *VisitPointerType(const clang::PointerType *);
  unify::Type *VisitReferenceType(const clang::ReferenceType *);
  unify::Type *VisitRecordType(const clang::RecordType *);
  unify::Type *VisitVectorType(const clang::VectorType *);

  unify::Type *VisitType(const clang::Type *Ty);

  TypeBuilder(unify::UnifyContext *Ctx, clang::ASTContext &ASTCtx,
              clang::QualType Ty)
      : Ctx{Ctx}, ASTCtx{ASTCtx}, Ty{Ty} {}

public:
  /// @brief Removes unneeded information from a clang type (desugar)
  /// @param ASTCtx The clang AST Context
  /// @param Ty The type to normalize
  /// @return a normalized type
  static const clang::Type *normalizeType(clang::ASTContext &ASTCtx,
                                          clang::QualType Ty);
  /// @brief Returns the "AddressSpace" type matching AS
  /// @param Ctx The Unifier context
  /// @param AS The clang language address space
  /// @return  The AddressSpace matching AS
  static AddressSpace *get(unify::UnifyContext *Ctx, clang::LangAS AS) {
    return GetAddressSpace(Ctx, AS);
  }
  /// @brief Returns a unifier type matching the clang one.
  /// Any place that requires an address space, the builder allocate a new slot
  /// if the clang address space is LangAS::Default.
  /// @param Ctx The Unifier context
  /// @param AS The clang language address space
  /// @return  The AddressSpace matching AS
  static Type *get(unify::UnifyContext *Ctx, clang::ASTContext &ASTCtx,
                   clang::QualType Ty) {
    return TypeBuilder{Ctx, ASTCtx, Ty}.Visit(Ty);
  }
  static GLorPRValue getPRValue(unify::UnifyContext *Ctx,
                                clang::ASTContext &ASTCtx, clang::QualType Ty) {
    return get(Ctx, ASTCtx, Ty);
  }
  template <clang::LangAS AS>
  static GLorPRValue getGLValue(unify::UnifyContext *Ctx,
                                clang::ASTContext &ASTCtx, clang::QualType Ty) {
    return {Ctx->getAddressSpaceType(AS), get(Ctx, ASTCtx, Ty)};
  }
  static GLorPRValue getGLValue(unify::UnifyContext *Ctx,
                                clang::ASTContext &ASTCtx, clang::LangAS AS,
                                clang::QualType Ty) {
    return {Ctx->getAddressSpaceType(AS), get(Ctx, ASTCtx, Ty)};
  }
};

class VarScope;

/// @brief Basic class to scope variable declaration (GV and function nest)
class VarScope {
public:
  using DeclToTypeMap = llvm::DenseMap<const clang::Decl *, GLorPRValue>;

private:
  VarScope *Parent = nullptr;
  VarScope *Root = nullptr;

  DeclToTypeMap DeclToType;

public:
  VarScope() = default;
  VarScope(VarScope &Parent)
      : Parent{&Parent}, Root{Parent.Root ? Parent.Root : &Parent} {}
  // Copy Ctor
  VarScope(const VarScope &Parent) = default;

  GLorPRValue pushDecl(const clang::Decl *D, GLorPRValue T) {
    return DeclToType[D] = T;
  }
  GLorPRValue pushDeclToRoot(const clang::Decl *D, GLorPRValue T) {
    return (Root ? Root : this)->pushDecl(D, T);
  }

  llvm::Optional<GLorPRValue> findType(const clang::Decl *D) const {
    if (auto it = DeclToType.find(D); it != DeclToType.end())
      return it->second;
    if (Parent)
      return Parent->findType(D);
    return {};
  }

  llvm::Optional<GLorPRValue> findType(const clang::DeclRefExpr *S) const {
    return findType(S->getDecl());
  }
};

class FunctionLookup;

/// @brief Statement and Declaration visitor.
/// For each variable and function, the class accumulates the constraint
/// that expression may bring into the unifier.
class ConstraintGather
    : public clang::StmtVisitor<ConstraintGather, llvm::Optional<GLorPRValue>>,
      public clang::DeclVisitor<ConstraintGather, llvm::Optional<GLorPRValue>>,
      public VarScope {
  using BaseStmt =
      clang::StmtVisitor<ConstraintGather, llvm::Optional<GLorPRValue>>;
  using BaseDecl =
      clang::DeclVisitor<ConstraintGather, llvm::Optional<GLorPRValue>>;

  /// @brief RAII to manage the address space to be used when creating GLValue
  /// types. This allows to track the context in which an expression is
  /// operating in.
  struct InitContextRAII {
    ConstraintGather &Ctx;
    clang::LangAS PrevDeclContextASP;
    InitContextRAII(ConstraintGather &Ctx, clang::LangAS NewASP)
        : Ctx{Ctx}, PrevDeclContextASP{Ctx.DeclContextASP} {
      Ctx.DeclContextASP = NewASP;
    }
    ~InitContextRAII() { Ctx.DeclContextASP = PrevDeclContextASP; }
  };

  unify::UnifyContext *Ctx;
  clang::ASTContext &ASTCtx;
  FunctionLookup &FuncLookup;
  Unifier LocalUnifier;

  Type *ThisType = nullptr;
  Type *RetType = nullptr;
  Type *FnType = nullptr;

  Type *ThisTypeExprType = nullptr;
  llvm::DenseMap<clang::Stmt *, GLorPRValue> StmtToType;

  llvm::SmallVector<const clang::Decl *, 32> DeclOrder;

  clang::LangAS DeclContextASP = clang::LangAS::sycl_private;

  unify::UnifyContext *getUnifyContext() const { return Ctx; }
  clang::ASTContext &getASTContext() const { return ASTCtx; }

  /// When processing a lambda expression, captured variables are actually the
  /// VarDecl and not the FieldDecl of the type closure.
  /// This function creates the required mapping before starting to process the
  /// function body so that the variable can be looked up.
  void FillImplicitVariable(clang::FunctionDecl *FD);

  Type *ProcessFunctionImpl(clang::FunctionDecl *FD);

  /// Does the same as ProcessFunction, but with a default scope.
  /// This is used when recursively processing a function,
  /// which should only happen when initially processing GVs.
  const ConstraintGather *RecursiveProcessFunction(clang::FunctionDecl *FD);

  Type *ProcessBuiltin(clang::FunctionDecl *FD);

public:
  ConstraintGather(unify::UnifyContext &Ctx, clang::ASTContext &ASTCtx,
                   FunctionLookup &FuncLookup)
      : VarScope{}, Ctx{&Ctx}, ASTCtx{ASTCtx}, FuncLookup{FuncLookup},
        LocalUnifier{Ctx.getUnifier()->buildNewScope()} {}
  ConstraintGather(ConstraintGather &Parent)
      : VarScope{Parent}, Ctx{Parent.Ctx}, ASTCtx{Parent.ASTCtx},
        FuncLookup{Parent.FuncLookup},
        LocalUnifier{Parent.LocalUnifier.buildNewScope()} {}
  // Copy ctor
  ConstraintGather(const ConstraintGather &Parent) = default;

  Type *ProcessFunction(clang::FunctionDecl *FD);

  bool shouldVisitImplicitCode() const { return true; }

  static clang::FunctionDecl *normalize(clang::FunctionDecl *);
  static const clang::FunctionDecl *normalize(const clang::FunctionDecl *);
  static clang::Decl *normalize(clang::Decl *);
  static const clang::Decl *normalize(const clang::Decl *);

  GLorPRValue ProcessCall(FunctionType *CalleeType,
                          clang::CallExpr::arg_range Args);
  /// When calling a function, we need to retrieve its type and allocate fresh
  /// slots for any pending address space (let-polymorphism). We do this because
  /// we assume there is a duplication process that will run as well. If we
  /// don't do this, the unifier will be forced to yield a unique solution for
  /// all functions.
  FunctionType *BindFunctionCall(clang::Decl *FD);

  llvm::Optional<GLorPRValue> VisitVarDecl(clang::VarDecl *D,
                                           bool AssumeGV = false);
  llvm::Optional<GLorPRValue> VisitEnumConstantDecl(clang::EnumConstantDecl *D);

  llvm::Optional<GLorPRValue> VisitCompoundStmt(clang::CompoundStmt *S);
  llvm::Optional<GLorPRValue> VisitDeclStmt(clang::DeclStmt *S);

  llvm::Optional<GLorPRValue> VisitReturnStmt(clang::ReturnStmt *S);
  llvm::Optional<GLorPRValue> VisitStringLiteral(clang::StringLiteral *S);
  llvm::Optional<GLorPRValue> VisitIntegerLiteral(clang::IntegerLiteral *S);
  llvm::Optional<GLorPRValue> VisitFloatingLiteral(clang::FloatingLiteral *S);
  llvm::Optional<GLorPRValue>
  VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *S);

  llvm::Optional<GLorPRValue>
  VisitConditionalOperator(clang::ConditionalOperator *S);
  llvm::Optional<GLorPRValue> VisitBinaryOperator(clang::BinaryOperator *S);

  llvm::Optional<GLorPRValue>
  VisitorSYCLUniqueStableIdExpr(clang::SYCLUniqueStableIdExpr *S);
  llvm::Optional<GLorPRValue>
  VisitCompoundAssignOperator(clang::CompoundAssignOperator *S);
  llvm::Optional<GLorPRValue> VisitCXXConstructExpr(clang::CXXConstructExpr *S);
  llvm::Optional<GLorPRValue>
  VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *S);
  llvm::Optional<GLorPRValue>
  VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *S);
  llvm::Optional<GLorPRValue> VisitCXXThisExpr(clang::CXXThisExpr *S);
  llvm::Optional<GLorPRValue> VisitCallExpr(clang::CallExpr *S);
  llvm::Optional<GLorPRValue>
  VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *S);
  llvm::Optional<GLorPRValue>
  VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *S);
  llvm::Optional<GLorPRValue> VisitCastExpr(clang::CastExpr *S);
  llvm::Optional<GLorPRValue> VisitDeclRefExpr(clang::DeclRefExpr *S);
  llvm::Optional<GLorPRValue>
  VisitExtVectorElementExpr(clang::ExtVectorElementExpr *S);
  llvm::Optional<GLorPRValue> VisitConstantExpr(clang::ConstantExpr *S);
  llvm::Optional<GLorPRValue> VisitExprWithCleanups(clang::ExprWithCleanups *S);
  llvm::Optional<GLorPRValue> VisitInitListExpr(clang::InitListExpr *S);
  llvm::Optional<GLorPRValue> VisitLambdaExpr(clang::LambdaExpr *S);
  llvm::Optional<GLorPRValue>
  VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *S);
  llvm::Optional<GLorPRValue> VisitMemberExpr(clang::MemberExpr *S);
  llvm::Optional<GLorPRValue> VisitParenExpr(clang::ParenExpr *S);
  llvm::Optional<GLorPRValue>
  VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *S);
  llvm::Optional<GLorPRValue> VisitUnaryOperator(clang::UnaryOperator *S);
  llvm::Optional<GLorPRValue>
  VisitArraySubscriptExpr(clang::ArraySubscriptExpr *S);

  llvm::Optional<GLorPRValue> VisitIfStmt(clang::IfStmt *If);
  llvm::Optional<GLorPRValue> VisitStmt(clang::Stmt *S);
};

class FunctionLookup {
  llvm::DenseMap<const clang::Decl *, ConstraintGather> FuncInfo;

public:
  bool hasFunction(const clang::Decl *D) const {
    assert(D == ConstraintGather::normalize(D) && "not normalized");
    return FuncInfo.count(D);
  }

  template <typename... Ts>
  ConstraintGather *allocateFunctionInfo(const clang::FunctionDecl *D,
                                         Ts &&...Args) {
    assert(D == ConstraintGather::normalize(D) && "not normalized");
    return &FuncInfo.try_emplace(D, std::forward<Ts>(Args)...)
                .first->getSecond();
  }

  const ConstraintGather *lookupFunction(const clang::Decl *D) const {
    assert(D == ConstraintGather::normalize(D) && "not normalized");
    auto it = FuncInfo.find(D);
    if (it == FuncInfo.end())
      return nullptr;
    return &it->second;
  }
};

} // namespace unify

#endif // LLVM_CLANG_INFERASPCHECKER_CONSTRAINTGATHER_H
