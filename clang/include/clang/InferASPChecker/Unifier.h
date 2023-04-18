//=== Unifier.h - Unifier engine                                -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INFERASPCHECKER_UNIFIER_H
#define LLVM_CLANG_INFERASPCHECKER_UNIFIER_H

#include <utility>

#include "UnifyType.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SetVector.h"

namespace clang {
class Expr;
}

namespace unify {

class UnifyContext;

/// @brief Unifier constraint representation.
/// For now it contains 2 types and the context in which each appears.
class Constraint {
public:
  /// @brief Wrapping class to store the context in which type is originating
  /// from.
  class TypeCtx {
    llvm::PointerUnion<const clang::Expr *, const clang::Decl *> Ctx;

  public:
    TypeCtx(std::nullptr_t) : Ctx{nullptr} {}
    TypeCtx(const clang::Expr *E) : Ctx{E} {}
    TypeCtx(const clang::Decl *D) : Ctx{D} {}

    operator bool() const { return !Ctx.isNull(); }

    clang::SourceLocation getLocation() const {
      if (Ctx.isNull())
        return {};
      if (Ctx.is<const clang::Expr *>())
        return Ctx.get<const clang::Expr *>()->getExprLoc();
      assert(Ctx.is<const clang::Decl *>());
      return Ctx.get<const clang::Decl *>()->getLocation();
    }
    void print(llvm::raw_ostream &OS, clang::ASTContext &Context) const {
      if (Ctx.is<const clang::Expr *>()) {
        Ctx.get<const clang::Expr *>()->dump(OS, Context);
      } else {
        Ctx.get<const clang::Decl *>()->dump(OS);
      }
    }
    void dump() const {
      if (Ctx.is<const clang::Expr *>()) {
        Ctx.get<const clang::Expr *>()->dump();
      } else {
        Ctx.get<const clang::Decl *>()->dump();
      }
    }
  };

private:
  Type *LHS;
  Type *RHS;

  TypeCtx LHSCtx;
  TypeCtx RHSCtx;

public:
  Constraint(TypeCtx LHSCtx, Type *LHS_, TypeCtx RHSCtx, Type *RHS_)
      : LHS{LHS_}, RHS{RHS_}, LHSCtx{LHSCtx}, RHSCtx{RHSCtx} {
    // order doesn't matter, normalize by address value
    if (RHS > LHS)
      std::swap(LHS, RHS);
  }

  Type *getLHS() const { return LHS; }
  Type *getRHS() const { return RHS; }

  TypeCtx getLHSCtx() const { return LHSCtx; }
  TypeCtx getRHSCtx() const { return RHSCtx; }

  std::pair<Type *, Type *> asPair() const { return {LHS, RHS}; }

  bool operator==(const Constraint &other) const {
    // LHS and RHS are normalized by address value, no need to test all
    // variants.
    return LHS == other.LHS && RHS == other.RHS;
  }
};

} // namespace unify
namespace llvm {
template <> struct DenseMapInfo<unify::Constraint> {
  using ProxyDenseMapInfo =
      llvm::DenseMapInfo<std::pair<unify::Type *, unify::Type *>>;
  static inline unify::Constraint getEmptyKey() {
    return unify::Constraint(nullptr, nullptr, nullptr, nullptr);
  }
  static inline unify::Constraint getTombstoneKey() {
    return unify::Constraint(nullptr, nullptr, nullptr,
                             reinterpret_cast<unify::Type *>((intptr_t)-1));
  }
  static unsigned getHashValue(const unify::Constraint &Val) {
    return ProxyDenseMapInfo::getHashValue(Val.asPair());
  }
  static bool isEqual(const unify::Constraint &A, const unify::Constraint &B) {
    return A == B;
  }
};
} // namespace llvm

namespace unify {

/// Unifier engine. Collects constraints, solves them, raises error if need be.
/// Note that constraints being pushed must be with types that only differ by
/// the address space they contain. Resolving the differences in type must be
/// done prior to pushing the constraint. For instance, consider this statement:
///    int& p = *in;
/// `p` is a reference to `int` but `*in` will be `int`, so it is the role of the
/// constraint gathering to ensure `p`'s type isn't pushed as reference but as `int`.
class Unifier {
public:
  using ConstraintSet = llvm::SetVector<Constraint>;
  using SubstitutionType = llvm::DenseMap<Type *, Type *>;

private:
  UnifyContext *Ctx;

  Unifier *Parent = nullptr;
  ConstraintSet Constraints;
  SubstitutionType Substitution;

  Unifier(UnifyContext *Ctx) : Ctx{Ctx} {};
  explicit Unifier(UnifyContext *Ctx, Unifier *Parent)
      : Ctx{Ctx}, Parent{Parent}, Constraints{Parent->Constraints} {}

  bool unifyVariableImpl(Constraint::TypeCtx, AddressSpace *,
                         Constraint::TypeCtx, Type *);
  bool unifyImpl(Constraint &);

  Type *getCannonImpl(Type *);
  void checkConstraints(Type *LHS, Type *RHS) const;

public:
  Unifier buildNewScope() { return Unifier{Ctx, this}; }

  clang::DiagnosticsEngine &getDiag() const;
  clang::DiagnosticBuilder Diag(clang::SourceLocation Loc,
                                unsigned DiagID) const;
  clang::DiagnosticBuilder Diag(unsigned DiagID) const;

  UnifyContext *getContext() const { return Ctx; }

  /// @brief After unification, the function returns a map of substitutions,
  /// allowing one to simplify types to either concrete address spaces or use a
  /// common slot for address space relating to each other.
  /// @return A map providing a mapping of all address spaces seen and to which
  /// type they should be substituted with.
  const SubstitutionType &getSubstitutions() const { return Substitution; }

  std::size_t getConstraintCount() const { return Constraints.size(); }

  void addConstraint(Constraint::TypeCtx LHSCtx, Type *LHS,
                     Constraint::TypeCtx RHSCtx, Type *RHS) {
    checkConstraints(LHS, RHS);
    if (LHS != RHS)
      Constraints.insert(Constraint{LHSCtx, LHS, RHSCtx, RHS});
  }

  void addConstraint(Constraint::TypeCtx LHSCtx, GLorPRValue LHS,
                     Constraint::TypeCtx RHSCtx, GLorPRValue RHS) {
    if (LHS.ASP && RHS.ASP)
      addConstraint(LHSCtx, LHS.ASP, RHSCtx, RHS.ASP);
    addConstraint(LHSCtx, LHS.Ty, RHSCtx, RHS.Ty);
  }

  /// Return a substituted and canonized type.
  template <typename T> T *getCannon(T *Ty) {
    return llvm::cast<T>(getCannonImpl(Ty));
  }
  GLorPRValue getCannon(GLorPRValue Ty) {
    if (Ty.ASP)
      Ty.ASP = getCannon(Ty.ASP);
    Ty.Ty = getCannon(Ty.Ty);
    return Ty;
  }

  /// @brief Perform the unification of all the accumulated constraints
  /// @return True if success, false otherwise.
  bool unify();

  void dump() const;

  friend UnifyContext;
};

} // namespace unify

#endif
