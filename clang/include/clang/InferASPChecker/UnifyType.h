//=== UnifyType.h -   Type system for the unifier               -*- C++ -*-===//
//
// Implement the minimal type system required to have the unifier to run on.
//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INFERASPCHECKER_UNIFYTYPE_H
#define LLVM_INFERASPCHECKER_UNIFYTYPE_H

#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
class FieldDecl;
class CXXBaseSpecifier;
} // namespace clang
namespace llvm {
class raw_ostream;
}

namespace unify {

class UnifyContext;

/// Type abstraction, this represent the type without a top level qualifier.
class Type : public llvm::FoldingSetNode {
public:
  /// Type abstraction for the unifier
  enum class TypeClass : uint32_t {
    AddressSpace, // < Represent an address space, solved or not
    // Clang Type below
    Fundamental, // < Fundamental
    Address,     // < Address kind: pair of address space and pointee type
    ArrayVector, // < Array or vector kind: pointee type, asp carried by ctx
    Function,    // < return + list of type
    Record       // < List of bases and list of fields
  };

protected:
  TypeClass TC;

  Type(TypeClass TC) : TC{TC} {}

  virtual bool isEqualImpl(const Type &Other) const {
    assert(TC == Other.TC);
    return true;
  }

public:
  virtual bool isSolved() const { return true; }
  bool isTypeVariable() const {
    return TC == Type::TypeClass::AddressSpace && !isSolved();
  }
  TypeClass getClass() const { return TC; }

  void Profile(llvm::FoldingSetNodeID &ID) const;

  bool operator==(const Type &Other) const {
    return this == &Other || (TC == Other.TC && isEqualImpl(Other));
  }
  bool operator!=(const Type &Other) const { return !operator==(Other); }

  void print() const;
  void print(llvm::raw_ostream &) const;
};

/// @brief  Representation of an address space, solved or not
///         We use Default as the unsolved address space and use the address of
///         the object to differentiate between 2 unresolved address space slots.
class AddressSpace : public Type {
  clang::LangAS ASP;

protected:
  bool isEqualImpl(const Type &Other) const override {
    assert(this != &Other);
    const AddressSpace &OtherASP = llvm::cast<AddressSpace>(Other);
    // Default asp and different address means different not equal
    if (ASP == clang::LangAS::Default && ASP == OtherASP.ASP)
      return false;
    // This is not default, simply checks against Other
    return ASP == OtherASP.ASP;
  }

public:
  explicit AddressSpace(clang::LangAS ASP)
      : Type{TypeClass::AddressSpace}, ASP{ASP} {}
  explicit AddressSpace()
      : Type{TypeClass::AddressSpace}, ASP{clang::LangAS::Default} {}

  static bool classof(const Type *E) {
    return E->getClass() == TypeClass::AddressSpace;
  }

  bool isSolved() const override { return ASP != clang::LangAS::Default; }

  clang::LangAS getAddressSpace() const { return ASP; }
};

/// @brief Intermediate class to model clang types
/// Concrete implementation will hold a break down
class ClangType : public Type {
protected:
  const clang::Type *SourceType;

  ClangType(TypeClass TC, const clang::Type *SourceType)
      : Type{TC}, SourceType{SourceType} {}

  bool isEqualImpl(const Type &Other) const override {
    assert(this != &Other);
    const ClangType &OtherCT = llvm::cast<ClangType>(Other);
    return SourceType == OtherCT.SourceType;
  }

public:
  const clang::Type *getSourceType() const { return SourceType; }

  static bool classof(const Type *E) {
    return static_cast<uint32_t>(E->getClass()) >=
           static_cast<uint32_t>(TypeClass::Fundamental);
  }
};

/// @brief Represent any clang type not holding an address space.
class FundamentalType : public ClangType {
public:
  FundamentalType(const clang::Type *SourceType)
      : ClangType{TypeClass::Fundamental, SourceType} {}

  static bool classof(const Type *E) {
    return E->getClass() == TypeClass::Fundamental;
  }
};

/// Pointer and reference types.
/// Represent the address space and pointee / element type.
class AddressType : public ClangType {
  AddressSpace *ASP;
  Type *UnderlyingType;

protected:
  bool isEqualImpl(const Type &Other) const override {
    assert(this != &Other);
    const AddressType &OtherCT = llvm::cast<AddressType>(Other);
    return ClangType::isEqualImpl(Other) && *ASP == *OtherCT.ASP &&
           *UnderlyingType == *OtherCT.UnderlyingType;
  }

public:
  AddressType(const clang::Type *SourceType, AddressSpace *AddressSpace,
              Type *UnderlyingType)
      : ClangType{TypeClass::Address, SourceType}, ASP{AddressSpace},
        UnderlyingType{UnderlyingType} {
    assert(isValidSourceType(SourceType));
    assert(ASP);
    assert(UnderlyingType);
  }

  static bool isValidSourceType(const clang::Type *SourceType) {
    return !SourceType || llvm::isa<clang::PointerType>(SourceType) ||
           llvm::isa<clang::MemberPointerType>(SourceType) ||
           llvm::isa<clang::ReferenceType>(SourceType);
  }

  static bool classof(const Type *E) {
    return E->getClass() == TypeClass::Address;
  }

  bool isSolved() const override {
    return ASP->isSolved() && UnderlyingType->isSolved();
  }

  bool isReference() const { return getSourceType()->isReferenceType(); }

  AddressSpace *getAddressSpace() const { return ASP; }
  Type *getElementType() const { return UnderlyingType; }

  template <typename T = Type> std::tuple<AddressSpace *, T *> get() const {
    return {ASP, llvm::cast<T>(UnderlyingType)};
  }
};

/// Array or vector type.
class ArrayOrVectorType : public ClangType {
  Type *UnderlyingType;

protected:
  bool isEqualImpl(const Type &Other) const override {
    assert(this != &Other);
    const ArrayOrVectorType &OtherCT = llvm::cast<ArrayOrVectorType>(Other);
    return ClangType::isEqualImpl(Other) &&
           *UnderlyingType == *OtherCT.UnderlyingType;
  }

public:
  ArrayOrVectorType(const clang::Type *SourceType, Type *UnderlyingType)
      : ClangType{TypeClass::ArrayVector, SourceType}, UnderlyingType{
                                                           UnderlyingType} {}

  static bool isValidSourceType(const clang::Type *SourceType) {
    return !SourceType || llvm::isa<clang::ArrayType>(SourceType) ||
           llvm::isa<clang::ExtVectorType>(SourceType);
  }

  static bool classof(const Type *E) {
    return E->getClass() == TypeClass::ArrayVector;
  }

  bool isSolved() const override { return UnderlyingType->isSolved(); }

  Type *getElementType() const { return UnderlyingType; }
};

/// Function type.
class FunctionType : public ClangType {
  Type *ReturnType;
  Type *ThisType;

  llvm::SmallVector<Type *, 8> Arguments;

protected:
  bool isEqualImpl(const Type &Other) const override {
    assert(this != &Other);
    const FunctionType &OtherCT = llvm::cast<FunctionType>(Other);
    return ClangType::isEqualImpl(Other) &&
           *ReturnType == *OtherCT.ReturnType &&
           llvm::all_of_zip(Arguments, OtherCT.Arguments,
                            [=](const Type *ArgTy, const Type *OtherArgTy) {
                              return *ArgTy == *OtherArgTy;
                            });
  }

public:
  FunctionType(const clang::Type *SourceType, Type *ThisType, Type *ReturnType,
               llvm::ArrayRef<Type *> Arguments)
      : ClangType{TypeClass::Function, SourceType},
        ReturnType{ReturnType}, ThisType{ThisType}, Arguments{Arguments} {}

  static bool classof(const Type *E) {
    return E->getClass() == TypeClass::Function;
  }

  bool isSolved() const override {
    return ReturnType->isSolved() ||
           llvm::any_of(Arguments, [](Type *A) { return A->isSolved(); });
  }

  Type *getReturnType() const { return ReturnType; }
  Type *getThisType() const { return ThisType; }

  llvm::ArrayRef<Type *> getArguments() const { return Arguments; }
};

/// Records: struct, union type
class RecordType : public ClangType {
public:
  using Field = std::pair<const clang::FieldDecl *, Type *>;
  using FieldList = llvm::SmallVector<Field, 8>;
  using Base = std::pair<const clang::CXXBaseSpecifier *, RecordType *>;
  using BaseList = llvm::SmallVector<Base, 8>;

protected:
  FieldList Fields;
  BaseList Bases;

  bool isEqualImpl(const Type &Other) const override {
    assert(this != &Other);
    const RecordType &OtherCT = llvm::cast<RecordType>(Other);
    return ClangType::isEqualImpl(Other) &&
           llvm::all_of_zip(Fields, OtherCT.Fields,
                            [=](auto &FTy, auto &OtherFTy) {
                              return FTy.first == OtherFTy.first &&
                                     FTy.second == OtherFTy.second;
                            });
  }

public:
  RecordType(const clang::Type *SourceType, llvm::ArrayRef<Field> Fields,
             llvm::ArrayRef<Base> Bases)
      : ClangType{TypeClass::Record, SourceType}, Fields{Fields}, Bases{Bases} {
  }

  static bool classof(const Type *E) {
    return E->getClass() == TypeClass::Record;
  }

  bool isSolved() const override {
    return llvm::any_of(Fields, [](auto &F) { return F.second->isSolved(); });
  }

  FieldList::const_iterator field_begin() const { return Fields.begin(); }
  FieldList::const_iterator field_end() const { return Fields.end(); }

  FieldList::const_iterator getField(const clang::FieldDecl *FD) const {
    return llvm::find_if(Fields, [&](const Field &F) { return F.first == FD; });
  }

  llvm::ArrayRef<Field> getFields() const { return Fields; }
  llvm::ArrayRef<Base> getBases() const { return Bases; }

  Type *findFieldType(const clang::FieldDecl *D) const {
    return llvm::find_if(Fields, [&](auto &F) { return F.first == D; })->second;
  }
};

/// Simple data structure to hold a LValue / RValue alternative.
/// GLValue holds a type and an address space while a PRValue only hold a type.
struct GLorPRValue {
  AddressSpace *ASP = nullptr;
  Type *Ty = nullptr;

  GLorPRValue() = default;
  GLorPRValue(Type *Ty) : ASP{nullptr}, Ty{Ty} { assert(Ty); }
  GLorPRValue(AddressSpace *ASP, Type *Ty) : ASP{ASP}, Ty{Ty} {
    assert(ASP);
    assert(Ty);
  }

  void print() const;
  void print(llvm::raw_ostream &) const;
};

/// @brief Simple CRTP class to implement a Type dispatcher
/// @tparam ImplClass The visitor class
/// @tparam RetType return type of visiting functions
template <typename ImplClass, typename RetType = void> struct UnifyTypeVisitor {
#define DISPATCH(CLASS, EXPR)                                                  \
  return static_cast<ImplClass *>(this)->visit##CLASS(                         \
      static_cast<CLASS *>(EXPR))

  void visit(GLorPRValue Value) {
    if (Value.ASP)
      DISPATCH(AddressSpace, Value.ASP);
    visit(Value.Ty);
  }

  RetType visit(Type *Ty) {
    switch (Ty->getClass()) {
    case Type::TypeClass::AddressSpace:
      DISPATCH(AddressSpace, Ty);
    case Type::TypeClass::Fundamental:
      DISPATCH(FundamentalType, Ty);
    case Type::TypeClass::Address:
      DISPATCH(AddressType, Ty);
    case Type::TypeClass::ArrayVector:
      DISPATCH(ArrayOrVectorType, Ty);
    case Type::TypeClass::Function:
      DISPATCH(FunctionType, Ty);
    case Type::TypeClass::Record:
      DISPATCH(RecordType, Ty);
    }
    DISPATCH(Type, Ty);
  }

  RetType visitType(Type *) { return RetType(); }

  RetType visitClangType(ClangType *Ty) { DISPATCH(Type, Ty); }

  RetType visitAddressSpace(AddressSpace *Ty) { DISPATCH(Type, Ty); }

  RetType visitFundamentalType(FundamentalType *Ty) { DISPATCH(ClangType, Ty); }

  RetType visitAddressType(AddressType *Ty) { DISPATCH(ClangType, Ty); }
  RetType visitArrayOrVectorType(ArrayOrVectorType *Ty) {
    DISPATCH(ClangType, Ty);
  }
  RetType visitFunctionType(FunctionType *Ty) { DISPATCH(ClangType, Ty); }

  RetType visitRecordType(RecordType *Ty) { DISPATCH(ClangType, Ty); }

#undef DISPATCH
};

/// @brief Simple CRTP class to implement a Type dispatcher for pair of Types
/// @tparam ImplClass The visitor class
/// @tparam RetType return type of visiting functions
template <typename ImplClass, typename RetType = void>
struct PairUnifyTypeVisitor {
#define DISPATCH(CLASS, EXPR1, EXPR2)                                          \
  return static_cast<ImplClass *>(this)->visit##CLASS(                         \
      static_cast<CLASS *>(EXPR1), static_cast<CLASS *>(EXPR2))

  void visit(GLorPRValue LHS, GLorPRValue RHS) {
    if (LHS.ASP && RHS.ASP)
      visit(LHS.ASP, RHS.ASP);
    visit(LHS.Ty, RHS.Ty);
  }

  RetType visit(Type *LHS, Type *RHS) {
    if (LHS->getClass() != RHS->getClass()) {
      DISPATCH(Type, LHS, RHS);
    }

    switch (LHS->getClass()) {
    case Type::TypeClass::AddressSpace:
      DISPATCH(AddressSpace, LHS, RHS);
    case Type::TypeClass::Fundamental:
      DISPATCH(FundamentalType, LHS, RHS);
    case Type::TypeClass::Address:
      DISPATCH(AddressType, LHS, RHS);
    case Type::TypeClass::ArrayVector:
      DISPATCH(ArrayOrVectorType, LHS, RHS);
    case Type::TypeClass::Function:
      DISPATCH(FunctionType, LHS, RHS);
    case Type::TypeClass::Record:
      DISPATCH(RecordType, LHS, RHS);
    }
    DISPATCH(Type, LHS, RHS);
  }

  RetType visitType(Type *, Type *) { return RetType(); }

  RetType visitClangType(ClangType *Ty1, ClangType *Ty2) {
    DISPATCH(Type, Ty1, Ty2);
  }

  RetType visitAddressSpace(AddressSpace *Ty1, AddressSpace *Ty2) {
    DISPATCH(Type, Ty1, Ty2);
  }

  RetType visitFundamentalType(FundamentalType *Ty1, FundamentalType *Ty2) {
    DISPATCH(ClangType, Ty1, Ty2);
  }

  RetType visitAddressType(AddressType *Ty1, AddressType *Ty2) {
    DISPATCH(ClangType, Ty1, Ty2);
  }
  RetType visitArrayOrVectorType(ArrayOrVectorType *Ty1,
                                 ArrayOrVectorType *Ty2) {
    DISPATCH(ClangType, Ty1, Ty2);
  }
  RetType visitFunctionType(FunctionType *Ty1, FunctionType *Ty2) {
    DISPATCH(ClangType, Ty1, Ty2);
  }

  RetType visitRecordType(RecordType *Ty1, RecordType *Ty2) {
    DISPATCH(ClangType, Ty1, Ty2);
  }

#undef DISPATCH
};
} // namespace unify

#endif
