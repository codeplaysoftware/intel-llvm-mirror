//=== TypeBuilder.cpp - Build a unifier type from a C++ Type  ---*- C++ -*-===//
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

using namespace unify;

const clang::Type *TypeBuilder::normalizeType(clang::ASTContext &ASTCtx,
                                              clang::QualType Ty) {
  return Ty.getDesugaredType(ASTCtx).getTypePtr();
}

unify::Type *TypeBuilder::Visit(clang::QualType Ty) {
  return Base::Visit(normalizeType(ASTCtx, Ty));
}

AddressSpace *TypeBuilder::GetAddressSpace(unify::UnifyContext *Ctx,
                                           clang::LangAS AS) {
  if (AS == clang::LangAS::Default) {
    return Ctx->getSlot();
  }

  return Ctx->getAddressSpaceType(AS);
}

AddressSpace *TypeBuilder::GetAddressSpace() {
  return GetAddressSpace(Ctx, Ty.getAddressSpace());
}

Type *TypeBuilder::VisitArrayType(const clang::ArrayType *A) {
  return Ctx->getArrayOrVectorType(
      A, TypeBuilder::get(Ctx, ASTCtx, A->getElementType()));
}

Type *TypeBuilder::VisitFunctionType(const clang::FunctionType *FT) {
  llvm_unreachable("A function type should be processed as part of the "
                   "FunctionConstraintGather process");
  return nullptr;
}

Type *TypeBuilder::VisitMemberPointerType(const clang::MemberPointerType *MP) {
  // We don't care about the class type
  return Ctx->getAddressLikeType(
      MP, GetAddressSpace(),
      TypeBuilder::get(Ctx, ASTCtx, MP->getPointeeType()));
}

Type *TypeBuilder::VisitPointerType(const clang::PointerType *P) {
  return Ctx->getAddressLikeType(
      P, get(Ctx, P->getPointeeType().getAddressSpace()),
      TypeBuilder::get(Ctx, ASTCtx, P->getPointeeType()));
}

Type *TypeBuilder::VisitReferenceType(const clang::ReferenceType *R) {
  return Ctx->getAddressLikeType(
      R, get(Ctx, R->getPointeeType().getAddressSpace()),
      TypeBuilder::get(Ctx, ASTCtx, R->getPointeeType()));
}

Type *TypeBuilder::VisitRecordType(const clang::RecordType *R) {
  const clang::CXXRecordDecl *RD =
      llvm::cast<clang::CXXRecordDecl>(R->getDecl());
  RecordType::BaseList Bases;
  // vbase needs some TLC here
  llvm::for_each(RD->bases(), [&](const clang::CXXBaseSpecifier &D) {
    Bases.emplace_back(
        &D, llvm::cast<RecordType>(TypeBuilder::get(Ctx, ASTCtx, D.getType())));
  });
  RecordType::FieldList Fields;
  llvm::for_each(R->getDecl()->fields(), [&](clang::FieldDecl *D) {
    Fields.emplace_back(D, TypeBuilder::get(Ctx, ASTCtx, D->getType()));
  });
  return Ctx->getRecordType(R, Fields, Bases);
}

Type *TypeBuilder::VisitVectorType(const clang::VectorType *V) {
  return Ctx->getArrayOrVectorType(
      V, TypeBuilder::get(Ctx, ASTCtx, V->getElementType()));
}

Type *TypeBuilder::VisitType(const clang::Type *Ty) {
  assert(Ty->isBuiltinType() || Ty->isEnumeralType() || Ty->isBitIntType() ||
         Ty->isOpenCLSpecificType() || Ty->isStdByteType());
  return Ctx->getFundamentalType(Ty);
}
