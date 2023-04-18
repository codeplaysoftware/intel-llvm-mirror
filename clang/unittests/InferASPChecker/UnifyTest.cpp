//== unittests/InferASPChecker/UnifyTest.cpp -------------------------========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/AddressSpaces.h"
#include "clang/InferASPChecker/UnifyContext.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace unify;
using namespace clang;
namespace {

TEST(UnifyTest, ContextAndType) {
  auto AST = tooling::buildASTFromCode("");
  UnifyContext Ctx{AST->getASTContext(), AST->getASTContext().getDiagnostics()};

  {
    auto Node = Ctx.getSlot();
    EXPECT_EQ(Node->getAddressSpace(), clang::LangAS::Default);
    EXPECT_FALSE(Node->isSolved());
  }
  {
    auto Node = Ctx.getAddressSpaceType(clang::LangAS::sycl_global);
    EXPECT_EQ(Node->getAddressSpace(), clang::LangAS::sycl_global);
    EXPECT_TRUE(Node->isSolved());
    // Test caching
    EXPECT_EQ(Node, Ctx.getAddressSpaceType(clang::LangAS::sycl_global));
  }
  {
    auto ASP = Ctx.getSlot();
    auto FTy = Ctx.getFundamentalType(nullptr);
    EXPECT_TRUE(FTy->isSolved());
    auto Node = Ctx.getAddressLikeType(nullptr, ASP, FTy);
    EXPECT_EQ(Node->getAddressSpace(), ASP);
    EXPECT_EQ(Node->getElementType(), FTy);
    EXPECT_FALSE(Node->isSolved());
  }
}

TEST(UnifyTest, SimpleSolve) {
  auto AST = tooling::buildASTFromCode("");
  UnifyContext Ctx{AST->getASTContext(), AST->getASTContext().getDiagnostics()};
  Unifier &Unify = *Ctx.getUnifier();

  auto ASPShouldInferToGlobal = Ctx.getSlot();
  auto ASPShouldInferToLocal = Ctx.getSlot();
  auto ASPGlobal = Ctx.getAddressSpaceType(clang::LangAS::sycl_global);
  auto ASPLocal = Ctx.getAddressSpaceType(clang::LangAS::sycl_local);

  EXPECT_NE(*ASPShouldInferToGlobal, *ASPGlobal);
  EXPECT_NE(*ASPShouldInferToLocal, *ASPLocal);

  // Check trivial constraints aren't added
  Unify.addConstraint(nullptr, ASPShouldInferToGlobal, nullptr,
                      ASPShouldInferToGlobal);
  EXPECT_EQ(Unify.getConstraintCount(), 0u);
  Unify.addConstraint(nullptr, ASPShouldInferToGlobal, nullptr, ASPGlobal);
  EXPECT_EQ(Unify.getConstraintCount(), 1u);
  // Check constraints aren't added twice
  Unify.addConstraint(nullptr, ASPShouldInferToGlobal, nullptr, ASPGlobal);
  EXPECT_EQ(Unify.getConstraintCount(), 1u);
  Unify.addConstraint(nullptr, ASPShouldInferToLocal, nullptr, ASPLocal);
  EXPECT_EQ(Unify.getConstraintCount(), 2u);
  // Solve
  EXPECT_TRUE(Unify.unify());

  // Check it was substituted
  EXPECT_EQ(*Unify.getCannon(ASPShouldInferToGlobal), *ASPGlobal);
  EXPECT_EQ(*Unify.getCannon(ASPShouldInferToLocal), *ASPLocal);
}

TEST(UnifyTest, RecursiveSolve) {
  auto AST = tooling::buildASTFromCode("");
  UnifyContext Ctx{AST->getASTContext(), AST->getASTContext().getDiagnostics()};
  Unifier &Unify = *Ctx.getUnifier();

  auto ASPShouldInferToGlobal = Ctx.getSlot();
  auto ASPShouldInferToLocal = Ctx.getSlot();
  auto ASPGlobal = Ctx.getAddressSpaceType(clang::LangAS::sycl_global);
  auto ASPLocal = Ctx.getAddressSpaceType(clang::LangAS::sycl_local);
  auto FTy = Ctx.getFundamentalType(nullptr);

  auto UnknownAddressShouldInferToGlobal =
      Ctx.getAddressLikeType(nullptr, ASPShouldInferToGlobal, FTy);
  auto UnknownAddressShouldInferToLocal =
      Ctx.getAddressLikeType(nullptr, ASPShouldInferToLocal, FTy);
  auto AddressToGlobal = Ctx.getAddressLikeType(nullptr, ASPGlobal, FTy);
  auto AddressToLocal = Ctx.getAddressLikeType(nullptr, ASPLocal, FTy);

  EXPECT_NE(*UnknownAddressShouldInferToGlobal, *AddressToGlobal);
  EXPECT_NE(*UnknownAddressShouldInferToLocal, *AddressToLocal);

  Unify.addConstraint(nullptr, UnknownAddressShouldInferToGlobal, nullptr,
                      AddressToGlobal);
  Unify.addConstraint(nullptr, UnknownAddressShouldInferToLocal, nullptr,
                      AddressToLocal);
  // Solve
  EXPECT_TRUE(Unify.unify());
  // Check it was substituted
  EXPECT_EQ(*Unify.getCannon(UnknownAddressShouldInferToGlobal),
            *AddressToGlobal);
  EXPECT_EQ(*Unify.getCannon(UnknownAddressShouldInferToLocal),
            *AddressToLocal);
}

TEST(UnifyTest, RecursiveSolve2) {
  auto AST = tooling::buildASTFromCode("");
  UnifyContext Ctx{AST->getASTContext(), AST->getASTContext().getDiagnostics()};
  Unifier &Unify = *Ctx.getUnifier();

  auto ASPShouldInferToGlobal = Ctx.getSlot();
  auto ASPShouldInferToLocal = Ctx.getSlot();
  auto ASPGlobal = Ctx.getAddressSpaceType(clang::LangAS::sycl_global);
  auto ASPLocal = Ctx.getAddressSpaceType(clang::LangAS::sycl_local);
  auto FTy = Ctx.getFundamentalType(nullptr);

  auto UnknownAddressShouldInferToGlobal =
      Ctx.getAddressLikeType(nullptr, ASPShouldInferToGlobal, FTy);
  auto UnknownAddressShouldInferToLocal =
      Ctx.getAddressLikeType(nullptr, ASPShouldInferToLocal, FTy);
  auto AddressToGlobal = Ctx.getAddressLikeType(nullptr, ASPGlobal, FTy);
  auto AddressToLocal = Ctx.getAddressLikeType(nullptr, ASPLocal, FTy);

  EXPECT_NE(*UnknownAddressShouldInferToGlobal, *AddressToGlobal);
  EXPECT_NE(*UnknownAddressShouldInferToLocal, *AddressToLocal);

  // same as before but also include sub asp as constraints as well
  Unify.addConstraint(nullptr, UnknownAddressShouldInferToGlobal, nullptr,
                      AddressToGlobal);
  Unify.addConstraint(nullptr, UnknownAddressShouldInferToLocal, nullptr,
                      AddressToLocal);
  Unify.addConstraint(nullptr, ASPShouldInferToGlobal, nullptr, ASPGlobal);
  Unify.addConstraint(nullptr, ASPShouldInferToLocal, nullptr, ASPLocal);
  // Solve
  EXPECT_TRUE(Unify.unify());
  // Check it was substituted.
  EXPECT_EQ(*Unify.getCannon(UnknownAddressShouldInferToGlobal),
            *AddressToGlobal);
  EXPECT_EQ(*Unify.getCannon(UnknownAddressShouldInferToLocal),
            *AddressToLocal);
}

TEST(UnifyTest, DerivationType) {
  auto AST = tooling::buildASTFromCode("");
  UnifyContext Ctx{AST->getASTContext(), AST->getASTContext().getDiagnostics()};

  auto Slot = Ctx.getSlot();
  auto ASPLocal = Ctx.getAddressSpaceType(clang::LangAS::sycl_local);
  auto FTy = Ctx.getFundamentalType(nullptr);

  auto PolyPtr = Ctx.getAddressLikeType(nullptr, Slot, FTy);
  auto LocalPtr = Ctx.getAddressLikeType(nullptr, ASPLocal, FTy);
  auto DerivedPolyPtr = Ctx.getDerivedType(PolyPtr);
  auto DerivedLocalPtr = Ctx.getDerivedType(LocalPtr);

  EXPECT_NE(PolyPtr, DerivedPolyPtr);
  EXPECT_NE(*PolyPtr, *DerivedPolyPtr);
  EXPECT_EQ(LocalPtr, DerivedLocalPtr);
  EXPECT_EQ(*LocalPtr, *DerivedLocalPtr);
}

} // namespace
