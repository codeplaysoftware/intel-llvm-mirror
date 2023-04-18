//=== TypePrinter.cpp - Print Unifier type                 ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InferASPChecker/TypePrinter.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace unify;

void TypePrinter::visit(GLorPRValue Value) {
  visit(Value.Ty);
  if (Value.ASP) {
    OS << " ";
    visit(Value.ASP);
  }
}

void TypePrinter::visitAddressSpace(AddressSpace *AS) {
  if (AS->isSolved()) {
    if (clang::isTargetAddressSpace(AS->getAddressSpace())) {
      OS << "__addrspace(" << clang::toTargetAddressSpace(AS->getAddressSpace())
         << ")";
      return;
    }
    switch (AS->getAddressSpace()) {
    case clang::LangAS::Default:
      llvm_unreachable("Default but solved ?");
      break;
    case clang::LangAS::opencl_global:
    case clang::LangAS::sycl_global:
      OS << "__global";
      break;
    case clang::LangAS::opencl_local:
    case clang::LangAS::sycl_local:
      OS << "__local";
      break;
    case clang::LangAS::opencl_private:
    case clang::LangAS::sycl_private:
      OS << "__private";
      break;
    case clang::LangAS::opencl_global_device:
    case clang::LangAS::sycl_global_device:
      OS << "__global_device";
      break;
    case clang::LangAS::opencl_global_host:
    case clang::LangAS::sycl_global_host:
      OS << "__global_host";
      break;
    case clang::LangAS::opencl_constant:
      OS << "__constant";
      break;
    default:
      llvm_unreachable("unhandled");
      break;
    }
  } else {
    // Use address as slot (lower bits, use the fact we rely on a bump ptr
    // allocator)
    OS << llvm::format_hex(0xFFFFFF & reinterpret_cast<std::intptr_t>(AS), 6);
  }
}

void TypePrinter::visitFundamentalType(FundamentalType *Ty) {
  OS << clang::QualType{Ty->getSourceType(), 0}.getAsString();
}

void TypePrinter::visitAddressType(AddressType *Ty) {
  llvm::TypeSwitch<const clang::Type *>(Ty->getSourceType())
      .Case<clang::PointerType>([&](auto *) {
        Base::visit(Ty->getElementType());
        OS << " ";
        Base::visit(Ty->getAddressSpace());
        OS << "*";
      })
      .Case<clang::ArrayType>([&](auto *) {
        Base::visit(Ty->getAddressSpace());
        OS << " ";
        Base::visit(Ty->getElementType());
        OS << "[]";
      })
      .Case<clang::MemberPointerType>([&](const clang::MemberPointerType *CT) {
        Base::visit(Ty->getAddressSpace());
        OS << " ";
        const clang::CXXRecordDecl *RD = CT->getMostRecentCXXRecordDecl();
        if (RD->getDeclName().isIdentifier())
          OS << RD->getName();
        OS << "::";
        Base::visit(Ty->getElementType());
      })
      .Case<clang::ReferenceType>([&](const clang::ReferenceType *RefType) {
        Base::visit(Ty->getElementType());
        OS << " ";
        Base::visit(Ty->getAddressSpace());
        OS << "&";
        if (RefType->isRValueReferenceType())
          OS << "&";
      })
      .Default([](auto *T) {
        T->dump();
        llvm_unreachable("Unknown type");
      });
}

void TypePrinter::visitArrayOrVectorType(ArrayOrVectorType *Ty) {
  if (Ty->getSourceType()->isExtVectorType()) {
    OS << "<";
    Base::visit(Ty->getElementType());
    OS << ">";
  } else {
    Base::visit(Ty->getElementType());
    OS << "[]";
  }
}

void TypePrinter::visitFunctionType(FunctionType *Ty) {
  if (Ty->getThisType()) {
    Base::visit(Ty->getThisType());
    OS << " -> ";
  }
  OS << "(";
  llvm::interleave(
      Ty->getArguments(), OS, [&](Type *ArgT) { Base::visit(ArgT); }, ", ");
  OS << ") -> ";
  Base::visit(Ty->getReturnType());
}

void TypePrinter::visitRecordType(RecordType *Ty) {
  OS << "{";
  llvm::interleave(
      Ty->getBases(), OS, [&](auto &ArgT) { Base::visit(ArgT.second); }, ", ");
  if (Ty->getBases().size() && Ty->getFields().size())
    OS << ", ";
  llvm::interleave(
      Ty->getFields(), OS, [&](auto &ArgT) { Base::visit(ArgT.second); }, ", ");
  OS << "}";
}
