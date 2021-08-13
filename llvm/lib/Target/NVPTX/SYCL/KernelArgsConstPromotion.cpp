//===- KernelArgsConstPromotion.cpp - Kernel Args Constant Promotion  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass operates on SYCL kernels being compiled to CUDA. It rewrites the
// signature of the kernel, removing all the arguments, instead the arguments
// are used to construct a struct, where each kernel argument is a struct
// member. The pass then constructs a global variable of the struct type (in
// constant address space) and replaces the use of all the arguments with the
// corresponding members. The SYCL runtime is expected to fill in the struct
// with the data taken from appropriate kernel arguments.
//
//===----------------------------------------------------------------------===//

#include "KernelArgsConstPromotion.h"
#include "../MCTargetDesc/NVPTXBaseInfo.h"
#include "Utils.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define DEBUG_TYPE "kernel-args-const-promotion" // kacp

namespace llvm {
void initializeKernelArgsConstPromotionPass(PassRegistry &);
} // namespace llvm

namespace {

class KernelArgsConstPromotion : public ModulePass {
public:
  KernelArgsConstPromotion() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
  virtual llvm::StringRef getPassName() const override {
    return "kernel-args-const-promotion";
  }

public:
  static char ID;

private:
  // Remove kernel arguments and store them in a struct (global variable in
  // constant address space).
  bool runOnKernel(Function *Kernel, MDNode *Node);
};
} // end anonymous namespace

char KernelArgsConstPromotion::ID = 0;

INITIALIZE_PASS(KernelArgsConstPromotion, "kernel-args-const-promotion",
                "SYCL Kernel Args Const Promotion", false, false)

ModulePass *llvm::createKernelArgsConstPromotionPass() {
  return new KernelArgsConstPromotion();
}

bool KernelArgsConstPromotion::runOnKernel(Function *Kernel, MDNode *Node) {
  auto *M = Kernel->getParent();
  std::string KernelName(Kernel->getName().str());
  auto &Context = M->getContext();
  auto InsertionPt = Kernel->getEntryBlock().getFirstInsertionPt();
  llvm::SmallVector<llvm::Type *, 8> ArgTys;
  for (auto const &Arg : Kernel->args()) {
    ArgTys.push_back(Arg.getType());
  }

  // Create a struct type corresponding to the aggregate of all kernel
  // arguments.
  auto *StructTy = StructType::create(Context, KernelName + "_kacp_struct_ty");
  // Make sure that the struct is packed.
  StructTy->setBody(ArgTys, true);
  // The struct has to be in constant AS.
  auto *SharedMemGlobal = new GlobalVariable(
      *M, StructTy, true, GlobalValue::LinkOnceODRLinkage,
      Constant::getNullValue(StructTy), KernelName + "_kacp_struct_data",
      nullptr, GlobalValue::NotThreadLocal, ADDRESS_SPACE_CONST);

  auto *Int32Ty = Type::getInt32Ty(Context);
  auto *Zero = ConstantInt::get(Int32Ty, 0);

  // Walk down the list of arguments and for each generate a GEP from a struct
  // and a load of the pointer. Finally replace all the uses of the argument
  // with the struct member load.
  for (unsigned i = 0; i < ArgTys.size(); ++i) {
    auto *Arg = Kernel->arg_begin() + i;
    std::string GepName("Arg_GEP_");
    GepName.append(std::to_string(i));
    auto *Gep = GetElementPtrInst::Create(
        StructTy, SharedMemGlobal,
        SmallVector<Value *>{Zero, ConstantInt::get(Int32Ty, i)},
        GepName.c_str());
    std::string LoadName("Arg_Load_");
    LoadName.append(std::to_string(i));
    auto *Load = new LoadInst(Arg->getType(), Gep, LoadName, &*InsertionPt);
    Gep->insertBefore(Load);
    Arg->replaceAllUsesWith(Load);
  }

  // Now get rid of the old kernel, splice the body of it into a new kernel,
  // that has no arguments.
  std::string NewFuncName(Kernel->getName().str() + "_kacp");
  FunctionType *NewFuncTy =
      FunctionType::get(Kernel->getReturnType(), {}, Kernel->isVarArg());
  SmallVector<AttributeSet, 4> ArgumentAttributes;
  Function *NewKernel = NVPTX::Utils::splice(Kernel, ArgumentAttributes,
                                             NewFuncTy, NewFuncName.c_str());
  Kernel->removeFromParent();
  // Update the metadata.
  Node->replaceOperandWith(0, llvm::ConstantAsMetadata::get(NewKernel));

  return true;
}

bool KernelArgsConstPromotion::runOnModule(Module &M) {
  // Invariant: This pass is only intended to operate on SYCL kernels being
  // compiled to the `nvptx{,64}-nvidia-cuda` triple.
  if (skipModule(M))
    return false;

  auto NodeKernelPairs = NVPTX::Utils::getNVVKernels(M);
  if (!NodeKernelPairs)
    return false;

  bool Changed = false;
  for (auto &NodeKernelPair : NodeKernelPairs.getValue()) {
    // Only promote kernels with "kernel-const-mem" attribute.
    if (!std::get<1>(NodeKernelPair)->hasFnAttribute("kernel-const-mem")) {
      continue;
    }
    Changed |=
        runOnKernel(std::get<1>(NodeKernelPair), std::get<0>(NodeKernelPair));
  }
  return Changed;
}
