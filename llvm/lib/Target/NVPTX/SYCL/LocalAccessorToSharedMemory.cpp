//===- LocalAccessorToSharedMemory.cpp - Local Accessor Support for CUDA --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass operates on SYCL kernels being compiled to CUDA. It modifies
// kernel entry points which take pointers to shared memory and modifies them
// to take offsets into shared memory (represented by a symbol in the shared
// address space). The SYCL runtime is expected to provide offsets rather than
// pointers to these functions.
//
//===----------------------------------------------------------------------===//

#include "LocalAccessorToSharedMemory.h"
#include "../MCTargetDesc/NVPTXBaseInfo.h"
#include "Utils.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

#define DEBUG_TYPE "localaccessortosharedmemory"

namespace llvm {
void initializeLocalAccessorToSharedMemoryPass(PassRegistry &);
}

namespace {

class LocalAccessorToSharedMemory : public ModulePass {
public:
  static char ID;
  LocalAccessorToSharedMemory() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    // Invariant: This pass is only intended to operate on SYCL kernels being
    // compiled to the `nvptx{,64}-nvidia-cuda` triple.
    // TODO: make sure that non-SYCL kernels are not impacted.
    if (skipModule(M))
      return false;

    // Keep track of whether the module was changed.
    auto Changed = false;

    // Get all the kernels in the module.
    auto NodeKernelPairs = NVPTX::Utils::getNVVKernels(M);
    if (!NodeKernelPairs)
      return false;

    for (auto &NodeKernelPair : NodeKernelPairs.getValue()) {
      auto NewKernel = this->ProcessFunction(M, std::get<1>(NodeKernelPair));
      if (NewKernel) {
        Changed |= true;
        std::get<0>(NodeKernelPair)
            ->replaceOperandWith(0, llvm::ConstantAsMetadata::get(NewKernel));
      }
    }

    return Changed;
  }

  Function *ProcessFunction(Module &M, Function *F) {
    // Check if this function is eligible by having an argument that uses shared
    // memory.
    auto UsesLocalMemory = false;
    for (Function::arg_iterator FA = F->arg_begin(), FE = F->arg_end();
         FA != FE; ++FA) {
      if (FA->getType()->isPointerTy()) {
        UsesLocalMemory =
            FA->getType()->getPointerAddressSpace() == ADDRESS_SPACE_SHARED;
      }
      if (UsesLocalMemory) {
        break;
      }
    }

    // Skip functions which are not eligible.
    if (!UsesLocalMemory)
      return nullptr;

    // Create a global symbol to CUDA shared memory.
    auto SharedMemGlobalName = F->getName().str();
    SharedMemGlobalName.append("_shared_mem");
    auto SharedMemGlobalType =
        ArrayType::get(Type::getInt8Ty(M.getContext()), 0);
    auto SharedMemGlobal = new GlobalVariable(
        /* Module= */ M,
        /* Type= */ &*SharedMemGlobalType,
        /* IsConstant= */ false,
        /* Linkage= */ GlobalValue::ExternalLinkage,
        /* Initializer= */ nullptr,
        /* Name= */ Twine{SharedMemGlobalName},
        /* InsertBefore= */ nullptr,
        /* ThreadLocalMode= */ GlobalValue::NotThreadLocal,
        /* AddressSpace= */ ADDRESS_SPACE_SHARED,
        /* IsExternallyInitialized= */ false);
    SharedMemGlobal->setAlignment(Align(4));

    FunctionType *FTy = F->getFunctionType();
    const AttributeList &FAttrs = F->getAttributes();

    // Store the arguments and attributes for the new function, as well as which
    // arguments were replaced.
    std::vector<Type *> Arguments;
    SmallVector<AttributeSet, 8> ArgumentAttributes;
    SmallVector<bool, 10> ArgumentReplaced(FTy->getNumParams(), false);

    unsigned i = 0;
    for (Function::arg_iterator FA = F->arg_begin(), FE = F->arg_end();
         FA != FE; ++FA, ++i) {
      if (FA->getType()->isPointerTy() &&
          FA->getType()->getPointerAddressSpace() == ADDRESS_SPACE_SHARED) {
        // Replace pointers to shared memory with i32 offsets.
        Arguments.push_back(Type::getInt32Ty(M.getContext()));
        ArgumentAttributes.push_back(
            AttributeSet::get(M.getContext(), ArrayRef<Attribute>{}));
        ArgumentReplaced[i] = true;
      } else {
        // Replace other arguments with the same type as before.
        Arguments.push_back(FA->getType());
        ArgumentAttributes.push_back(FAttrs.getParamAttrs(i));
      }
    }

    // Create new function type.
    AttributeList NAttrs =
        AttributeList::get(F->getContext(), FAttrs.getFnAttrs(),
                           FAttrs.getRetAttrs(), ArgumentAttributes);
    FunctionType *NFTy =
        FunctionType::get(FTy->getReturnType(), Arguments, FTy->isVarArg());
    Function *NF = NVPTX::Utils::splice(F, ArgumentAttributes, NFTy);

    i = 0;
    for (Function::arg_iterator FA = F->arg_begin(), FE = F->arg_end(),
                                NFA = NF->arg_begin();
         FA != FE; ++FA, ++NFA, ++i) {
      Value *NewValueForUse = NFA;
      if (ArgumentReplaced[i]) {
        // If this argument was replaced, then create a `getelementptr`
        // instruction that uses it to recreate the pointer that was replaced.
        auto InsertBefore = &NF->getEntryBlock().front();
        auto PtrInst = GetElementPtrInst::CreateInBounds(
            /* PointeeType= */ SharedMemGlobalType,
            /* Ptr= */ SharedMemGlobal,
            /* IdxList= */
            ArrayRef<Value *>{
                ConstantInt::get(Type::getInt32Ty(M.getContext()), 0, false),
                NFA,
            },
            /* NameStr= */ Twine{NFA->getName()}, InsertBefore);
        // Then create a bitcast to make sure the new pointer is the same type
        // as the old one. This will only ever be a `i8 addrspace(3)*` to `i32
        // addrspace(3)*` type of cast.
        auto CastInst = new BitCastInst(PtrInst, FA->getType());
        CastInst->insertAfter(PtrInst);
        NewValueForUse = CastInst;
      }

      // Replace uses of the old function's argument with the new argument or
      // the result of the `getelementptr`/`bitcast` instructions.
      FA->replaceAllUsesWith(&*NewValueForUse);
      NewValueForUse->takeName(&*FA);
    }

    // There should be no callers of kernel entry points.
    assert(F->use_empty());
    // Now that the old function is dead, delete it.
    F->eraseFromParent();

    return NF;
  }

  virtual llvm::StringRef getPassName() const {
    return "localaccessortosharedmemory";
  }
};

} // end anonymous namespace

char LocalAccessorToSharedMemory::ID = 0;

INITIALIZE_PASS(LocalAccessorToSharedMemory, "localaccessortosharedmemory",
                "SYCL Local Accessor to Shared Memory", false, false)

ModulePass *llvm::createLocalAccessorToSharedMemoryPass() {
  return new LocalAccessorToSharedMemory();
}
