//===- PtrUseVisitor.cpp - InstVisitors over a pointers uses --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of the pointer use visitors.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include <algorithm>

using namespace llvm;

void detail::PtrUseVisitorBase::enqueueUsers(Instruction &I) {
  for (Use &U : I.uses()) {
    if (VisitedUses.insert(&U).second) {
      UseToVisit NewU = {
        UseToVisit::UseAndIsOffsetKnownPair(&U, IsOffsetKnown),
        LowerBoundType,
        Offset
      };
      Worklist.push_back(std::move(NewU));
    }
  }
}

bool detail::PtrUseVisitorBase::adjustOffsetForGEP(GetElementPtrInst &GEPI,
                                                   bool TrackLowerBound) {
  if (!IsOffsetKnown)
    return false;

  Type * LowBound = nullptr;

  auto GEPLowerBound = [&](Value &V, APInt &ROffset) {
    ROffset = 0;
    if (!LowBound) {
      auto begin = gep_type_begin(&GEPI);
      auto end = gep_type_end(&GEPI);
      for (auto GTI = begin, GTE = end; GTI != GTE; ++GTI) {
        if (GTI.getOperand() == &V) {
          break;
        }
        LowBound = GTI.getIndexedType();
      }
    }
    return true;
  };

  function_ref<bool(Value &, APInt &)> ExternalAnalysis = nullptr;
  if (TrackLowerBound) {
    ExternalAnalysis = GEPLowerBound;
  }

  APInt TmpOffset(DL.getIndexTypeSizeInBits(GEPI.getType()), 0);
  if (cast<GEPOperator>(&GEPI)->accumulateConstantOffset(DL, TmpOffset,
                                                         ExternalAnalysis)) {
    Offset += TmpOffset.sextOrTrunc(Offset.getBitWidth());
    if (TrackLowerBound && !LowerBoundType && LowBound) {
      LowerBoundType = LowBound;
    }
    return true;
  }

  return false;
}
