//===- AggPeel.cpp - Aggregates Peeling   ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This transformation implements an aggregate peeler, that is akin to SROA.
/// It tries to identify portion of struct aggregates that can be split
/// into multiple allocas (one per fields).
///
/// The purpose is to give SROA more opportunities. The normal algorithm
/// bails out as soon as index are not predictable. The leaves partially
/// breakable structs untouched.
///

#include "llvm/Transforms/Scalar/AggPeel.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/ConstantFolder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "aggpeel"
namespace peel {

static Value *foldSelectInst(SelectInst &SI) {
  // If the condition being selected on is a constant or the same value is
  // being selected between, fold the select. Yes this does (rarely) happen
  // early on.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(SI.getCondition()))
    return SI.getOperand(1 + CI->isZero());
  if (SI.getOperand(1) == SI.getOperand(2))
    return SI.getOperand(1);

  return nullptr;
}

/// A helper that folds a PHI node or a select.
static Value *foldPHINodeOrSelectInst(Instruction &I) {
  if (PHINode *PN = dyn_cast<PHINode>(&I)) {
    // If PN merges together the same value, return that value.
    return PN->hasConstantValue();
  }
  return foldSelectInst(cast<SelectInst>(I));
}

/// A used slice of an alloca.
///
/// This structure represents a slice of an alloca used by some instruction. It
/// stores both the begin and end offsets of this use, a pointer to the use
/// itself, and a flag indicating whether we can classify the use as splittable
/// or not when forming partitions of the alloca.
class PeelSlice {
  /// The beginning offset of the range.
  uint64_t BeginOffset = 0;
  uint64_t EndOffset = 0;

  /// Storage for both the use of this slice and whether it can be
  /// split.
  Use *UsePtr;

  llvm::SmallVector<uint64_t, 8> TypeOffsets;
  Type *GEPType;

public:
  PeelSlice(uint64_t BeginOffset, Use *U)
      : BeginOffset{BeginOffset}, UsePtr(U) {}

  uint64_t beginOffset() const { return BeginOffset; }
  uint64_t endOffset() const { return EndOffset; }
  void setEndOffset(uint64_t End) { EndOffset = End; }

  llvm::SmallVectorImpl<uint64_t> &getIndexes() { return TypeOffsets; }
  void pushIndex(uint64_t idx) { TypeOffsets.push_back(idx); }

  Type *getGEPType() const { return GEPType; }
  void setGEPType(Type *ty) {GEPType = ty; }

  Use *getUse() const { return UsePtr; }

  bool isDead() const { return getUse() == nullptr; }
  void kill() { UsePtr = nullptr; }

  void dumpindices() const {
      for (uint64_t i : TypeOffsets)
        dbgs() << " " << i;
      dbgs() << "\n";
  }

  /// Support for ordering ranges.
  ///
  /// This provides an ordering over ranges such that start offsets are
  /// always increasing, and within equal start offsets, the end offsets are
  /// decreasing. Thus the spanning range comes first in a cluster with the
  /// same start position.
  bool operator<(const PeelSlice &RHS) const {
    if (beginOffset() < RHS.beginOffset())
      return true;
    if (beginOffset() > RHS.beginOffset())
      return false;
    if (endOffset() > RHS.endOffset())
      return true;
    return false;
      return false;
  }

#if 0
  bool operator==(const PeelSlice &RHS) const {
    return TypeOffsets == RHS.TypeOffsets;
  }
  bool operator!=(const PeelSlice &RHS) const { return !operator==(RHS); }
#endif
};

class AllocaPeels {
public:
  /// Construct the peels of a particular alloca.
  AllocaPeels(const DataLayout &DL, AllocaInst &AI);

  /// Test whether a pointer to the allocation escapes our analysis.
  ///
  /// If this is true, the slices are never fully built and should be
  /// ignored.
  bool isEscaped() const { return PointerEscapingInstr; }
  bool isAborted() const { return WasAborted; }

  /// Support for iterating over the slices.
  /// @{
  using iterator = SmallVectorImpl<PeelSlice>::iterator;
  using range = iterator_range<iterator>;

  iterator begin() { return PeelSlices.begin(); }
  iterator end() { return PeelSlices.end(); }

  using const_iterator = SmallVectorImpl<PeelSlice>::const_iterator;
  using const_range = iterator_range<const_iterator>;

  const_iterator begin() const { return PeelSlices.begin(); }
  const_iterator end() const { return PeelSlices.end(); }
  /// @}

  /// Erase a range of slices.
  void erase(iterator Start, iterator Stop) { PeelSlices.erase(Start, Stop); }

  /// Insert new slices for this alloca.
  ///
  /// This moves the slices into the alloca's slices collection, and re-sorts
  /// everything so that the usual ordering properties of the alloca's slices
  /// hold.
  void insert(ArrayRef<PeelSlice> NewPeelSlices) {
    int OldSize = PeelSlices.size();
    PeelSlices.append(NewPeelSlices.begin(), NewPeelSlices.end());
    auto PeelSliceI = PeelSlices.begin() + OldSize;
    llvm::sort(PeelSliceI, PeelSlices.end());
    std::inplace_merge(PeelSlices.begin(), PeelSliceI, PeelSlices.end());
  }

  /// Access the dead users for this alloca.
  ArrayRef<Instruction *> getDeadUsers() const { return DeadUsers; }

  /// Access Uses that should be dropped if the alloca is promotable.
  ArrayRef<Use *> getDeadUsesIfPromotable() const {
    return DeadUseIfPromotable;
  }

  /// Access the dead operands referring to this alloca.
  ///
  /// These are operands which have cannot actually be used to refer to the
  /// alloca as they are outside its range and the user doesn't correct for
  /// that. These mostly consist of PHI node inputs and the like which we just
  /// need to replace with undef.
  ArrayRef<Use *> getDeadOperands() const { return DeadOperands; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
   void print(raw_ostream &OS, const_iterator I, StringRef Indent = "  ") const;
   void printPeelSlice(raw_ostream &OS, const_iterator I,
                       StringRef Indent = "  ") const;
   void printUse(raw_ostream &OS, const_iterator I,
                 StringRef Indent = "  ") const;
   void print(raw_ostream &OS) const;
   void dump(const_iterator I) const;
   void dump() const;
#endif

private:
  template <typename DerivedT, typename RetT = void> class BuilderBase;
  friend class AggPeelerBuilder;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Handle to alloca instruction to simplify method interfaces.
  AllocaInst &AI;
#endif

  /// The instruction responsible for this alloca not having a known set
  /// of slices.
  ///
  /// When an instruction (potentially) escapes the pointer to the alloca, we
  /// store a pointer to that here and abort trying to form slices of the
  /// alloca. This will be null if the alloca slices are analyzed successfully.
  Instruction *PointerEscapingInstr;
  // Tell if the escaping pointer is actually an aborted exec.
  bool WasAborted;

  /// The slices of the alloca.
  ///
  /// We store a vector of the slices formed by uses of the alloca here. This
  /// vector is sorted by increasing begin offset, and then the unsplittable
  /// slices before the splittable ones. See the PeelSlice inner class for more
  /// details.
  SmallVector<PeelSlice, 8> PeelSlices;

  /// Instructions which will become dead if we rewrite the alloca.
  ///
  /// Note that these are not separated by slice. This is because we expect an
  /// alloca to be completely rewritten or not rewritten at all. If rewritten,
  /// all these instructions can simply be removed and replaced with undef as
  /// they come from outside of the allocated space.
  SmallVector<Instruction *, 8> DeadUsers;

  /// Uses which will become dead if can promote the alloca.
  SmallVector<Use *, 8> DeadUseIfPromotable;

  /// Operands which will become dead if we rewrite the alloca.
  ///
  /// These are operands that in their particular use can be replaced with
  /// undef when we rewrite the alloca. These show up in out-of-bounds inputs
  /// to PHI nodes and the like. They aren't entirely dead (there might be
  /// a GEP back into the bounds using it elsewhere) and nor is the PHI, but we
  /// want to swap this particular input for undef to simplify the use lists of
  /// the alloca.
  SmallVector<Use *, 8> DeadOperands;
};

/// Builder for the alloca slices.
///
/// This class builds a set of alloca slices by recursively visiting the uses
/// of an alloca and making a slice for each load and store at each offset.
class AggPeelerBuilder
    : public PtrUseVisitor<AggPeelerBuilder, /*TrackLowerBound =*/true> {
  friend class PtrUseVisitor<AggPeelerBuilder, true>;
  friend class InstVisitor<AggPeelerBuilder>;

  using Base = PtrUseVisitor<AggPeelerBuilder, true>;

  peel::AllocaPeels &AP;
  Type* AllocatedType;
  const uint64_t AllocSize;

  SmallDenseMap<Instruction *, unsigned> MemTransferSliceMap;
  SmallDenseMap<Instruction *, uint64_t> PHIOrSelectSizes;

  /// Set to de-duplicate dead instructions found in the use walk.
  SmallPtrSet<Instruction *, 4> VisitedDeadInsts;

public:
  AggPeelerBuilder(const DataLayout &DL, AllocaInst &AI, peel::AllocaPeels &AP)
      : PtrUseVisitor<AggPeelerBuilder, true>(DL), AP(AP),
        AllocatedType(AI.getAllocatedType()),
        AllocSize(DL.getTypeAllocSize(AllocatedType).getFixedSize()) {}

private:
  void markAsDead(Instruction &I) {
    if (VisitedDeadInsts.insert(&I).second)
      AP.DeadUsers.push_back(&I);
  }

  static Type *buildIndexListFromOffset(const DataLayout &DL, Type *Ty,
                                        llvm::SmallVectorImpl<uint64_t> &Idxs,
                                        APInt Offset, Type *FinalType) {
    SmallVector<APInt> IdxList = DL.getGEPIndicesForOffset(Ty, Offset);
    auto ToUIntMap =
        map_range(IdxList, [](APInt &I) { return I.getZExtValue(); });
    Idxs.insert(Idxs.begin(), ToUIntMap.begin(), ToUIntMap.end());

#if 0
    while (Ty != FinalType) {
      Idxs.push_back(0);
      if (!(isa<StructType>(Ty) ||  isa<ArrayType>(Ty))) {
        // Not a struct or array, we are done.
        return Ty;
      }
      Ty = Ty->getContainedType(0);
    }
#endif

    return Ty;
  }

  void insertUse(Instruction &I, const APInt &Offset, uint64_t Size) {
    // Completely skip uses which have a zero size or start either before or
    // past the end of the allocation.
    if (Size == 0 || Offset.uge(AllocSize)) {
      LLVM_DEBUG(dbgs() << "WARNING: Ignoring " << Size << " byte use @"
                        << Offset
                        << " which has zero size or starts outside of the "
                        << AllocSize << " byte alloca:\n"
                        << "    alloca: " << AP.AI << "\n"
                        << "       use: " << I << "\n");
      return markAsDead(I);
    }

    PeelSlice Slice{Offset.getZExtValue(), U};

    Type *Ty = buildIndexListFromOffset(DL, AllocatedType, Slice.getIndexes(),
                                        Offset, LowerBoundType);
    Slice.setGEPType(Ty);
    LLVM_DEBUG(dbgs() << "-> " << Slice.beginOffset() << ";" <<
        Size << ": insertUse\n");
    if (U) {
        LLVM_DEBUG(U->get()->dump());
        LLVM_DEBUG(U->getUser()->dump());
    }
    LLVM_DEBUG(dbgs() << "getGEP Type: ");
    LLVM_DEBUG(Ty->dump());
    LLVM_DEBUG(dbgs() << "GEP indices:");
    LLVM_DEBUG(Slice.dumpindices());
    if (LowerBoundType) {
      Slice.setEndOffset(Slice.beginOffset() + DL.getTypeStoreSize(Ty));
      LLVM_DEBUG(dbgs() << "LowerBoundType: ");
      LLVM_DEBUG(LowerBoundType->dump());
    } else {
      Slice.setEndOffset(Slice.beginOffset() + Size);
    }

    AP.PeelSlices.emplace_back(std::move(Slice));
  }

  void visitBitCastInst(BitCastInst &BC) {
    if (BC.use_empty())
      return markAsDead(BC);

    return Base::visitBitCastInst(BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &APC) {
    if (APC.use_empty())
      return markAsDead(APC);

    return Base::visitAddrSpaceCastInst(APC);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEPI) {
    if (GEPI.use_empty())
      return markAsDead(GEPI);

    return Base::visitGetElementPtrInst(GEPI);
  }

  void handleLoadOrStore(Type *Ty, Instruction &I, const APInt &Offset,
                         uint64_t Size, bool IsVolatile) {
    insertUse(I, Offset, Size);
  }

  void visitLoadInst(LoadInst &LI) {
    assert((!LI.isSimple() || LI.getType()->isSingleValueType()) &&
           "All simple FCA loads should have been pre-split");

    if (!IsOffsetKnown)
      return PI.setAborted(&LI);

    if (LI.isVolatile())
      return PI.setAborted(&LI);

    if (isa<ScalableVectorType>(LI.getType()))
      return PI.setAborted(&LI);

    uint64_t Size = DL.getTypeStoreSize(LI.getType()).getFixedSize();
    return handleLoadOrStore(LI.getType(), LI, Offset, Size, LI.isVolatile());
  }

  void visitStoreInst(StoreInst &SI) {
    Value *ValOp = SI.getValueOperand();
    if (ValOp == *U)
      return PI.setEscapedAndAborted(&SI);
    if (!IsOffsetKnown)
      return PI.setAborted(&SI);

    if (SI.isVolatile())
      return PI.setAborted(&SI);

    if (isa<ScalableVectorType>(ValOp->getType()))
      return PI.setAborted(&SI);

    uint64_t Size = DL.getTypeStoreSize(ValOp->getType()).getFixedSize();

    // If this memory access can be shown to *statically* extend outside the
    // bounds of the allocation, it's behavior is undefined, so simply
    // ignore it. Note that this is more strict than the generic clamping
    // behavior of insertUse. We also try to handle cases which might run the
    // risk of overflow.
    // FIXME: We should instead consider the pointer to have escaped if this
    // function is being instrumented for addressing bugs or race conditions.
    if (Size > AllocSize || Offset.ugt(AllocSize - Size)) {
      LLVM_DEBUG(dbgs() << "WARNING: Ignoring " << Size << " byte store @"
                        << Offset << " which extends past the end of the "
                        << AllocSize << " byte alloca:\n"
                        << "    alloca: " << AP.AI << "\n"
                        << "       use: " << SI << "\n");
      return markAsDead(SI);
    }

    assert((!SI.isSimple() || ValOp->getType()->isSingleValueType()) &&
           "All simple FCA stores should have been pre-split");
    handleLoadOrStore(ValOp->getType(), SI, Offset, Size, SI.isVolatile());
  }

  void visitMemSetInst(MemSetInst &II) {
    assert(II.getRawDest() == *U && "Pointer use is not the destination?");
    ConstantInt *Length = dyn_cast<ConstantInt>(II.getLength());
    if ((Length && Length->getValue() == 0) ||
        (IsOffsetKnown && Offset.uge(AllocSize)))
      // Zero-length mem transfer intrinsics can be ignored entirely.
      return markAsDead(II);

    if (!IsOffsetKnown)
      return PI.setAborted(&II);

    // Don't replace this with a store with a different address space.  TODO:
    // Use a store with the casted new alloca?
    if (II.isVolatile() && II.getDestAddressSpace() != DL.getAllocaAddrSpace())
      return PI.setAborted(&II);

    insertUse(II, Offset,
              Length ? Length->getLimitedValue()
                     : AllocSize - Offset.getLimitedValue());
  }

  void visitMemTransferInst(MemTransferInst &II) {
    ConstantInt *Length = dyn_cast<ConstantInt>(II.getLength());
    if (Length && Length->getValue() == 0)
      // Zero-length mem transfer intrinsics can be ignored entirely.
      return markAsDead(II);

    // Because we can visit these intrinsics twice, also check to see if the
    // first time marked this instruction as dead. If so, skip it.
    if (VisitedDeadInsts.count(&II))
      return;

    if (!IsOffsetKnown)
      return PI.setAborted(&II);

    // Don't replace this with a load/store with a different address space.
    // TODO: Use a store with the casted new alloca?
    if (II.isVolatile())
      return PI.setAborted(&II);

    // This side of the transfer is completely out-of-bounds, and so we can
    // nuke the entire transfer. However, we also need to nuke the other side
    // if already added to our partitions.
    // FIXME: Yet another place we really should bypass this when
    // instrumenting for APan.
    if (Offset.uge(AllocSize)) {
      SmallDenseMap<Instruction *, unsigned>::iterator MTPI =
          MemTransferSliceMap.find(&II);
      if (MTPI != MemTransferSliceMap.end())
        AP.PeelSlices[MTPI->second].kill();
      return markAsDead(II);
    }

    uint64_t RawOffset = Offset.getLimitedValue();
    uint64_t Size = Length ? Length->getLimitedValue() : AllocSize - RawOffset;

    // Check for the special case where the same exact value is used for both
    // source and dest.
    if (*U == II.getRawDest() && *U == II.getRawSource()) {
      // For non-volatile transfers this is a no-op.
      if (!II.isVolatile())
        return markAsDead(II);

      return insertUse(II, Offset, Size);
    }

    // If we have seen both source and destination for a mem transfer, then
    // they both point to the same alloca.
    bool Inserted;
    SmallDenseMap<Instruction *, unsigned>::iterator MTPI;
    std::tie(MTPI, Inserted) =
        MemTransferSliceMap.insert(std::make_pair(&II, AP.PeelSlices.size()));
    unsigned PrevIdx = MTPI->second;
    if (!Inserted) {
      PeelSlice &PrevP = AP.PeelSlices[PrevIdx];

      // Check if the begin offsets match and this is a non-volatile transfer.
      // In that case, we can completely elide the transfer.
      if (!II.isVolatile() && PrevP.beginOffset() == RawOffset) {
        PrevP.kill();
        return markAsDead(II);
      }
    }

    // Insert the use now that we've fixed up the splittable nature.
    insertUse(II, Offset, Size);

    // Check that we ended up with a valid index in the map.
    assert(AP.PeelSlices[PrevIdx].getUse()->getUser() == &II &&
           "Map index doesn't point back to a slice with this user.");
  }

  // Disable SRoA for any intrinsics except for lifetime invariants and
  // invariant group.
  // FIXME: What about debug intrinsics? This matches old behavior, but
  // doesn't make sense.
  void visitIntrinsicInst(IntrinsicInst &II) {
    if (II.isDroppable()) {
      AP.DeadUseIfPromotable.push_back(U);
      return;
    }

    if (!IsOffsetKnown)
      return PI.setAborted(&II);

    if (II.isLifetimeStartOrEnd()) {
      ConstantInt *Length = cast<ConstantInt>(II.getArgOperand(0));
      uint64_t Size = std::min(AllocSize - Offset.getLimitedValue(),
                               Length->getLimitedValue());
      insertUse(II, Offset, Size);
      return;
    }

    if (II.isLaunderOrStripInvariantGroup()) {
      enqueueUsers(II);
      return;
    }

    Base::visitIntrinsicInst(II);
  }

  Instruction *hasUnsafePHIOrSelectUse(Instruction *Root, uint64_t &Size) {
    // We consider any PHI or select that results in a direct load or store of
    // the same offset to be a viable use for slicing purposes. These uses
    // are considered unsplittable and the size is the maximum loaded or
    // stored size.
    SmallPtrSet<Instruction *, 4> Visited;
    SmallVector<std::pair<Instruction *, Instruction *>, 4> Uses;
    Visited.insert(Root);
    Uses.push_back(std::make_pair(cast<Instruction>(*U), Root));
    const DataLayout &DL = Root->getModule()->getDataLayout();
    // If there are no loads or stores, the access is dead. We mark that as
    // a size zero access.
    Size = 0;
    do {
      Instruction *I, *UsedI;
      std::tie(UsedI, I) = Uses.pop_back_val();

      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        Size =
            std::max(Size, DL.getTypeStoreSize(LI->getType()).getFixedSize());
        continue;
      }
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        Value *Op = SI->getOperand(0);
        if (Op == UsedI)
          return SI;
        Size =
            std::max(Size, DL.getTypeStoreSize(Op->getType()).getFixedSize());
        continue;
      }

      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
        if (!GEP->hasAllZeroIndices())
          return GEP;
      } else if (!isa<BitCastInst>(I) && !isa<PHINode>(I) &&
                 !isa<SelectInst>(I) && !isa<AddrSpaceCastInst>(I)) {
        return I;
      }

      for (User *U : I->users())
        if (Visited.insert(cast<Instruction>(U)).second)
          Uses.push_back(std::make_pair(I, cast<Instruction>(U)));
    } while (!Uses.empty());

    return nullptr;
  }

  void visitPHINodeOrSelectInst(Instruction &I) {
    assert(isa<PHINode>(I) || isa<SelectInst>(I));
    if (I.use_empty())
      return markAsDead(I);

    // TODO: We could use SimplifyInstruction here to fold PHINodes and
    // SelectInsts. However, doing so requires to change the current
    // dead-operand-tracking mechanism. For instance, suppose neither loading
    // from %U nor %other traps. Then "load (select undef, %U, %other)" does
    // not trap either.  However, if we simply replace %U with undef using the
    // current dead-operand-tracking mechanism, "load (select undef, undef,
    // %other)" may trap because the select may return the first operand
    // "undef".
    if (Value *Result = foldPHINodeOrSelectInst(I)) {
      if (Result == *U)
        // If the result of the constant fold will be the pointer, recurse
        // through the PHI/select as if we had RAUW'ed it.
        enqueueUsers(I);
      else
        // Otherwise the operand to the PHI/select is dead, and we can replace
        // it with undef.
        AP.DeadOperands.push_back(U);

      return;
    }

    if (!IsOffsetKnown)
      return PI.setAborted(&I);

    // See if we already have computed info on this node.
    uint64_t &Size = PHIOrSelectSizes[&I];
    if (!Size) {
      // This is a new PHI/Select, check for an unsafe use of it.
      if (Instruction *UnsafeI = hasUnsafePHIOrSelectUse(&I, Size))
        return PI.setAborted(UnsafeI);
    }

    // For PHI and select operands outside the alloca, we can't nuke the
    // entire phi or select -- the other side might still be relevant, so we
    // special case them here and use a separate structure to track the
    // operands themselves which should be replaced with undef.
    // FIXME: This should instead be escaped in the event we're instrumenting
    // for address sanitization.
    if (Offset.uge(AllocSize)) {
      AP.DeadOperands.push_back(U);
      return;
    }

    insertUse(I, Offset, Size);
  }

  void visitPHINode(PHINode &PN) { visitPHINodeOrSelectInst(PN); }

  void visitSelectInst(SelectInst &SI) { visitPHINodeOrSelectInst(SI); }

  /// Disable SROA entirely if there are unhandled users of the alloca.
  void visitInstruction(Instruction &I) { PI.setAborted(&I); }
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

void AllocaPeels::print(raw_ostream &OS, const_iterator I,
                         StringRef Indent) const {
  printPeelSlice(OS, I, Indent);
  OS << "\n";
  printUse(OS, I, Indent);
}

void AllocaPeels::printPeelSlice(raw_ostream &OS, const_iterator I,
                              StringRef Indent) const {
  OS << Indent << "[" << I->beginOffset() << "," << I->endOffset() << ")"
     << " slice #" << (I - begin());
}

void AllocaPeels::printUse(raw_ostream &OS, const_iterator I,
                            StringRef Indent) const {
  OS << Indent << "  used by: " << *I->getUse()->getUser() << "\n";
}

void AllocaPeels::print(raw_ostream &OS) const {
  if (PointerEscapingInstr) {
    OS << "Can't analyze slices for alloca: " << AI << "\n";
    if (isAborted())
      OS   << "  A pointer to this alloca escaped (due to abort) by:\n";
    else
      OS   << "  A pointer to this alloca escaped by:\n";
    OS << "  " << *PointerEscapingInstr << "\n";
    return;
  }

  OS << "AggPeel Slices of alloca: " << AI << "\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    print(OS, I);
}

LLVM_DUMP_METHOD void AllocaPeels::dump(const_iterator I) const {
  print(dbgs(), I);
}
LLVM_DUMP_METHOD void AllocaPeels::dump() const { print(dbgs()); }

#endif // !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
AllocaPeels::AllocaPeels(const DataLayout &DL, AllocaInst &AI)
    :
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      AI(AI),
#endif
      PointerEscapingInstr(nullptr) {
  AggPeelerBuilder PB(DL, AI, *this);
  detail::PtrUseVisitorBase::PtrInfo PtrI = PB.visitPtr(AI);
  if (PtrI.isEscaped() || PtrI.isAborted()) {
    PointerEscapingInstr = PtrI.getEscapingInst() ? PtrI.getEscapingInst()
                                                  : PtrI.getAbortingInst();
    assert(PointerEscapingInstr && "Did not track a bad instruction");
    return;
  }

  llvm::erase_if(PeelSlices, [](const PeelSlice &S) { return S.isDead(); });

  // Sort the uses. This arranges for the offsets to be in ascending order,
  // and the sizes to be in descending order.
  llvm::stable_sort(PeelSlices);
}


class AggPeel {
  LLVMContext *C = nullptr;
  DominatorTree *DT = nullptr;
  AssumptionCache *AC = nullptr;

  /// Worklist of alloca instructions to rewrite
  SetVector<AllocaInst *, SmallVector<AllocaInst *, 16>> Worklist;

  bool runOnAlloca(AllocaInst &AI);
  void rewrite(AllocaInst &NewAI,
    AllocaPeels::iterator &itbegin,
    AllocaPeels::iterator &itlast);
public:
  PreservedAnalyses runImpl(Function &, DominatorTree &, AssumptionCache &);
};

bool AggPeel::runOnAlloca(AllocaInst &AI) {
  LLVM_DEBUG(dbgs() << "AggPeel alloca: " << AI << "\n");
      
  // Special case dead allocas, as they're trivial.
  if (AI.use_empty()) { 
    AI.eraseFromParent();
    return true;
  }
  const DataLayout &DL = AI.getModule()->getDataLayout();

  // Skip alloca forms that this analysis can't handle.
  auto *AT = AI.getAllocatedType();
  if (AI.isArrayAllocation() || !AT->isSized() || isa<ScalableVectorType>(AT) ||
      DL.getTypeAllocSize(AT).getFixedSize() == 0)
    return false;

  peel::AllocaPeels AP(DL, AI);
  // AP has the slices
  LLVM_DEBUG(AP.print(dbgs()));
  if (AP.isEscaped())
    return false;

  bool Changed = false;
  // FIXME should use const_iterator but then we can't getIndexes so we'd
  // have to use iterators for that... lazy
  AllocaPeels::iterator it = AP.begin();
  for (;;) {
    for (; it != AP.end(); it++) {
      uint64_t Size = it->endOffset() - it->beginOffset();
      Use *U = it->getUse();
      assert(U && "Don't have a use");
      auto I = dyn_cast<Instruction>(U->getUser());
      assert(I && "User is not an instruction");
      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
          if (DL.getTypeStoreSize(LI->getType()).getFixedSize() == Size) {
            continue;
          }
          break;
      }
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        Value *ValOp = SI->getValueOperand();
        if (DL.getTypeStoreSize(ValOp->getType()).getFixedSize() == Size) {
          continue;
        }
        break;
      }
      if (dyn_cast<MemTransferInst>(I)) {
        break;
      }
      if (dyn_cast<MemSetInst>(I)) {
        break;
      }
      if (dyn_cast<IntrinsicInst>(I)) {
        continue;
      }
      if (isa<PHINode>(*I) || isa<SelectInst>(*I)) {
        llvm::errs() << "Can't handle PHI or Select: ";
        I->dump();
        continue;
      }
      break;
    }
    if (it == AP.end())
      break;
    // now we should have the beginning of a partition to rewrite
    Changed = true;
    LLVM_DEBUG(llvm::dbgs() << "Begin partition: " << "[" <<
      it->beginOffset() << "," << it->endOffset() <<
      ")" << " slice #" << (it - AP.begin()) << "\n");
    AllocaPeels::iterator itbegin = it;
    AllocaPeels::iterator itend = it;
    for (it++; it != AP.end(); itend = it, it++) {
      if (it->beginOffset() >= itbegin->beginOffset() &&
          it->endOffset() <= itbegin->endOffset())
        continue;
      break;
    }
    AllocaPeels::iterator itlast = it;
    LLVM_DEBUG(llvm::dbgs() << "End partition: " << "[" <<
      itend->beginOffset() << "," << itend->endOffset() <<
      ")" << " slice #" << (itend - AP.begin()) << "\n");
    // Now rewrite the slices from itbegin to itend
    LLVM_DEBUG(llvm::dbgs() << "Alloca Type: ");
    LLVM_DEBUG(itbegin->getGEPType()->dump());
    AllocaInst *NewAI = new AllocaInst(
      itbegin->getGEPType(), AI.getType()->getAddressSpace(), nullptr,
      AI.getAlign(),
      AI.getName() + ".aggpeel." + Twine(itbegin - AP.begin()),
      &AI);
    // Copy the old AI debug location over to the new one.
    NewAI->setDebugLoc(AI.getDebugLoc());
    LLVM_DEBUG(dbgs() << "New alloc: ");
    LLVM_DEBUG(NewAI->dump());
    rewrite(*NewAI, itbegin, itlast);

    if (it == AP.end())
      break;
  }


  return Changed;
}

void AggPeel::rewrite(AllocaInst &NewAI,
    AllocaPeels::iterator &itbegin,
    AllocaPeels::iterator &itlast) {
  // save the indices for the aggrgate itself
  llvm::SmallVectorImpl<uint64_t> &indices = itbegin->getIndexes();
  // new geps so we don't replace them multiple times
  SmallPtrSet<Instruction *, 4> newgeps;
  for (AllocaPeels::iterator itrw = itbegin; itrw != itlast; itrw++) {
    Use *U = itrw->getUse();
    LLVM_DEBUG(llvm::dbgs() << "rewriting use: ");
    LLVM_DEBUG(U->get()->dump());
    GetElementPtrInst *gep;
    // FIXME? Is there a better way to do this?
    if (BitCastInst *BI = dyn_cast<BitCastInst>(U->get()))
      gep = dyn_cast<GetElementPtrInst>(BI->getOperand(0));
    else
      gep = dyn_cast<GetElementPtrInst>(U->get());
    assert(gep && "Can't find GEP instruction");
    LLVM_DEBUG(gep->dump());

    if (indices.size() == gep->getNumIndices()) {
      LLVM_DEBUG(dbgs() << "Replacing gep with alloca value\n");
      BasicBlock::iterator ii(gep);
      ReplaceInstWithValue(gep->getParent()->getInstList(), ii, &NewAI);
    }
    else {
      // make the new gep
      if (newgeps.contains(gep)) {
        LLVM_DEBUG(dbgs() << "gep already rewritten\n:");
        continue;
      }
      SmallVector<Value *, 4> newindices;
      newindices.push_back(ConstantInt::get(*C, APInt(64, 0L, true)));
      newindices.append(gep->idx_begin() + indices.size(), gep->idx_end());
      GetElementPtrInst *newgep = GetElementPtrInst::Create(
        NewAI.getAllocatedType(), &NewAI, newindices);
      newgep->setIsInBounds(gep->isInBounds());
      newgeps.insert(newgep);
      LLVM_DEBUG(dbgs() << "newgep: ");
      LLVM_DEBUG(newgep->dump());
      // replace the gep
      BasicBlock::iterator ii(gep);
      ReplaceInstWithInst(gep->getParent()->getInstList(), ii, newgep);
    }
  }
}



PreservedAnalyses AggPeel::runImpl(Function &F, DominatorTree &RunDT,
                                AssumptionCache &RunAC) {
  LLVM_DEBUG(dbgs() << "AggPeel function: " << F.getName() << "\n");
  C = &F.getContext();
  DT = &RunDT;
  AC = &RunAC;

  bool Changed = false;

  BasicBlock &EntryBB = F.getEntryBlock();
  for (BasicBlock::iterator I = EntryBB.begin(), E = std::prev(EntryBB.end());
       I != E; ++I) {
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) {
      if (isa<ScalableVectorType>(AI->getAllocatedType())) {
        //if (isAllocaPromotable(AI))
        //  PromotableAllocas.push_back(AI);
      } else {
        Changed |= runOnAlloca(*AI);
      }
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
} // namespace peel

PreservedAnalyses AggPeelPass::run(Function &F, FunctionAnalysisManager &AM) {
  peel::AggPeel P;
  return P.runImpl(F, AM.getResult<DominatorTreeAnalysis>(F),
                 AM.getResult<AssumptionAnalysis>(F));
}

///
class AggPeelLegacyPass : public FunctionPass {
  peel::AggPeel P;

public:
  static char ID;

  AggPeelLegacyPass() : FunctionPass(ID) {
    initializeAggPeelLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto PA = P.runImpl(
        F, getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F));
    return !PA.areAllPreserved();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.setPreservesCFG();
  }

  StringRef getPassName() const override { return "AggPeel"; }
};

char AggPeelLegacyPass::ID = 0;

FunctionPass *llvm::createAggPeelPass() { return new AggPeelLegacyPass(); }

INITIALIZE_PASS_BEGIN(AggPeelLegacyPass, "aggpeel",
                      "Peel aggregates into separate allocas", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AggPeelLegacyPass, "aggpeel",
    "Peel aggregates into separate allocas",
                    false, false)
