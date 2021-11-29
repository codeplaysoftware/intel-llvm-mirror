//===- AggPeeling.cpp - Aggregates Peeling   ------------------------------===//
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
//===----------------------------------------------------------------------===//

namespace peel {

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

public:
  PeelSlice(uint64_t BeginOffset, Use *U)
      : BeginOffset{BeginOffset}, UsePtr(U) {}

  uint64_t beginOffset() const { return BeginOffset; }
  uint64_t endOffset() const { return EndOffset; }
  void setEndOffset(uint64_t End) { EndOffset = End; }

  llvm::SmallVectorImpl<uint64_t> &getIndexes() { return TypeOffsets; }
  void pushIndex(uint64_t idx) { TypeOffsets.push_back(idx); }

  Use *getUse() const { return UsePtr; }

  bool isDead() const { return getUse() == nullptr; }
  void kill() { UsePtr = nullptr; }

  /// Support for ordering ranges.
  ///
  /// This provides an ordering over ranges such that start offsets are
  /// always increasing, and within equal start offsets, the end offsets are
  /// decreasing. Thus the spanning range comes first in a cluster with the
  /// same start position.
  bool operator<(const PeelSlice &RHS) const {
    size_t i = 0;
    for (; i < TypeOffsets.size(); i++) {
      if (TypeOffsets[i] == RHS.TypeOffsets[i]) {
        break;
      }
    }
    if (i == TypeOffsets.size()) {
      return false;
    }
    uint64_t Offset = TypeOffsets[i];
    uint64_t RHSOffset = TypeOffsets[i];
    if (Offset < RHSOffset)
      return true;
    if (Offset > RHSOffset)
      return false;
    return false;
  }

  bool operator==(const PeelSlice &RHS) const {
    return TypeOffsets == RHS.TypeOffsets;
  }
  bool operator!=(const PeelSlice &RHS) const { return !operator==(RHS); }
};

class AllocaPeels {
public:
  /// Construct the peels of a particular alloca.
  AllocaPeels(const DataLayout &DL, AllocaInst &AI, AllocaSlices &AS,
              detail::PtrUseVisitorBase::PtrInfo &PtrI);

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
  // void print(raw_ostream &OS, const_iterator I, StringRef Indent = "  ") const;
  // void printPeelSlice(raw_ostream &OS, const_iterator I,
  //                     StringRef Indent = "  ") const;
  // void printUse(raw_ostream &OS, const_iterator I,
  //               StringRef Indent = "  ") const;
  // void print(raw_ostream &OS) const;
  // void dump(const_iterator I) const;
  // void dump() const;
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

    while (Ty != FinalType) {
      Idxs.push_back(0);
      if (!(isa<StructType>(Ty) ||  isa<ArrayType>(Ty))) {
        // Not a struct or array, we are done.
        return Ty;
      }
      Ty = Ty->getContainedType(0);
    }

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
    
    if (LowerBoundType) {
      Slice.setEndOffset(Slice.beginOffset() + DL.getTypeStoreSize(Ty));
      if (U) {
        LowerBoundType->dump();
        U->get()->dump();
        U->getUser()->dump();
        llvm::errs() << "-> " << DL.getTypeStoreSize(Ty) << "; " << Slice.beginOffset() << " " << Slice.endOffset() << "\n";
      }
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

AllocaPeels::AllocaPeels(const DataLayout &DL, AllocaInst &AI, AllocaSlices &AS,
                         detail::PtrUseVisitorBase::PtrInfo &PtrI)
    :
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      AI(AI),
#endif
      PointerEscapingInstr(nullptr) {
  AggPeelerBuilder PB(DL, AI, *this);
  PtrI = PB.visitPtr(AI);
  if (PtrI.isEscaped() || PtrI.isAborted()) {
    return;
  }

  llvm::erase_if(PeelSlices, [](const PeelSlice &S) { return S.isDead(); });

  // Sort the uses. This arranges for the offsets to be in ascending order,
  // and the sizes to be in descending order.
  llvm::stable_sort(PeelSlices);

  AS.Slices.clear();
  for (PeelSlice &S : PeelSlices) {
    AS.Slices.push_back(
        Slice(S.beginOffset(), S.endOffset(), S.getUse(), false));
  }
  AS.DeadUsers = std::move(DeadUsers);
  AS.DeadUseIfPromotable = std::move(DeadUseIfPromotable);
  AS.DeadOperands = std::move(DeadOperands);
}

} // namespace peel
