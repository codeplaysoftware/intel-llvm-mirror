#include "Utils.h"

Optional<SmallVector<std::pair<MDNode *, Function *>>>
NVPTX::Utils::getNVVKernels(Module &M) {
  SmallVector<std::pair<MDNode *, Function *>> NodeKernelPairs;
  auto *NvvmMetadata = M.getNamedMetadata("nvvm.annotations");
  if (!NvvmMetadata)
    return None;

  for (auto *MetadataNode : NvvmMetadata->operands()) {
    if (MetadataNode->getNumOperands() != 3)
      continue;

    // NVPTX identifies kernel entry points using metadata nodes of the form:
    //   !X = !{<function>, !"kernel", i32 1}
    const MDOperand &TypeOperand = MetadataNode->getOperand(1);
    auto *Type = dyn_cast<MDString>(TypeOperand);
    if (!Type)
      continue;
    // Only process kernel entry points.
    if (Type->getString() != "kernel")
      continue;

    // Get a pointer to the entry point function from the metadata.
    const MDOperand &FuncOperand = MetadataNode->getOperand(0);
    if (!FuncOperand)
      continue;
    auto *FuncConstant = dyn_cast<ConstantAsMetadata>(FuncOperand);
    if (!FuncConstant)
      continue;
    auto *Func = dyn_cast<Function>(FuncConstant->getValue());
    if (!Func)
      continue;
    NodeKernelPairs.push_back(std::make_pair(MetadataNode, Func));
  }
  if (NodeKernelPairs.empty())
    return None;

  return NodeKernelPairs;
}

Function *
NVPTX::Utils::splice(Function *OldFn,
                     SmallVectorImpl<AttributeSet> &ArgumentAttributes,
                     FunctionType *NewFnTy, const char *NewFnName) {
  auto &Context = OldFn->getContext();
  auto *M = OldFn->getParent();
  const AttributeList &OldAttributes = OldFn->getAttributes();
  auto NewAttributes =
      AttributeList::get(Context, OldAttributes.getFnAttributes(),
                         OldAttributes.getRetAttributes(), ArgumentAttributes);
  // Create the new function body and insert it into the module.
  Function *NewFn =
      Function::Create(NewFnTy, OldFn->getLinkage(), OldFn->getAddressSpace(),
                       Twine{NewFnName ? NewFnName : ""}, M);
  if (!NewFnName)
    NewFn->takeName(OldFn);
  NewFn->copyAttributesFrom(OldFn);
  NewFn->setComdat(OldFn->getComdat());
  NewFn->setAttributes(NewAttributes);

  // Splice the body of the old function right into the new function.
  NewFn->getBasicBlockList().splice(NewFn->begin(), OldFn->getBasicBlockList());
  // Clone metadata of the old function, including debug info descriptor.
  SmallVector<std::pair<unsigned, MDNode *>, 1> MDNodes;
  OldFn->getAllMetadata(MDNodes);
  for (auto MDNode : MDNodes)
    NewFn->addMetadata(MDNode.first, *MDNode.second);

  return NewFn;
}
