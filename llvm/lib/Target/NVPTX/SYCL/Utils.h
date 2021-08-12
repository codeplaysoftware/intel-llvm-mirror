#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace NVPTX {
namespace Utils {
// Access `nvvm.annotations` to determine which functions are kernel entry
// points.
Optional<SmallVector<std::pair<MDNode *, Function *>>> getNVVKernels(Module &M);

// Create a function of NewFnTy and splice the body of OldFn into it; handle the
// metadata and attributes, but do not replace the uses or remove the OldFn
// function.
Function *splice(Function *OldFn,
                 SmallVectorImpl<AttributeSet> &ArgumentAttributes,
                 FunctionType *NewFnTy, const char *NewFnName = nullptr);
} // namespace Utils
} // namespace NVPTX
