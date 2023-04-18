set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_clang_library(clangInferASPChecker
  ConstraintGather.cpp
  ConstraintGatherCast.cpp
  InferActions.cpp
  TypeBuilder.cpp
  TypePrinter.cpp
  UnifyContext.cpp
  UnifyType.cpp
  Unifier.cpp

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangFrontend
  clangLex
  clangSema
  )
