set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(InferASPCheckerTests
  UnifyTest.cpp
  )

clang_target_link_libraries(InferASPCheckerTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangInferASPChecker
  clangSerialization
  clangTooling
  )

target_link_libraries(InferASPCheckerTests
  PRIVATE
  LLVMTestingAnnotations
  LLVMTestingSupport
  clangTesting
  )
