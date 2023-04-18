// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device -ftime-report -fsycl-check-address-space-infer -emit-llvm %s -o %t 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device -ftime-report -emit-llvm %s -o  %t 2>&1 | FileCheck %s --check-prefix=NO-INFER

#include "Inputs/sycl.hpp"

// CHECK: SYCL Address Space Inference Time
// NO-INFER-NOT: SYCL Address Space Inference Time
class f0_kernel;

void f0(sycl::queue &myQueue, sycl::buffer<int, 1> &in_buf,
        sycl::buffer<int, 1> &out_buf) {
  myQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<class f0_kernel>([] {}); });
}
