// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s

__attribute__((sycl_device)) int *
ptr_cast(__attribute__((opencl_global)) int *l) {
  return l;
}
// CHECK: ptr_cast : (int __global*) -> int __global*
// CHECK-NEXT: __global int *l : int __global* __private

__attribute__((sycl_device)) void
ptr_cast2(__attribute__((opencl_global)) int *l) {
  int &i = *l;
}
// CHECK: ptr_cast2 : (int __global*) -> void
// CHECK-NEXT: __global int *l : int __global* __private
// CHECK-NEXT: int &i = *l : int __global& __private
