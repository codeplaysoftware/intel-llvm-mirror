// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s

template <class T, int N>
using raw_vector = T __attribute__((ext_vector_type(N)));

__attribute__((sycl_device)) void basic() {
  raw_vector<int, 4> v1 = {1, 2, 3, 4};
  raw_vector<int, 4> v2{1, 2, 3, 4};
  raw_vector<int, 4> v3 = 1;
  int arr[] = {1, 2, 3};
}
// CHECK: basic : () -> void
// CHECK-NEXT: raw_vector<int, 4> v1 = {1, 2, 3, 4} : <int> __private
// CHECK-NEXT: raw_vector<int, 4> v2{1, 2, 3, 4} : <int> __private
// CHECK-NEXT: raw_vector<int, 4> v3 = 1 : <int> __private
// CHECK-NEXT: int arr[] = {1, 2, 3} : int[] __private

const int gv = 42;

__attribute__((sycl_device)) void sub_evalution(int i) {
  int *j;
  raw_vector<int, 4> v2 = {*(j = &i), 1, 2, 3};
}
// CHECK: sub_evalution : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j : int __private* __private
// CHECK-NEXT: raw_vector<int, 4> v2 = {*(j = &i), 1, 2, 3} : <int> __private

__attribute__((sycl_device)) void sub_evalution_array(int i) {
  int *j = &i;
  const int *arr[] = {&i, j};
  const int *arr2[] = {&gv};
  const int *arr3[1];
  arr3[0] = &gv;
}
// CHECK: sub_evalution_array : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j = &i : int __private* __private
// CHECK-NEXT: const int *arr[] = {&i, j} : int __private*[] __private
// CHECK-NEXT: const int *arr2[] = {&gv} : int __constant*[] __private
// CHECK-NEXT: const int *arr3[1] : int __constant*[] __private

__attribute__((sycl_device)) void vector_subscript() {
  raw_vector<int, 4> v2 = {0, 1, 2, 3};
  (void)v2[0];
}
// CHECK: vector_subscript : () -> void
// CHECK-NEXT: raw_vector<int, 4> v2 = {0, 1, 2, 3} : <int> __private
