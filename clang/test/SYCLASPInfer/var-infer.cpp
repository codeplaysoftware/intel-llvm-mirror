// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s

#include "Inputs/sycl.hpp"

// Test basic propagation
__attribute__((sycl_device)) void test(int i) { int *j = &i; }

// CHECK: test : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j = &i : int __private* __private

// Return propagation
__attribute__((sycl_device)) int *test_return(int i) { return &i; }

// CHECK: test_return : (int) -> int __private*
// CHECK-NEXT: int i : int __private

// Symbolic argument
__attribute__((sycl_device)) void test_simplification(int *i, int j) { i = &j; }

// CHECK: test_simplification : (int __private*, int) -> void
// CHECK-NEXT: int *i : int __private* __private
// CHECK-NEXT: int j : int __private

// Symbolic argument, linked
__attribute__((sycl_device)) void test_simplification2(int *i, int *j) {
  i = j;
}

// CHECK: test_simplification2 : (int [[SLOT:0x.*]]*, int [[SLOT]]*) -> void
// CHECK-NEXT: int *i : int [[SLOT]]* __private
// CHECK-NEXT: int *j : int [[SLOT]]* __private

// Test basic propagation
__attribute__((sycl_device)) void test_ref_binding(int i) { int &j = i; }

// CHECK: test_ref_binding : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int &j = i : int __private& __private

const int i = 0;

__attribute__((sycl_device)) void test_gv() { const int *j = &i; }

// CHECK: test_gv : () -> void
// CHECK-NEXT: const int *j = &i : int __constant* __private

__attribute__((sycl_device)) void test_double_assignment() {
  const int *j;
  const int *k = j = &i;
}

// CHECK: test_double_assignment : () -> void
// CHECK-NEXT: const int *j : int __constant* __private
// CHECK-NEXT: const int *k = j = &i : int __constant* __private

__attribute__((sycl_device)) void test_float_lit() { float j = 42.f; }

// CHECK: test_float_lit : () -> void
// CHECK-NEXT: float j = 42.F : float __private

const char *const gv_s = "hello";

__attribute__((sycl_device)) void test_string_lit() { const char *s = gv_s; }

// CHECK: test_string_lit : () -> void
// CHECK-NEXT: const char *s = gv_s : char __constant* __private
