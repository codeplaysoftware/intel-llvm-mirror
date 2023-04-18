// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s

#include "Inputs/sycl.hpp"

__attribute__((sycl_device)) const int *id(const int *i) { return i; }
// CHECK: id : (int [[SLOT:0x.*]]*) -> int [[SLOT]]*
// CHECK: const int *i : int [[SLOT]]* __private

__attribute__((sycl_device)) void basic_call(int i) { const int *j = id(&i); }
// CHECK: basic_call : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: const int *j = id(&i) : int __private* __private

__attribute__((sycl_device)) void deduced_by_return_call(int i) {
  const int *j = &i;
  const int *k;
  j = id(k);
}

// CHECK: deduced_by_return_call : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: const int *j = &i : int __private* __private
// CHECK-NEXT: const int *k : int __private* __private

struct Foo {
  const int *i;
  void set(const int *j) { i = j; }
};

const int f = 42;

// CHECK: Foo : {int {{0x.*}}*} {{0x.*}}* -> () -> void
// CHECK: set : {int [[SLOT:0x.*]]*} {{0x.*}}* -> (int [[SLOT]]*) -> void
// CHECK-NEXT: int *j : int [[SLOT]]* __private

__attribute__((sycl_device)) void deduced_method(int i) {
  Foo o1;
  o1.set(&i);
  Foo o2;
  (&o2)->set(&f);
}

// CHECK: deduced_method : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: Foo o1 : {int __private*} __private
// CHECK-NEXT: Foo o2 : {int __constant*} __private

__attribute__((sycl_device)) const int *default_arg(const int *i = &f) {
  return i;
}

// CHECK: default_arg : (int [[SLOT:0x.*]]*) -> int [[SLOT]]*
// CHECK-NEXT: const int *i = &f : int [[SLOT]]* __private

__attribute__((sycl_device)) void call_default_arg() {
  int i = 42;
  const int *j = default_arg();
  const int *k = default_arg(&i);
}

// CHECK: call_default_arg : () -> void
// CHECK-NEXT: int i = 42 : int __private
// CHECK-NEXT: const int *j = default_arg() : int __constant* __private
// CHECK-NEXT: const int *k = default_arg(&i) : int __private* __private
