// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s

#include "Inputs/sycl.hpp"

struct Foo {
  const int *a;
  const int *b;
};

// Test basic propagation
__attribute__((sycl_device)) void test(int i) {
  int *j = &i;
  Foo f{j, j};
}

// CHECK: test : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j = &i : int __private* __private
// CHECK-NEXT: Foo f{j, j} : {int __private*, int __private*} __private

const int gv_i = 0;

// Test basic propagation
__attribute__((sycl_device)) void test2(int i) {
  int *j = &i;
  Foo f{j, &gv_i};
  Foo g{&gv_i, j};
}

// CHECK: test2 : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j = &i : int __private* __private
// CHECK-NEXT: Foo f{j, &gv_i} : {int __private*, int __constant*} __private
// CHECK-NEXT: Foo g{&gv_i, j} : {int __constant*, int __private*} __private

struct Foo2 {
  const int *a;
  const int *b;
  Foo2 &operator=(const Foo2 &v) {
    a = v.a;
    b = v.b;
    return *this;
  }
};

__attribute__((sycl_device)) void test3(int i) {
  int *j = &i;
  Foo2 f{j, &gv_i};
  Foo2 g;
  g = {f};
}

// CHECK: test3 : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j = &i : int __private* __private
// CHECK-NEXT: Foo2 f{j, &gv_i} : {int __private*, int __constant*} __private
// CHECK-NEXT: Foo2 g : {int __private*, int __constant*} __private
