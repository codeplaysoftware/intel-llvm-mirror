// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s

//#include "Inputs/sycl.hpp"

__attribute__((sycl_device)) const int *id(const int *i) {
  return [&]() { return i; }();
}

// CHECK: id(const int *)::(anonymous class)::operator() : {int [[SLOT_L:0x.*]]* __private&} [[SLOT_PARENT:0x.*]]* -> () -> int [[SLOT_L]]*
// CHECK: const int *i : int [[SLOT_L]]* __private& [[SLOT_PARENT]]
// CHECK: id : (int [[SLOT:0x.*]]*) -> int [[SLOT]]*
// CHECK: const int *i : int [[SLOT]]* __private

__attribute__((sycl_device)) const int *id2(const int *i) {
  auto l = [&]() { return i; };
  return l();
}

// CHECK: id2(const int *)::(anonymous class)::operator() : {int [[SLOT_L:0x.*]]* __private&} [[SLOT_PARENT:0x.*]]* -> () -> int [[SLOT_L]]*
// CHECK: const int *i : int [[SLOT_L]]* __private& [[SLOT_PARENT]]
// CHECK: id2 : (int [[SLOT:0x.*]]*) -> int [[SLOT]]*
// CHECK: const int *i : int [[SLOT]]* __private
// CHECK: auto l = [&]() {
// CHECK:     return i;
// CHECK: } : {int [[SLOT]]* __private&} __private
