// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s


#include "Inputs/sycl.hpp"

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &h) {
    sycl::accessor<int, 1, sycl::access::mode::read> accessorA;
    sycl::local_accessor<int, 1> accessorB;
    sycl::accessor<int, 1, sycl::access::mode::read> accessorC;

    h.parallel_for(sycl::range<1>{42}, [=](sycl::id<1> wiID) {
          accessorC[wiID[0]] = accessorA[wiID[0]] + accessorB[wiID[0]];
        });
  });
}
// CHECK{LITERAL}: main()::(anonymous class)::operator()(sycl::handler &)::(anonymous class)::operator() : {{{{int[]}, {int[]}, {int[]}}, int __global*, {int[]}, {int[]}, {int[]}}, {{{int[]}, {int[]}, {int[]}}, int __global*, {int[]}, {int[]}, {int[]}}, {{{{int[]}, {int[]}, {int[]}}, int __local*, {int[]}, {int[]}, {int[]}}, {{int[]}, {int[]}, {int[]}}}}
// CHECK-SAME: [[THIS_ASP:.*]]* -> ({int[]}) -> void
// CHECK-NEXT{LITERAL}: sycl::id<1> wiID : {int[]} __private
// CHECK-NEXT{LITERAL}: sycl::accessor<int, 1, sycl::access::mode::read> accessorC : {{{int[]}, {int[]}, {int[]}}, int __global*, {int[]}, {int[]}, {int[]}}
// CHECK-SAME: [[THIS_ASP]]
// CHECK-NEXT{LITERAL}: sycl::accessor<int, 1, sycl::access::mode::read> accessorA : {{{int[]}, {int[]}, {int[]}}, int __global*, {int[]}, {int[]}, {int[]}}
// CHECK-SAME: [[THIS_ASP]]
// CHECK-NEXT{LITERAL}: sycl::local_accessor<int, 1> accessorB : {{{{int[]}, {int[]}, {int[]}}, int __local*, {int[]}, {int[]}, {int[]}}, {{int[]}, {int[]}, {int[]}}}
// CHECK-SAME: [[THIS_ASP]]
// CHECK{LITERAL}: _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_2idILi1EEEE_ : (int __global*, {int[]}, {int[]}, {int[]}, int __global*, {int[]}, {int[]}, {int[]}, int __local*, {int[]}, {int[]}, {int[]}) -> void
// CHECK-NEXT{LITERAL}: __global int *_arg_accessorC : int __global* __private
// CHECK-NEXT{LITERAL}: sycl::range<1> _arg_accessorC : {int[]} __private
// CHECK-NEXT{LITERAL}: sycl::range<1> _arg_accessorC : {int[]} __private
// CHECK-NEXT{LITERAL}: sycl::id<1> _arg_accessorC : {int[]} __private
// CHECK-NEXT{LITERAL}: __global int *_arg_accessorA : int __global* __private
// CHECK-NEXT{LITERAL}: sycl::range<1> _arg_accessorA : {int[]} __private
// CHECK-NEXT{LITERAL}: sycl::range<1> _arg_accessorA : {int[]} __private
// CHECK-NEXT{LITERAL}: sycl::id<1> _arg_accessorA : {int[]} __private
// CHECK-NEXT{LITERAL}: __local int *_arg_accessorB : int __local* __private
// CHECK-NEXT{LITERAL}: sycl::range<1> _arg_accessorB : {int[]} __private
// CHECK-NEXT{LITERAL}: sycl::range<1> _arg_accessorB : {int[]} __private
// CHECK-NEXT{LITERAL}: sycl::id<1> _arg_accessorB : {int[]} __private
// CHECK-NEXT{LITERAL}: auto (__SYCLKernel)(sycl::id<1>) const = {, , } : {{{{int[]}, {int[]}, {int[]}}, int __global*, {int[]}, {int[]}, {int[]}}, {{{int[]}, {int[]}, {int[]}}, int __global*, {int[]}, {int[]}, {int[]}}, {{{{int[]}, {int[]}, {int[]}}, int __local*, {int[]}, {int[]}, {int[]}}, {{int[]}, {int[]}, {int[]}}}} __private
