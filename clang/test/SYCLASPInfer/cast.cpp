// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s

// type forward type cast
__attribute__((sycl_device)) void type_forward_cast(long l) {
  int *p = nullptr;
  int *p2 = (int *)l;
  long l1 = (long)p2;
  bool b = p2;
  (void)p2;
  int i = l;
  bool b2 = l;
  double f = l;
  int i2 = f;
  bool b3 = f;
  int i3 = b3;
  float f2 = f;
}

// CHECK: type_forward_cast : (long) -> void
// CHECK-NEXT: long l : long __private
// CHECK-NEXT: int *p = nullptr : int {{.*}}* __private
// CHECK-NEXT: int *p2 = (int *)l : int {{.*}}* __private
// CHECK-NEXT: long l1 = (long)p2 : long __private
// CHECK-NEXT: bool b = p2 : _Bool __private
// CHECK-NEXT: int i = l : int __private
// CHECK-NEXT: bool b2 = l : _Bool __private
// CHECK-NEXT: double f = l : double __private
// CHECK-NEXT: int i2 = f : int __private
// CHECK-NEXT: bool b3 = f : _Bool __private
// CHECK-NEXT: int i3 = b3 : int __private
// CHECK-NEXT: float f2 = f : float __private

namespace test1 {

struct Foo {
  const int *a;
  const int *b;
};

struct Bar : public Foo {};

// Test basic propagation
__attribute__((sycl_device)) void derive_to_base(int i) {
  Bar b{&i, &i};
  Foo *f = &b;
}

// CHECK: test1::derive_to_base : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT{LITERAL}: Bar b{&i, &i} : {{int __private*, int __private*}} __private
// CHECK-NEXT: Foo *f = &b : {int __private*, int __private*} __private* __private

// Test basic propagation
__attribute__((sycl_device)) void base_to_derive(Foo *f) {
  Bar *b = static_cast<Bar *>(f);
}

// CHECK: test1::base_to_derive : ({int [[ASP_A:.*]]*, int [[ASP_B:.*]]*} [[ASP_F:.*]]*) -> void
// CHECK-NEXT: Foo *f : {int [[ASP_A]]*, int [[ASP_B]]*} [[ASP_F]]* __private
// COM: matching {{int [[ASP_A]]*, int [[ASP_B]]*}} [[ASP_F]]* __private
// CHECK-NEXT: Bar *b = static_cast<Bar *>(f) : {{[{][{]}}int [[ASP_A]]*, int [[ASP_B]]*{{[}][}]}} [[ASP_F]]* __private

} // namespace test1

namespace test2 {

struct Foo {
  const int *a;
  const int *b;
};

struct Bar : public Foo {
  const int *other;
};

const int GV = 42;

// Test basic propagation
__attribute__((sycl_device)) void derive_to_base(int i) {
  Bar b{{&i, &i}, &GV};
  Foo *f = &b;
}

// CHECK: test2::derive_to_base : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT{LITERAL}: Bar b{{&i, &i}, &GV} : {{int __private*, int __private*}, int __constant*} __private
// CHECK-NEXT: Foo *f = &b : {int __private*, int __private*} __private* __private

// Test basic propagation
__attribute__((sycl_device)) void base_to_derive(Foo *f) {
  Bar *b = static_cast<Bar *>(f);
}

// CHECK: test2::base_to_derive : ({int [[ASP_A:.*]]*, int [[ASP_B:.*]]*} [[ASP_F:.*]]*) -> void
// CHECK-NEXT: Foo *f : {int [[ASP_A]]*, int [[ASP_B]]*} [[ASP_F]]* __private
// CHECK-NEXT: Bar *b = static_cast<Bar *>(f) : {{[{][{]}}int [[ASP_A]]*, int [[ASP_B]]*}, int {{.*}}*} [[ASP_F]]* __private

} // namespace test2
