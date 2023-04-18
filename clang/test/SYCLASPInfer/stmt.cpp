// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -emit-llvm %s -o %t 2>&1 | FileCheck %s

__attribute__((sycl_device)) void if_no_else(bool t, int i) {
  int *j;
  if (t) {
    j = &i;
  }
}
// CHECK: if_no_else : (_Bool, int) -> void
// CHECK-NEXT: bool t : _Bool __private
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j : int __private* __private

__attribute__((sycl_device)) void if_else(bool t, int i) {
  int *j;
  int *k;
  if (t) {
    j = &i;
  } else {
    k = &i;
  }
}
// CHECK: if_else : (_Bool, int) -> void
// CHECK-NEXT: bool t : _Bool __private
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j : int __private* __private
// CHECK-NEXT: int *k : int __private* __private

__attribute__((sycl_device)) void for_(int i) {
  int *j;
  for (;;) {
    j = &i;
  }
}
// CHECK: for_ : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j : int __private* __private

__attribute__((sycl_device)) void for_2(int i) {
  int *j;
  for (j = &i;;) {
  }
}
// CHECK: for_2 : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j : int __private* __private

__attribute__((sycl_device)) void for_3(int i) {
  for (int *j = &i;;) {
  }
}
// CHECK: for_3 : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j = &i : int __private* __private

__attribute__((sycl_device)) void for_4(int i) {
  int *j;
  for (; (j = &i);) {
  }
}
// CHECK: for_4 : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j : int __private* __private

__attribute__((sycl_device)) void for_5(int i) {
  int *j;
  for (;; j = &i) {
  }
}
// CHECK: for_5 : (int) -> void
// CHECK-NEXT: int i : int __private
// CHECK-NEXT: int *j : int __private* __private
