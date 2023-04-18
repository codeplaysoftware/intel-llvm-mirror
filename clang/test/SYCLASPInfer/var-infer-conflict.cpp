// RUN: %clang_cc1 -triple spir64-unknown-unknown -Wno-return-stack-address -fsycl-is-device -fsycl-check-address-space-infer -fsycl-dump-inferred-address-space -fsyntax-only -verify %s -o %t

#include "Inputs/sycl.hpp"

const int i = 0;

__attribute__((sycl_device)) void test_gv(int j) {
  // expected-error@* {{Conflicting deduced address space 'address space '__private'' and 'address space '__constant''}}
  // expected-note@* {{Address space deduced to 'address space '__private'' from here}}
  // expected-note@* {{Address space deduced to 'address space '__constant'' from here}}
  const int *k = &i;
  k = &j;
}

__attribute__((sycl_device)) void test_gv2(int j) {
  // expected-error@* {{Conflicting deduced address space 'address space '__private'' and 'address space '__constant''}}
  // expected-note@* {{Address space deduced to 'address space '__private'' from here}}
  // expected-note@* {{Address space deduced to 'address space '__constant'' from here}}
  const int *k[2];
  k[0] = &i;
  k[1] = &j;
}
