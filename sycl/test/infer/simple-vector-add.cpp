// RUN: %clangxx -fsycl -fsycl-device-only -fsycl-check-address-space-infer -Xclang -fsycl-dump-inferred-address-space -c -o %t.bc %s | FileCheck %s

#include <sycl/sycl.hpp>

#include <array>
#include <iostream>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N> &VA, const std::array<T, N> &VB,
                 std::array<T, N> &VC) {
  sycl::queue deviceQueue;
  sycl::range<1> numOfItems{N};
  sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  deviceQueue.submit([&](sycl::handler &cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for(numOfItems, [=](sycl::id<1> wiID) {
      auto &pC = accessorC[wiID];
      auto &pA = accessorA[wiID];
      auto &pB = accessorB[wiID];

      pC = pA + pB;

// CHECK{LITERAL}: simple_vadd(const std::array<int, 4UL> &, const std::array<int, 4UL> &, std::array<int, 4UL> &)::(anonymous class)::operator()(sycl::handler &)::(anonymous class)::operator() : {{{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {int __global*}}, {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {int __global*}}, {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {int __global*}}}
// CHECK-SAME: {{.*}}*
// CHECK-SAME{LITERAL}: -> ({{unsigned long[]}}) -> void
// CHECK-NEXT{LITERAL}: sycl::id<1> wiID : {{unsigned long[]}} __private
// CHECK-NEXT{LITERAL}: auto accessorC = bufferC.template get_access<sycl_write>(cgh) : {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {int __global*}}
// CHECK-NEXT{LITERAL}: auto accessorA = bufferA.template get_access<sycl_read>(cgh) : {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {int __global*}}
// CHECK-NEXT{LITERAL}: auto accessorB = bufferB.template get_access<sycl_read>(cgh) : {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {int __global*}}
// CHECK-NEXT{LITERAL}: auto &pC = accessorC[wiID] : int __global& __private
// CHECK-NEXT{LITERAL}: auto &pA = accessorA[wiID] : int __global& __private
// CHECK-NEXT{LITERAL}: auto &pB = accessorB[wiID] : int __global& __private

// CHECK{LITERAL}: simple_vadd(const std::array<float, 4UL> &, const std::array<float, 4UL> &, std::array<float, 4UL> &)::(anonymous class)::operator()(sycl::handler &)::(anonymous class)::operator() : {{{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {float __global*}}, {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {float __global*}}, {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {float __global*}}}
// CHECK-SAME: {{.*}}*
// CHECK-SAME{LITERAL}: -> ({{unsigned long[]}}) -> void
// CHECK-NEXT{LITERAL}: sycl::id<1> wiID : {{unsigned long[]}} __private
// CHECK-NEXT{LITERAL}: auto accessorC = bufferC.template get_access<sycl_write>(cgh) : {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {float __global*}}
// CHECK-NEXT{LITERAL}: auto accessorA = bufferA.template get_access<sycl_read>(cgh) : {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {float __global*}}
// CHECK-NEXT{LITERAL}: auto accessorB = bufferB.template get_access<sycl_read>(cgh) : {{}, {}, {{{unsigned long[]}}, {{unsigned long[]}}, {{unsigned long[]}}}, {float __global*}}
// CHECK-NEXT{LITERAL}: auto &pC = accessorC[wiID] : float __global& __private
// CHECK-NEXT{LITERAL}: auto &pA = accessorA[wiID] : float __global& __private
// CHECK-NEXT{LITERAL}: auto &pB = accessorB[wiID] : float __global& __private
    });
  });
}

int main() {
  const size_t array_size = 4;
  std::array<sycl::cl_int, array_size> A = {{1, 2, 3, 4}}, B = {{1, 2, 3, 4}},
                                       C;
  std::array<sycl::cl_float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                         E = {{1.f, 2.f, 3.f, 4.f}}, F;
  simple_vadd(A, B, C);
  simple_vadd(D, E, F);
  for (unsigned int i = 0; i < array_size; i++) {
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
    if (F[i] != D[i] + E[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << F[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}
