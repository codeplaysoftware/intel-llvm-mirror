// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

//==------- bindless_textures.cpp - SYCL bindless textures test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
class test_kernel;

int main() {

#if SYCL_EXT_ONEAPI_BACKEND_CUDA != 1
  std::cerr << "Test unsupported for non-cuda backends" << std::endl;
  exit(1);
#endif

  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (dev.is_host() || q.is_host() || ctxt.is_host()) {
    std::cout << "Test unsupported for non-cuda backends" << std::endl;
    exit(1);
  }

  const size_t texWidth = 5;
  const size_t texHeight = 5;
  const size_t numPixels = texWidth * texHeight;
  const size_t numFloats =
      numPixels * 4; // Using float4 textures and float vectors

  std::array<float4, numPixels> dat{};
  for (int i = 0; i < numPixels; ++i) {
    for (int j = 0; j < 4; ++j) {
      dat[i][j] = (4 * i) + j;
    }
  }

  std::vector<float4> expectedValue(numPixels, {-1.f, -1.f, -1.f, -1.f});
  float4 *usmPtr{nullptr};
  _V1::ext::oneapi::image_handle imgHandle{0};

  try {
    usmPtr = malloc_device<float4>(numPixels, q);
  } catch (...) {
    std::cerr << "USM allocation failed." << std::endl;
    assert(false);
  }

  auto e = q.memcpy(usmPtr, dat.data(), sizeof(float) * 4 * numPixels);
  e.wait();

  try {
    _V1::ext::oneapi::image_descriptor imgDesc{{texWidth, texHeight}};
    imgHandle = _V1::ext::oneapi::create_image_handle(imgDesc, usmPtr, ctxt);
  } catch (...) {
    std::cerr << "Failed to create image handle." << std::endl;
    assert(false);
  }

  try {
    buffer<float4, 2> buf(expectedValue.data(), range<2>{texWidth, texHeight});
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh, range<2>{texWidth, texHeight});
      cgh.parallel_for<test_kernel>(
          nd_range<2>{{texWidth, texHeight}, {texWidth, texHeight}},
          [=](nd_item<2> it) {
            size_t x = it.get_local_id(0);
            size_t y = it.get_local_id(1);
            float4 pixelVal =
                _V1::ext::oneapi::read<float4>(imgHandle, int2(x, y));
            acc[id<2>{x, y}] = pixelVal;
            // acc[id<2>{x, y}] = float4{x, y, 42, 42};
          });
    });
    q.wait_and_throw();
  } catch (...) {
    std::cerr << "Kernel submission failed." << std::endl;
    assert(false);
  }

  try {
    _V1::ext::oneapi::destroy_image_handle(imgHandle, ctxt);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }
  try {
    free(usmPtr, ctxt);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }

  for (size_t i{0}; i < numPixels; ++i) {
    printf("pixel[%zu] = (%f, %f, %f, %f)\n", i, expectedValue[i][0],
           expectedValue[i][1], expectedValue[i][2], expectedValue[i][3]);
    for (int j = 0; j < 4; ++j)
      if (expectedValue[i][j] != dat[i][j]) {
        fprintf(stderr, "Incorrect value. Expected: %f Actual %f\n", dat[i][j],
                expectedValue[i][j]);
        // assert(false);
      }
  }
  return 0;
}
