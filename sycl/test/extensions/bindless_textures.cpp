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

  std::cerr << "USING BACKEND: " << q.get_backend() << "\n";

  if (dev.is_host() || q.is_host() || ctxt.is_host()) {
    std::cout << "Test unsupported for non-cuda backends" << std::endl;
    exit(1);
  }

  const size_t texSize{5};
  const size_t numFloats{texSize *
                         4}; // Using float4 textures and float vectors

  std::array<float, numFloats> dat{};
  for (int i = 0; i < numFloats; i++) {
    dat[i] = i;
  }

  std::vector<float> expectedValue(numFloats, -1.0);
  float4 *usmPtr{nullptr};
  _V1::ext::oneapi::image_handle imgHandle{0};

  try {
    usmPtr = malloc_device<float4>(texSize, q);
  } catch (...) {
    std::cerr << "USM allocation failed." << std::endl;
    assert(false);
  }

  auto e = q.memcpy(usmPtr, dat.data(), sizeof(float4) * texSize);
  e.wait();

  try {
    _V1::ext::oneapi::image_descriptor imgDesc{{texSize}};
    imgHandle = _V1::ext::oneapi::create_image_handle(imgDesc, usmPtr, ctxt);
  } catch (...) {
    std::cerr << "Failed to create image handle." << std::endl;
    assert(false);
  }

  try {
    buffer<float> buf(expectedValue.data(), range<1>{numFloats});
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.parallel_for<test_kernel>(
          nd_range<1>{texSize, texSize}, [=](nd_item<1> it) {
            size_t gId = it.get_global_linear_id();
            int idx = static_cast<int>(gId);
            float4 pixelVal = _V1::ext::oneapi::read<float4>(imgHandle, idx);
            acc[4 * gId] = pixelVal[0];
            acc[4 * gId + 1] = pixelVal[1];
            acc[4 * gId + 2] = pixelVal[2];
            acc[4 * gId + 3] = pixelVal[3];
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

  for (size_t i{0}; i < numFloats; ++i) {
    std::cout << " value at i = " << i << ": " << expectedValue[i] << std::endl;
    if (expectedValue[i] != dat[i]) {
      std::cerr << "Incorrect value at position " << i << std::endl;
      assert(false);
    }
  }
  return 0;
}
