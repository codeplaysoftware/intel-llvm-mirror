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
#include <sycl/sycl.hpp>
#include <iostream>

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

  if(dev.is_host() || q.is_host() || ctxt.is_host()){
    std::cout << "Test unsupported for non-cuda backends" << std::endl;
    exit(1);
  }
    
  const size_t texSize{99};

  std::vector<int> expectedValue(texSize, -1);
  float4* usmPtr{nullptr};
  _V1::ext::oneapi::image_handle imgHandle{0};

  try {
    float4 *usmPtr = malloc_host<float4>(texSize, q);
    for (size_t i{0}; i < texSize; ++i){
      float fidx = static_cast<float>(i);
      usmPtr[i] = float4{fidx, fidx, fidx, fidx};
    }
  } catch (...) {
    std::cerr << "USM allocation failed." << std::endl;
    assert(false);
  }

  try {
    _V1::ext::oneapi::image_descriptor imgDesc{range<1>{texSize}};
    auto imgHandle = _V1::ext::oneapi::create_image_handle(imgDesc, usmPtr, ctxt);
  } catch (...) {
    std::cerr << "Failed to create image handle." << std::endl;
    assert(false);
  }

  try {
    buffer<int> buf(expectedValue.data(), range<1>{texSize});
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.parallel_for<test_kernel>(
          nd_range<1>{texSize, texSize}, [=](nd_item<1> it) {
            size_t gId = it.get_global_linear_id();
            float idx = static_cast<float>(gId);
            float4 pixelVal = _V1::ext::oneapi::read<float4>(imgHandle, idx);
            acc[gId] = pixelVal[0] == idx && pixelVal[1] == idx && pixelVal[2] == idx && pixelVal[3] == idx;
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

  for (size_t i{0}; i < texSize; ++i) {
    if(expectedValue[i] != 1){
      std::cerr << "Incorrect value at i = " << i << std::endl;
      assert(false);
    }
  }
  return 0;
}
