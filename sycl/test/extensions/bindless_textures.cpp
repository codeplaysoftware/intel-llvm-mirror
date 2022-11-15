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

#if SYCL_BACKEND_OPENCL == 1
  std::cout << "Backend is opencl!" << std::endl;
#endif

#if SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO == 3
  std::cout << "Backend is level zero!" << std::endl;
#endif

#if SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO != 3
  std::cerr << "Test unsupported for non level zero backends" << std::endl;
  exit(1);
#endif



  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  std::cout << "Running on "
        << q.get_device().get_info<sycl::info::device::name>()
        << "\n";

  if(dev.is_host() || q.is_host() || ctxt.is_host()){
    std::cout << "Test unsupported for non level zero backends" << std::endl;
    exit(1);
  }
    
  const size_t texSize{99};

  std::vector<int> expectedValue(texSize, -1);
  float4* usmPtr{nullptr};
  _V1::ext::oneapi::image_handle imgHandle{0};

  try {
    std::cout << "Allocating USM\n" << std::endl;
    float4 *usmPtr = malloc_host<float4>(texSize, q);
    std::cout << "Created USM Pointer\n";
    for (size_t i{0}; i < texSize; ++i){
      float fidx = static_cast<float>(i);
      usmPtr[i] = float4{fidx, fidx, fidx, fidx};
    }
    std::cout << "Allocated USM\n";
  } catch (...) {
    std::cerr << "USM allocation failed." << std::endl;
    assert(false);
  }

  try {
    _V1::ext::oneapi::image_descriptor imgDesc{range<1>{texSize}};
    imgHandle = _V1::ext::oneapi::create_image_handle(imgDesc, usmPtr, ctxt);
    std::cout << "Created image handle: " << imgHandle <<"\n";
  } catch (...) {
    std::cerr << "Failed to create image handle." << std::endl;
    assert(false);
  }

  try {
    std::cout << "Setting up kernel\n";
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
    std::cout << "Running kernel\n";
    q.wait_and_throw();
    std::cout << "After wait and throw\n";
  } catch (const std::exception &exc) {
    std::cerr << "Kernel submission failed." << std::endl;
    std::cerr << exc.what();
    assert(false);
  } catch (...){
    std::cerr << "Other kernel submission failure." << std::endl;
    assert(false);
  }
  
  try {
    std::cout << "About to destroy handle\n";
    std::cout << "imgHandle: " << imgHandle << "\n";
    _V1::ext::oneapi::destroy_image_handle(imgHandle, ctxt);
    std::cout << "Destroyed handle\n";
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
/*
  {
    sycl::image<1> MyImage1d(sycl::image_channel_order::rgbx, sycl::image_channel_type::unorm_short_565, sycl::range<1>(3));
    sycl::queue Q;
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = MyImage1d.get_access<int, sycl::access::mode::read>(cgh);

      cgh.single_task<class image_accessor1dro>([=]() {
        Acc.use();
      });
    });
  }
  */


  return 0;
}
