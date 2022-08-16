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

using namespace sycl;
class test_kernel;

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  const size_t texSize{99};

  std::vector<int> expectedValue(texSize, -1);

  float4 *usmPtr = malloc_host<float4>(texSize, q);
  
  // TODO: Set values of usm memory.
  ext::oneapi::image_descriptor imgDesc(range<1>{texSize});
  auto imgHandle = ext::oneapi::create_image_handle(imgDesc, usmPtr, ctxt);

  {
    buffer<int> buf(expectedValue.data(), range<1>{texSize});
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh);
      cgh.parallel_for<test_kernel>(
          nd_range<1>{texSize, texSize}, [=](item<1> it) {
            size_t gId = it.get_global_linear_id();
            float idx = static_cast<float>(gId);
            float4 pixelVal = ext::oneapi::read_image_handle(imgHandle, idx);
            acc[gId] = pixelVal == float4{idx, idx, idx, idx};
          });
    });
  }
  ext::oneapi::destroy_image_handle(imgHandle, ctxt);
  free(usmPtr, ctxt);
  for (size_t i{0}; i < texSize; ++i) {
    assert(expectedValue == 1);
  }
}
