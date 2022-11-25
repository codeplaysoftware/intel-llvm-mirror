
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

#include <cuda.h>
#include <cuda_device_runtime_api.h>

using namespace sycl;
class test_kernel;

// int roundUp(int numToRound, int multiple) {
//   if (multiple == 0 || (numToRound % multiple == 0)) {
//     return numToRound;
//   }

//   int roundDown = ((int)(numToRound) / multiple) * multiple;
//   int roundUp = roundDown + multiple;
//   int roundCalc = roundUp;
//   return (roundCalc);
// }

int main() {

#if SYCL_EXT_ONEAPI_BACKEND_CUDA != 1
  std::cerr << "Test unsupported for non-cuda backends" << std::endl;
  exit(1);
#endif

  device dev;
  gpu_selector_v(dev);

  queue q(dev);
  std::cerr << "USING BACKEND: " << q.get_backend() << "\n";
  // auto dev = q.get_device();
  auto ctxt = q.get_context();

  if (dev.is_host() || q.is_host() || ctxt.is_host()) {
    std::cout << "Test unsupported for non-cuda backends" << std::endl;
    exit(1);
  }

  constexpr size_t texWidth = 4;
  constexpr size_t texHeight = 4;
  constexpr size_t numPixels = texWidth * texHeight;
  // const size_t pitch = roundUp(texWidth, 32 / 16);
  // const size_t pitchBytes = pitch * sizeof(float);
  int32_t pitchBytes;
  // const size_t allocSize = numPixels * 4 * sizeof(float);
  // constexpr size_t numFloats =
  // numPixels * 4; // Using float4 textures and float vectors

  float4 *textureDataUSM = nullptr;
  _V1::ext::oneapi::image_handle imgHandle{0};

  try {
    std::cerr << "ALLOCATING\n";
    // textureDataUSM = aligned_alloc_device<float4>(32, 0, q);
    textureDataUSM = aligned_alloc_device<float4>(32, texWidth * texHeight, q);
    // textureDataUSM = malloc_device<float4>(texWidth * texHeight, q);
    std::cerr << "ALLOCATED AT: " << textureDataUSM << "\n";
    // pitchBytes = 512;
    pitchBytes = 64;
  } catch (...) {
    std::cerr << "USM allocation failed." << std::endl;
    assert(false);
  }

  const size_t allocSize = pitchBytes * texHeight;

  std::vector<char> textureData(allocSize, 0);
  std::vector<char> actualOutput(allocSize);

  for (int i = 0; i < texHeight; ++i) {
    for (int j = 0; j < texWidth; ++j) {
      // float* elem = (float*)((char*)textureData.data() + i * pitchBytes) + j;
      // *elem = i + j;
      // float4 *elem = (float4 *)((char *)textureData.data() +
      //                           i * (pitchBytes / (4 * sizeof(float)))) + j;
      float4 *elem =
          (float4 *)((char *)textureData.data() + (i * pitchBytes)) + j;
      (*elem)[0] = i + j;
      (*elem)[1] = i + j;
      (*elem)[2] = i + j;
      (*elem)[3] = i + j;
      // *elem = {0, 0, 0, 0};
    }
  }

  // // for (int i = 0; i < texHeight; ++i) {
  // //   float4* rowDat = textureData.data() + pitch * i;
  // //   float4* outDat = actualOutput.data() + pitch * i;
  // //   for (int j = 0; j < texWidth; ++j) {
  // //     rowDat[j] = j + i;
  // //     outDat[j] = -1.f;
  // //   }
  // // }

  // for (int i = 0; i < texHeight; ++i) {
  //   for (int j = 0; j < pitchBytes / (4 * sizeof(float) * texWidth); ++j) {
  //     // for (int j = 0; j < texWidth; ++j) {
  //     // float elem = *(float*)((char*)textureData.data() + i * pitchBytes) +
  //     j;

  //     // float4 elem =
  //     //     *(float4 *)((char *)textureData.data() + (i * pitchBytes)) + j;

  //     // // elem[0] = 0.f;

  //     // if (j >= texWidth)
  //     //   continue;
  //     // // std::cout << "PADDING: ";

  //     // // std::cout << "elem[" << i << ", " << j << "] == " << elem <<
  //     "\n";

  //     // std::cout << "(" << elem[0] << ", ";
  //     // std::cout << elem[1] << ", ";
  //     // std::cout << elem[2] << ", ";
  //     // std::cout << elem[3] << "), ";

  //     // std::cout << "(" << textureData[i * pitch + j][0] << ", ";
  //     // std::cout << textureData[i * pitch + j][1] << ", ";
  //     // std::cout << textureData[i * pitch + j][2] << ", ";
  //     // std::cout << textureData[i * pitch + j][3] << "), ";
  //   }
  //   // std::cout << "\n";
  // }

  // CUDA_MEMCPY2D cpy_desc;
  // memset(&cpy_desc, 0, sizeof(cpy_desc));
  // cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
  // cpy_desc.srcHost = (void*)textureData.data();
  // cpy_desc.srcPitch = pitchBytes;
  // cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
  // cpy_desc.dstDevice = (CUdeviceptr)textureDataUSM;
  // cpy_desc.dstPitch = pitchBytes;

  // cpy_desc.WidthInBytes = texWidth * sizeof(float) * 4;
  // cpy_desc.Height = texHeight;
  
  // auto retErr = cuMemcpy2D(&cpy_desc);
  // assert(retErr == CUDA_SUCCESS);

  auto e = q.memcpy(textureDataUSM, textureData.data(), allocSize);
  e.wait();

  // return 0;

  // for (int i = 0; i < numPixels; ++i) {
  //   for (int j = 0; j < 4; ++j) {
  //     textureData[i][j] = (4 * i) + j;
  //     actualOutput[i][j] = -1.f;
  //   }
  // }

  try {
    _V1::ext::oneapi::image_descriptor imgDesc{{texWidth, texHeight},
                                               pitchBytes};
    imgHandle =
        _V1::ext::oneapi::create_image_handle(imgDesc, textureDataUSM, ctxt);
  } catch (...) {
    std::cerr << "Failed to create image handle." << std::endl;
    assert(false);
  }

  try {
    buffer<float4, 2> buf((float4 *)actualOutput.data(),
                          range<2>{texWidth, texHeight});
    q.submit([&](handler &cgh) {
      auto acc = buf.get_access(cgh, range<2>{texWidth, texHeight});
      cgh.parallel_for<test_kernel>(
          nd_range<2>{{texWidth, texHeight}, {texWidth, texHeight}},
          [=](nd_item<2> it) {
            // Column major access
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            int id0 = int(dim0);
            int id1 = int(dim1);
            float4 pixelVal =
                _V1::ext::oneapi::read<float4>(imgHandle, int2(id1, id0));
            // float4 pixelVal =
            //     _V1::ext::oneapi::read<float4>(imgHandle, id0);
            acc[id<2>{dim0, dim1}] = pixelVal;
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

  bool validated = true;

  for (int i = 0; i < texHeight; ++i) {
    for (int j = 0; j < texWidth; ++j) {
      float4 expected =
          *((float4 *)((char *)textureData.data() + (i * pitchBytes)) + j);
      float4 actual =
          *((float4 *)((char *)actualOutput.data() + (i * pitchBytes)) + j);

      bool mismatch = false;
      for (int k = 0; k < 4; ++k) {
        if (expected[j] != actual[k]) {
          mismatch = true;
          validated = false;
        }
      }
      if (mismatch) {
        printf("Result mismatch! expected[%zu, %zu] = (%f, %f, %f, %f). "
               "actual[%zu, %zu] = (%f, %f, %f, %f)\n",
               i, j, expected[0], expected[1], expected[2], expected[3], i, j,
               actual[0], actual[1], actual[2], actual[3]);
        mismatch = false;
        continue;
      }
    }
  }

  try {
    free(textureDataUSM, ctxt);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }

  // for (int i = 0; i < numPixels; ++i) {
  //   bool mismatch = false;
  //   for (int j = 0; j < 4; ++j) {
  //     if (actualOutput[i][j] != textureData[i][j]) {
  //       mismatch = true;
  //       validated = false;
  //     }
  //   }
  //   if (mismatch) {
  //     printf("Result mismatch! actualOutput[%zu] = (%f, %f, %f, %f). "
  //            "expectedOutput[%zu] = (%f, %f, %f, %f)\n",
  //            i, actualOutput[i][0], actualOutput[i][1], actualOutput[i][2],
  //            actualOutput[i][3], i, textureData[i][0], textureData[i][1],
  //            textureData[i][2], textureData[i][3]);
  //     mismatch = false;
  //     continue;
  //   }
  // }

  // try {
  //   free(textureDataUSM, ctxt);
  // } catch (...) {
  //   std::cerr << "Failed to destroy image handle." << std::endl;
  //   assert(false);
  // }

  if (validated) {
    printf("Correct output!\n");
    return 0;
  } else {
    printf("Incorrect output!\n");
    return 1;
  }

  return 1;
}
