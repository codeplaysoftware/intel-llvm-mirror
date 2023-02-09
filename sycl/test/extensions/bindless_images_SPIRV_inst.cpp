// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

// RUN: %clangxx -fsycl -fsycl-targets=spir64-unknown-unknown %s -fsycl-device-only -o %t.out
// RUN: llvm-spirv %t.out -spirv-text -spirv-ext=+SPV_NV_bindless_texture -o %t.out
// RUN: FileCheck %s --input-file %t.out --check-prefix=CHECK-SPIRV

// CHECK-SPIRV: Capability BindlessTextureNV
// CHECK-SPIRV: Decorate 12 BindlessImageNV
// CHECK-SPIRV: Decorate 16 BindlessSamplerNV
// CHECK-SPIRV: Decorate 20 BindlessImageNV


// CHECK-SPIRV: Decorate 23 BindlessImageNV
// CHECK-SPIRV: Decorate 31 BindlessImageNV
// CHECK-SPIRV: Decorate 45 BindlessImageNV

// CHECK-SPIRV: Decorate 50 BindlessImageNV
// CHECK-SPIRV: Decorate 56 BindlessImageNV
// CHECK-SPIRV: Decorate 67 BindlessImageNV


// CHECK-SPIRV: Constant 9 10 234343 0
// CHECK-SPIRV: Constant 9 14 484670474 10
// CHECK-SPIRV: Constant 9 18 433334553 0

// CHECK-SPIRV: Constant 9 22 1234 0
// CHECK-SPIRV: Constant 9 29 4321 0
// CHECK-SPIRV: Constant 9 43 1357 0

// CHECK-SPIRV: Constant 9 49 6789 0
// CHECK-SPIRV: Constant 9 53 9876 0
// CHECK-SPIRV: Constant 9 65 9753 0


// CHECK-SPIRV: TypeImage 11 2 0 0 0 0 0 0 0
// CHECK-SPIRV: TypeSampler 15
// CHECK-SPIRV: TypeSampledImage 19 11

// CHECK-SPIRV: TypeImage 30 2 1 0 0 0 0 0 0
// CHECK-SPIRV: TypeImage 44 2 2 0 0 0 0 0 1
// CHECK-SPIRV: TypeImage 54 2 2 0 0 0 0 0 0
// CHECK-SPIRV: TypeSampledImage 55 54
// CHECK-SPIRV: TypeImage 66 2 1 0 0 0 0 0 1


// CHECK-SPIRV: ConvertUToImageNV 11 12 10
// CHECK-SPIRV: ConvertImageToUNV 9 13 12

// CHECK-SPIRV: ConvertUToSamplerNV 15 16 14
// CHECK-SPIRV: ConvertSamplerToUNV 9 17 16

// CHECK-SPIRV: ConvertUToSampledImageNV 19 20 18
// CHECK-SPIRV: ConvertSampledImageToUNV 9 21 20


// CHECK-SPIRV: ConvertUToImageNV 11 23 22
// CHECK-SPIRV: ImageRead 25 28 23 27

// CHECK-SPIRV: ConvertUToImageNV 30 31 29
// CHECK-SPIRV: ImageRead 25 36 31 35 

// CHECK-SPIRV: ConvertUToImageNV 44 45 43
// CHECK-SPIRV: ImageWrite 45 48 42

// CHECK-SPIRV: ConvertUToSampledImageNV 19 50 49
// CHECK-SPIRV: ImageSampleExplicitLod 25 52 50 27 2 51

// CHECK-SPIRV: ConvertUToSampledImageNV 55 56 53
// CHECK-SPIRV: ImageSampleExplicitLod 25 59 56 58 2 51

// CHECK-SPIRV: ConvertUToImageNV 66 67 65
// CHECK-SPIRV: ImageWrite 67 35 64


#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/detail/image_ocl_types.hpp>

#ifdef __SYCL_DEVICE_ONLY__
using OCLImageTyRead =
    typename sycl::detail::opencl_image_type<1, sycl::access::mode::read,
                                             sycl::access::target::image>::type;
using OCLImageTyWrite =
    typename sycl::detail::opencl_image_type<1, sycl::access::mode::write,
                                             sycl::access::target::image>::type;
using OCLSampledImageTy =
    typename sycl::detail::sampled_opencl_image_type<OCLImageTyRead>::type;
#endif

// Checking that SPIR-V image/sampler convering works.
template <typename DataT>
DataT handleToImageToHandle(const unsigned long &imageHandle) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
#else
  return __spirv_ConvertImageToUNV<OCLImageTyRead>(
      __spirv_ConvertUToImageNV<OCLImageTyRead>(imageHandle));
#endif
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

template <typename DataT>
DataT handleToSamplerToHandle(const unsigned long &samplerHandle) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
#else
  return __spirv_ConvertSamplerToUNV<__ocl_sampler_t>(
      __spirv_ConvertUToSamplerNV<__ocl_sampler_t>(samplerHandle));
#endif
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

template <typename DataT>
DataT handleToSampImageToHandle(const unsigned long &sampImageHandle) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
#else
  return __spirv_ConvertSampledImageToUNV<OCLSampledImageTy>(
      __spirv_ConvertUToSampledImageNV<OCLSampledImageTy>(sampImageHandle));
#endif
#else
  assert(false); // Bindless images not yet implemented on host.
#endif
}

using namespace sycl;
class test_kernel;

int main() {

#if SYCL_EXT_ONEAPI_BACKEND_CUDA != 1
  std::cerr << "Test unsupported for non-cuda backends" << std::endl;
  exit(1);
#endif

  device dev;
  queue q(dev);
  auto ctxt = q.get_context();

  try {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<test_kernel>(nd_range<1>{10, 10}, [=](nd_item<1> id) {
        // Test converting handle to image back to handle
        unsigned long test = handleToImageToHandle<unsigned long>(234343);

        // Test converting handle to sampler back to handle
        unsigned long test2 =
            handleToSamplerToHandle<unsigned long>(43434343434);

        // Test converting handle to sampled image back to handle
        unsigned long test3 =
            handleToSampImageToHandle<unsigned long>(433334553);

        // Extension: read image data from handle
        const sycl::ext::oneapi::unsampled_image_handle imgIn1{1234};
        const sycl::ext::oneapi::unsampled_image_handle imgIn2{4321};
        const sycl::ext::oneapi::unsampled_image_handle imgOut{1357};
        float4 px1 = sycl::ext::oneapi::read_image<float4>(imgIn1, 0);
        float4 px2 = sycl::ext::oneapi::read_image<float4>(imgIn2, int2(3, 5));

        float sum = 0;
        sum = px1[0] + px2[0];

        // Extension: write to image with handle
        sycl::ext::oneapi::write_image<float4>(imgOut, int4(3, 5, 7, 0),
                                               float4(sum));

        // Extension: read image data from handle
        const sycl::ext::oneapi::sampled_image_handle imgSmpIn1{6789};
        const sycl::ext::oneapi::sampled_image_handle imgSmpIn2{9876};
        const sycl::ext::oneapi::unsampled_image_handle imgSmpOut{9753};
        float4 px3 = sycl::ext::oneapi::read_image<float4>(imgSmpIn1, 0);
        float4 px4 =
            sycl::ext::oneapi::read_image<float4>(imgSmpIn2, int4(3, 5, 9, 0));

        float sum2 = 0;
        sum2 = px3[0] + px4[0];

        // Extension: write to image with handle
        sycl::ext::oneapi::write_image<float4>(imgSmpOut, int2(3, 5),
                                               float4(sum2));
      });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }
  return 0;
}
