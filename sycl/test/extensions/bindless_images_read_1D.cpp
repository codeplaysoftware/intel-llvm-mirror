// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
class image_addition;

int main() {

  device dev;
  queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  // we use float4s but only take the first element
  // data, sampling, calculation
  //   A: 0,1,2,3,... B: 512, 511, 510, 509,...        -- data in
  //   C: 512,512,512,512,...                          -- sum (A+B)
  constexpr size_t N = 512;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N);
  float exp = 512;
  for (int i = 0; i < N; i++) {
    expected[i] = exp;
    dataIn1[i] = float4(i, i, i, i);
    dataIn2[i] = float4(N - i, N - i, N - i, N - i);
  }

  size_t width = N;

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::image_descriptor desc({width}, image_channel_order::rgba,
                                           image_channel_type::fp32);

  // Extension: returns the device pointer to the allocated memory
  auto img_mem_0 = sycl::ext::oneapi::allocate_image(ctxt, desc);
  auto img_mem_1 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (img_mem_0.value == nullptr || img_mem_1.value == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: create the image and return the handle
  sycl::ext::oneapi::unsampled_image_handle imgHandle1 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_0, desc);
  sycl::ext::oneapi::unsampled_image_handle imgHandle2 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_1, desc);

  // Extension: copy over data to device
  q.ext_image_memcpy(img_mem_0, dataIn1.data(), desc);
  q.ext_image_memcpy(img_mem_1, dataIn2.data(), desc);

  q.wait();

  try {
    buffer<float, 1> buf((float *)out.data(), N);
    q.submit([&](handler &cgh) {
      auto outAcc = buf.get_access<access_mode::write>(cgh, N);

      cgh.parallel_for<image_addition>(N, [=](id<1> id) {
        float sum = 0;
        // Extension: read image data from handle
        float4 px1 =
            sycl::ext::oneapi::read_image<float4>(imgHandle1, int(id[0]));
        float4 px2 =
            sycl::ext::oneapi::read_image<float4>(imgHandle2, int(id[0]));

        sum = px1[0] + px2[0];
        outAcc[id] = sum;
      });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  q.wait();

  // Cleanup
  try {
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandle1);
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandle2);
    sycl::ext::oneapi::free_image(ctxt, img_mem_0);
    sycl::ext::oneapi::free_image(ctxt, img_mem_1);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }

  // collect and validate output
  // we use float4s but only take the first element
  // data, sampling, calculation
  //   A: 0,1,2,3,... B: 512, 511, 510, 509,...        -- data in
  //   C: 512,512,512,512,...                          -- sum (A+B)
  bool validated = true;
  for (int i = 0; i < N; i++) {
    bool mismatch = false;
    if (out[i] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i] << std::endl;
    }
  }
  if (validated) {
    std::cout << "Correct output!" << std::endl;
  }

  return 0;
}
