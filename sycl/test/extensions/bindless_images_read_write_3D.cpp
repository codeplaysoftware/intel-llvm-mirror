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
  size_t height = 13;
  size_t width = 7;
  size_t depth = 11;
  size_t N = height * width * depth;
  std::vector<float4> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N);
  for (int i = 0; i < height; i++) {  // row
    for (int j = 0; j < width; j++) { // column
      for (int k = 0; k < depth; k++) // depth
      {
        expected[i + width * (j + depth * k)] = j * 3;
        dataIn1[i + width * (j + depth * k)] = {j, j, j, j};
        dataIn2[i + width * (j + depth * k)] = {j * 2, j * 2, j * 2, j * 2};
      }
    }
  }

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::image_descriptor desc({width, height, depth},
                                           image_channel_order::rgba,
                                           image_channel_type::fp32);

  // Extension: returns the device pointer to the allocated memory
  // Input images memory
  auto img_mem_0 = sycl::ext::oneapi::allocate_image(ctxt, desc);
  auto img_mem_1 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  // Output image memory
  auto img_mem_2 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (img_mem_0.value == nullptr || img_mem_1.value == nullptr ||
      img_mem_2.value == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  q.ext_image_memcpy(img_mem_0, dataIn1.data(), desc);
  q.ext_image_memcpy(img_mem_1, dataIn2.data(), desc);
  q.wait();

  // Extension: create the image and return the handle
  sycl::ext::oneapi::unsampled_image_handle imgIn1 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_0, desc);
  sycl::ext::oneapi::unsampled_image_handle imgIn2 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_1, desc);

  sycl::ext::oneapi::unsampled_image_handle imgOut =
      sycl::ext::oneapi::create_image(ctxt, img_mem_2, desc);

  try {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<image_addition>(
          nd_range<3>{{width, height, depth}, {width, height, depth}},
          [=](nd_item<3> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            size_t dim2 = it.get_local_id(2);
            float sum = 0;
            // Extension: read image data from handle
            float4 px1 = sycl::ext::oneapi::read_image<float4>(
                imgIn1, int4(dim0, dim1, dim2, 0));
            float4 px2 = sycl::ext::oneapi::read_image<float4>(
                imgIn2, int4(dim0, dim1, dim2, 0));

            sum = px1[0] + px2[0];
            // Extension: write to image with handle
            sycl::ext::oneapi::write_image<float4>(
                imgOut, int4(dim0, dim1, dim2, 0), float4(sum));
          });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  q.wait();
  q.ext_image_memcpy(out.data(), img_mem_2, desc);
  q.wait();

  // Cleanup
  try {
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgIn1);
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgIn2);
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgOut);
    sycl::ext::oneapi::free_image(ctxt, img_mem_0);
    sycl::ext::oneapi::free_image(ctxt, img_mem_1);
    sycl::ext::oneapi::free_image(ctxt, img_mem_2);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }

  // collect and validate output
  // we use float4s but only take the first element
  bool validated = true;
  for (int i = 0; i < N; i++) {
    bool mismatch = false;
    if (out[i][0] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i][0] << std::endl;
    }
  }
  if (validated) {
    std::cout << "Correct output!" << std::endl;
  }

  return 0;
}
