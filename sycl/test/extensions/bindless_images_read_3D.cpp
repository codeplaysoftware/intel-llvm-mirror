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
  size_t width = 1024;
  size_t height = 1024;
  size_t depth = 16;
  size_t N = width * height * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < depth; k++) {
        expected[k + depth * (j + height * i)] = j * 3;
        dataIn1[k + depth * (j + height * i)] = {j, j, j, j};
        dataIn2[k + depth * (j + height * i)] = {j * 2, j * 2, j * 2, j * 2};
      }
    }
  }

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::image_descriptor desc({width, height, depth},
                                           image_channel_order::rgba,
                                           image_channel_type::fp32, 0);

  // Extension: returns the device pointer to the allocated memory
  auto img_mem_0 = sycl::ext::oneapi::allocate_image(ctxt, desc);
  auto img_mem_1 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (img_mem_0.value == nullptr || img_mem_1.value == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  q.ext_image_memcpy(img_mem_0, dataIn1.data(), desc);
  q.ext_image_memcpy(img_mem_1, dataIn2.data(), desc);
  q.wait();

  // Extension: create the image and return the handle
  sycl::ext::oneapi::unsampled_image_handle imgHandle1 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_0, desc);
  sycl::ext::oneapi::unsampled_image_handle imgHandle2 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_1, desc);

  try {
    // Cuda stores data in column-major fashion
    // SYCL deals with indexing in row-major fashion
    // Reverse output buffer dimensions and access to convert
    // the cuda column-major data back to row-major
    buffer<float, 3> buf((float *)out.data(), range<3>{depth, height, width});
    q.submit([&](handler &cgh) {
      auto outAcc = buf.get_access<access_mode::write>(
          cgh, range<3>{depth, height, width});

      cgh.parallel_for<image_addition>(
          nd_range<3>{{width, height, depth}, {16, 16, 4}},
          [=](nd_item<3> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);
            size_t dim2 = it.get_global_id(2);
            float sum = 0;
            // Extension: read image data from handle
            float4 px1 = sycl::ext::oneapi::read_image<float4>(
                imgHandle1, int4(dim0, dim1, dim2, 0));
            float4 px2 = sycl::ext::oneapi::read_image<float4>(
                imgHandle2, int4(dim0, dim1, dim2, 0));

            sum = px1[0] + px2[0];
            outAcc[id<3>{dim2, dim1, dim0}] = sum;
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
