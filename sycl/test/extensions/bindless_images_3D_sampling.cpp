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
  size_t width = 4;
  size_t height = 6;
  size_t depth = 8;
  size_t N = width * height * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < depth; k++) {
        expected[k + depth * (j + height * i)] = k + depth * (j + height * i);
        dataIn1[k + depth * (j + height * i)] = {k + depth * (j + height * i),
                                                 0, 0, 0};
      }
    }
  }

  // Image descriptor
  sycl::ext::oneapi::image_descriptor desc({width, height, depth},
                                           image_channel_order::rgba,
                                           image_channel_type::fp32);

  sampler samp1(coordinate_normalization_mode::normalized,
                addressing_mode::clamp, filtering_mode::linear);

  // Extension: returns the device pointer to the allocated memory
  auto img_mem_0 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (img_mem_0.value == nullptr) {
    std::cout << "Error allocating image!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  q.ext_image_memcpy(img_mem_0, dataIn1.data(), desc);
  q.wait();

  // Extension: create the image and return the handle
  sycl::ext::oneapi::sampled_image_handle imgHandle1 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_0, samp1, desc);

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
          nd_range<3>{{width, height, depth}, {width, height, depth}},
          [=](nd_item<3> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            size_t dim2 = it.get_local_id(2);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5) / (float)width;
            float fdim1 = float(dim1 + 0.5) / (float)height;
            float fdim2 = float(dim2 + 0.5) / (float)depth;

            // Extension: read image data from handle
            float4 px1 = sycl::ext::oneapi::read_image<float4>(
                imgHandle1, float4(fdim0, fdim1, fdim2, (float)0));

            outAcc[id<3>{dim2, dim1, dim0}] = px1[0];
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
    sycl::ext::oneapi::free_image(ctxt, img_mem_0);
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
