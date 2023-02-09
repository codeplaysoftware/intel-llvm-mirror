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
  size_t width = 7;
  size_t height = 3;
  size_t N = width * height;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[j + (height * i)] = j * 3;
      dataIn1[j + (height * i)] = {j, j, j, j};
      dataIn2[j + (height * i)] = {j * 2, j * 2, j * 2, j * 2};
    }
  }

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::image_descriptor desc(
      {width, height}, image_channel_order::rgba, image_channel_type::fp32);

  // Extension: returns the device pointer to the allocated memory
  auto device_ptr1 = sycl::ext::oneapi::allocate_image(ctxt, desc);
  auto device_ptr2 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (device_ptr1 == nullptr || device_ptr2 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  sycl::ext::oneapi::copy_image(q, device_ptr1, dataIn1.data(), desc,
                                sycl::ext::oneapi::image_copy_flags::HtoD);
  sycl::ext::oneapi::copy_image(q, device_ptr2, dataIn2.data(), desc,
                                sycl::ext::oneapi::image_copy_flags::HtoD);

  // Extension: create the image and return the handle
  sycl::ext::oneapi::unsampled_image_handle imgHandle1 =
      sycl::ext::oneapi::create_image(ctxt, device_ptr1, desc);
  sycl::ext::oneapi::unsampled_image_handle imgHandle2 =
      sycl::ext::oneapi::create_image(ctxt, device_ptr2, desc);

  try {
    // Cuda stores data in column-major fashion
    // SYCL deals with indexing in row-major fashion
    // Reverse output buffer dimensions and access to convert
    // the cuda column-major data back to row-major
    buffer<float, 2> buf((float *)out.data(), range<2>{height, width});
    q.submit([&](handler &cgh) {
      auto outAcc =
          buf.get_access<access_mode::write>(cgh, range<2>{height, width});

      cgh.parallel_for<image_addition>(
          nd_range<2>{{width, height}, {width, height}}, [=](nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            float sum = 0;
            // Extension: read image data from handle
            float4 px1 = sycl::ext::oneapi::read_image<float4>(
                imgHandle1, int2(dim0, dim1));
            float4 px2 = sycl::ext::oneapi::read_image<float4>(
                imgHandle2, int2(dim0, dim1));

            sum = px1[0] + px2[0];
            outAcc[id<2>{dim1, dim0}] = sum;
          });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  // Cleanup
  try {
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandle1);
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandle2);
    sycl::ext::oneapi::free_image(ctxt, device_ptr1);
    sycl::ext::oneapi::free_image(ctxt, device_ptr2);
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
