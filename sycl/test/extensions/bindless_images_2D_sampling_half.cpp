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
  size_t width = 5;
  size_t height = 6;
  size_t N = width * height;
  std::vector<half> out(N);
  std::vector<half> expected(N);
  std::vector<half4> dataIn1(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[j + (height * i)] = j + (height * i);
      dataIn1[j + (height * i)] = {j + (height * i), 0, 0, 0};
    }
  }

  sampler samp1(coordinate_normalization_mode::normalized,
                addressing_mode::repeat, filtering_mode::linear);

  unsigned int element_size_bytes = sizeof(half) * 4;
  size_t width_in_bytes = width * element_size_bytes;
  size_t pitch = 0;
  // Extension: returns the device pointer to USM allocated pitched memory
  auto img_mem_0 = sycl::ext::oneapi::pitched_alloc_device(
      &pitch, width_in_bytes, height, element_size_bytes, q);

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::image_descriptor desc({width, height},
                                           image_channel_order::rgba,
                                           image_channel_type::fp16, pitch);

  if (img_mem_0 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  for (int i = 0; i < height; ++i) {
    q.memcpy((char *)img_mem_0 + (pitch * i), &dataIn1[width * i],
             width_in_bytes);
  }

  // Extension: create the image and return the handle
  sycl::ext::oneapi::sampled_image_handle imgHandle1 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_0, samp1, desc);

  try {
    // Cuda stores data in column-major fashion
    // SYCL deals with indexing in row-major fashion
    // Reverse output buffer dimensions and access to convert
    // the cuda column-major data back to row-major
    buffer<half, 2> buf((half *)out.data(), range<2>{height, width});
    q.submit([&](handler &cgh) {
      auto outAcc =
          buf.get_access<access_mode::write>(cgh, range<2>{height, width});

      cgh.parallel_for<image_addition>(
          nd_range<2>{{width, height}, {width, height}}, [=](nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5) / (float)width;
            float fdim1 = float(dim1 + 0.5) / (float)height;

            // Extension: read image data from handle
            half4 px1 = sycl::ext::oneapi::read_image<half4>(
                imgHandle1, float2(fdim0, fdim1));

            outAcc[id<2>{dim1, dim0}] = px1[0];
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
    sycl::free(img_mem_0, ctxt);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }

  // collect and validate output
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
