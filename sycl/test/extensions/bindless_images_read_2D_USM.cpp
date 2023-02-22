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
  size_t width = 4;
  size_t height = 7;
  size_t N = width * height;
  std::vector<uint32_t> out(N);
  std::vector<uint32_t> expected(N);
  std::vector<uint32_t> dataIn0(N);
  std::vector<uint32_t> dataIn1(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[j + (height * i)] = 2 * (j + (height * i));
      dataIn0[j + (height * i)] = j + (height * i);
      dataIn1[j + (height * i)] = j + (height * i);
      out[j + (height * i)] = 0;
    }
  }

  unsigned int element_size_bytes = sizeof(uint32_t);
  size_t width_in_bytes = width * element_size_bytes;
  size_t pitch_in_bytes0 = 0;
  size_t pitch_in_bytes1 = 0;
  size_t pitch_in_bytes2 = 0;

  // USM allocation
  auto img_mem_0 = sycl::ext::oneapi::pitched_alloc_device(
      &pitch_in_bytes0, width_in_bytes, height, element_size_bytes, q);
  auto img_mem_1 = sycl::ext::oneapi::pitched_alloc_device(
      &pitch_in_bytes1, width_in_bytes, height, element_size_bytes, q);
  uint32_t *img_mem_out = (uint32_t *)sycl::ext::oneapi::pitched_alloc_device(
      &pitch_in_bytes2, width_in_bytes, height, element_size_bytes, q);

  // Image descriptor - USM
  sycl::ext::oneapi::image_descriptor desc0(
      {width, height}, image_channel_order::r,
      image_channel_type::unsigned_int32, pitch_in_bytes0);
  sycl::ext::oneapi::image_descriptor desc1(
      {width, height}, image_channel_order::r,
      image_channel_type::unsigned_int32, pitch_in_bytes1);
  sycl::ext::oneapi::image_descriptor desc2(
      {width, height}, image_channel_order::r,
      image_channel_type::unsigned_int32, pitch_in_bytes2);

  if (img_mem_0 == nullptr || img_mem_1 == nullptr || img_mem_out == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Copy over data via USM
  q.ext_image_memcpy(img_mem_0, dataIn0.data(), desc0);
  q.ext_image_memcpy(img_mem_1, dataIn1.data(), desc1);
  q.ext_image_memcpy(img_mem_out, out.data(), desc2);
  q.wait();

  // Extension: create the image and return the handle
  sycl::ext::oneapi::unsampled_image_handle imgHandle0 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_0, desc0);
  sycl::ext::oneapi::unsampled_image_handle imgHandle1 =
      sycl::ext::oneapi::create_image(ctxt, img_mem_1, desc1);

  try {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<image_addition>(
          nd_range<2>{{width, height}, {width, height}}, [=](nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            uint32_t sum = 0;

            // Extension: read image data from handle
            uint32_t px1 = sycl::ext::oneapi::read_image<uint32_t>(
                imgHandle0, int2(dim0, dim1));
            uint32_t px2 = sycl::ext::oneapi::read_image<uint32_t>(
                imgHandle1, int2(dim0, dim1));

            sum = px1 + px2;
            size_t index = dim0 + dim1 * pitch_in_bytes2 / sizeof(uint32_t);
            img_mem_out[index] = sum;
          });
    });

  } catch (sycl::exception e) {
    std::cerr << "Kernel submission failed! " << e.what() << std::endl;
    assert(false);
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  q.ext_image_memcpy(out.data(), img_mem_out, desc2);
  q.wait();

  // Cleanup
  try {
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandle0);
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandle1);
    sycl::free(img_mem_0, ctxt); // USM
    sycl::free(img_mem_1, ctxt); // USM
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
