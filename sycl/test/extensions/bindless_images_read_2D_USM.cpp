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
  std::vector<uint32_t> dataIn1(N);
  std::vector<uint32_t> dataIn2(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[j + (height * i)] = j * 3;
      dataIn1[j + (height * i)] = j * 2;
      dataIn2[j + (height * i)] = j;
    }
  }

  unsigned int element_size_bytes = sizeof(uint32_t);
  size_t width_in_bytes = width * element_size_bytes;
  size_t pitch_in_bytes1 = 0;

  // USM allocation
  auto device_ptr1 = sycl::ext::oneapi::pitched_alloc_device(
      &pitch_in_bytes1, width_in_bytes, height, element_size_bytes, q);

  // Image descriptor - USM
  sycl::ext::oneapi::image_descriptor descUSM(
      {width, height}, image_channel_order::r,
      image_channel_type::unsigned_int32, pitch_in_bytes1);
  // Image descriptor - non-USM
  sycl::ext::oneapi::image_descriptor desc({width, height},
                                           image_channel_order::r,
                                           image_channel_type::unsigned_int32);

  // non-USM allocation
  auto device_ptr2 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (device_ptr1 == nullptr || device_ptr2 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Copy over data via USM
  for (int i = 0; i < height; ++i) {
    q.memcpy((char *)device_ptr1 + (pitch_in_bytes1 * i), &dataIn1[width * i],
             width * sizeof(uint32_t));
  }
  // Copy over data via extension
  q.ext_image_memcpy(device_ptr2, dataIn2.data(), desc,
                     sycl::ext::oneapi::image_copy_flags::HtoD);
  q.wait();

  // Extension: create the image and return the handle
  sycl::ext::oneapi::unsampled_image_handle imgHandle1 =
      sycl::ext::oneapi::create_image(ctxt, device_ptr1, descUSM);
  sycl::ext::oneapi::unsampled_image_handle imgHandle2 =
      sycl::ext::oneapi::create_image(ctxt, device_ptr2, desc);

  try {
    // Cuda stores data in column-major fashion
    // SYCL deals with indexing in row-major fashion
    // Reverse output buffer dimensions and access to convert
    // the cuda column-major data back to row-major
    buffer<uint32_t, 2> buf((uint32_t *)out.data(), range<2>{height, width});
    q.submit([&](handler &cgh) {
      auto outAcc =
          buf.get_access<access_mode::write>(cgh, range<2>{height, width});

      cgh.parallel_for<image_addition>(
          nd_range<2>{{width, height}, {width, height}}, [=](nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            uint32_t sum = 0;
            // Extension: read image data from handle
            uint32_t px1 = sycl::ext::oneapi::read_image<uint32_t>(
                imgHandle1, int2(dim0, dim1));
            uint32_t px2 = sycl::ext::oneapi::read_image<uint32_t>(
                imgHandle2, int2(dim0, dim1));
            sum = px1 + px2;
            outAcc[id<2>{dim1, dim0}] = sum;
          });
    });

  } catch (sycl::exception e) {
    std::cerr << "Kernel submission failed! " << e.what() << std::endl;
    assert(false);
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  // Cleanup
  try {
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandle1);
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandle2);
    sycl::free(device_ptr1, ctxt);                    // USM
    sycl::ext::oneapi::free_image(ctxt, device_ptr2); // non-USM
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
