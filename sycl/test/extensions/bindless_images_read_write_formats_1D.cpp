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
  // data, sampling, calculation
  //   A: 0,1,2,3,... B: 512, 511, 510, 509,...        -- data in
  //   C: 512,512,512,512,...                          -- sum (A+B)
  constexpr uint N = 512;
  std::vector<uint4> out(N);
  std::vector<uint> expected(N);
  std::vector<uint4> dataIn1(N);
  std::vector<uint4> dataIn2(N);
  uint exp = 512;
  for (uint i = 0; i < N; i++) {
    expected[i] = exp;
    dataIn1[i] = {i};
    dataIn2[i] = {N - i};
  }

  size_t width = N;

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::image_descriptor desc({width}, image_channel_order::rgba,
                                           image_channel_type::unsigned_int32);

  // Extension: returns the device pointer to the allocated memory
  // Input images memory
  auto device_ptr1 = sycl::ext::oneapi::allocate_image(ctxt, desc);
  auto device_ptr2 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  // Output image memory
  auto device_ptr3 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (device_ptr1 == nullptr || device_ptr2 == nullptr ||
      device_ptr3 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  sycl::ext::oneapi::copy_image(ctxt, device_ptr1, dataIn1.data(), desc,
                                sycl::ext::oneapi::image_copy_flags::HtoD);
  sycl::ext::oneapi::copy_image(ctxt, device_ptr2, dataIn2.data(), desc,
                                sycl::ext::oneapi::image_copy_flags::HtoD);

  // Extension: create the image and return the handle
  sycl::ext::oneapi::image_handle imgIn1 =
      sycl::ext::oneapi::create_image(ctxt, device_ptr1);
  sycl::ext::oneapi::image_handle imgIn2 =
      sycl::ext::oneapi::create_image(ctxt, device_ptr2);

  sycl::ext::oneapi::image_handle imgOut =
      sycl::ext::oneapi::create_image(ctxt, device_ptr3);

  try {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<image_addition>(N, [=](id<1> id) {
        float sum = 0;
        // Extension: read image data from handle
        uint4 px1 = sycl::ext::oneapi::read_image<uint4>(imgIn1, int(id[0]));
        uint4 px2 = sycl::ext::oneapi::read_image<uint4>(imgIn2, int(id[0]));

        sum = px1[0] + px2[0];
        // Extension: write to image with handle
        sycl::ext::oneapi::write_image<uint4>(imgOut, int(id[0]), uint4(sum));
      });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  sycl::ext::oneapi::copy_image(ctxt, out.data(), device_ptr3, desc,
                                sycl::ext::oneapi::image_copy_flags::DtoH);

  // Cleanup
  try {
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgIn1);
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgIn2);
    sycl::ext::oneapi::destroy_image_handle(ctxt, imgOut);
    sycl::ext::oneapi::free_image(ctxt, device_ptr1);
    sycl::ext::oneapi::free_image(ctxt, device_ptr2);
    sycl::ext::oneapi::free_image(ctxt, device_ptr3);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }

  // collect and validate output
  // we use floats but only take the first element
  // data, sampling, calculation
  //   A: 0,1,2,3,... B: 512, 511, 510, 509,...        -- data in
  //   C: 512,512,512,512,...                          -- sum (A+B)
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
