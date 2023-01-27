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
  constexpr size_t N = 32;
  size_t width = N;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn1(N);
  for (int i = 0; i < N; i++) {
    expected[i] = i;
    dataIn1[i] = float(i);
  }

  // Image descriptor
  sycl::ext::oneapi::image_descriptor desc({width}, image_channel_order::r,
                                           image_channel_type::fp32);

  sampler samp1(coordinate_normalization_mode::normalized,
                addressing_mode::repeat, filtering_mode::linear);

  // Extension: returns the device pointer to the allocated memory
  auto device_ptr1 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (device_ptr1 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  sycl::ext::oneapi::copy_image(q, device_ptr1, dataIn1.data(), desc,
                                sycl::ext::oneapi::image_copy_flags::HtoD);
  q.wait();

  // Extension: create the image and return the handle
  auto imgHandle1 =
    sycl::ext::oneapi::create_image(ctxt, device_ptr1, samp1, desc);

  try {
    buffer<float, 1> buf((float *)out.data(), N);
    q.submit([&](handler &cgh) {
      auto outAcc = buf.get_access<access_mode::write>(cgh, N);

      cgh.parallel_for<image_addition>(N, [=](id<1> id) {
        // Normalize coordinate -- +0.5 to look towards centre of pixel
        float x = float(id[0] + 0.5) / (float)N;
        // Extension: read image data from handle
        float px1 = sycl::ext::oneapi::read_image<float>(imgHandle1, x);

        outAcc[id] = px1;
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
    sycl::ext::oneapi::free_image(ctxt, device_ptr1);
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
