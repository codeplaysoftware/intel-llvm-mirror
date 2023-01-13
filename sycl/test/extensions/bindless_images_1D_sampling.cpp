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
  constexpr size_t N = 64;
  size_t width = N;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  for (int i = 0; i < N; i++) {
    expected[i] = i;
    dataIn1[i] = float4(i, i, i, i);
  }

  // Image descriptor
  _V1::ext::oneapi::image_descriptor desc({width},
                                          image_format::r32g32b32a32_sfloat);

  sampler samp1(coordinate_normalization_mode::normalized,
                addressing_mode::clamp, filtering_mode::linear);

  // Extension: returns the device pointer to the allocated memory
  auto device_ptr1 = _V1::ext::oneapi::allocate_image(ctxt, desc);

  if (device_ptr1 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  _V1::ext::oneapi::copy_image(ctxt, device_ptr1, dataIn1.data(), desc,
                               _V1::ext::oneapi::image_copy_flags::HtoD);

  // Extension: create the image and return the handle
  _V1::ext::oneapi::image_handle imgHandle1 =
      _V1::ext::oneapi::create_image(ctxt, device_ptr1, samp1);

  try {
    buffer<float, 1> buf((float *)out.data(), N);
    q.submit([&](handler &cgh) {
      auto outAcc = buf.get_access<access_mode::write>(cgh, N);

      cgh.parallel_for<image_addition>(N, [=](id<1> id) {
        // Normalize coordinate -- +0.5 to look towards centre of pixel
        float x = float(id[0] + 0.5) / (float)N;
        // Extension: read image data from handle
        float4 px1 = _V1::ext::oneapi::read_image<float4>(imgHandle1, x);

        outAcc[id] = px1[0];
      });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  // Cleanup
  try {
    _V1::ext::oneapi::destroy_image_handle(ctxt, imgHandle1);
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
