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
  // data, sampling, calculation
  //   A: 0,1,2,3,... B: 512, 511, 510, 509,...        -- data in
  //   C: 512,512,512,512,...                          -- sum (A+B)
  constexpr size_t N = 512;
  std::vector<float4> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N);
  float exp = 512;
  for (int i = 0; i < N; i++) {
    expected[i] = exp;
    dataIn1[i] = float4(i, i, i, i);
    dataIn2[i] = float4(N - i, N - i, N - i, N - i);
  }

  size_t width = N;

  // Image descriptor - can use the same for both images
  _V1::ext::oneapi::image_descriptor desc({width}, image_channel_order::rgba,
                                          image_channel_type::fp32);

  // Extension: returns the device pointer to the allocated memory
  // Input images memory
  auto device_ptr1 = _V1::ext::oneapi::allocate_image(ctxt, desc);
  auto device_ptr2 = _V1::ext::oneapi::allocate_image(ctxt, desc);

  // Output image memory
  auto device_ptr3 = _V1::ext::oneapi::allocate_image(ctxt, desc);

  if (device_ptr1 == nullptr || device_ptr2 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  _V1::ext::oneapi::copy_image(ctxt, device_ptr1, dataIn1.data(), desc,
                               _V1::ext::oneapi::image_copy_flags::HtoD);
  _V1::ext::oneapi::copy_image(ctxt, device_ptr2, dataIn2.data(), desc,
                               _V1::ext::oneapi::image_copy_flags::HtoD);

  // Extension: create the image and return the handle
  _V1::ext::oneapi::image_handle imgIn1 =
      _V1::ext::oneapi::create_image(ctxt, device_ptr1);
  _V1::ext::oneapi::image_handle imgIn2 =
      _V1::ext::oneapi::create_image(ctxt, device_ptr2);

  _V1::ext::oneapi::image_handle imgOut =
      _V1::ext::oneapi::create_image(ctxt, device_ptr3);

  try {
    q.submit([&](handler &cgh) {
      cgh.parallel_for<image_addition>(N, [=](id<1> id) {
        float sum = 0;
        // Extension: read image data from handle
        float4 px1 = _V1::ext::oneapi::read_image<float4>(imgIn1, int(id[0]));
        float4 px2 = _V1::ext::oneapi::read_image<float4>(imgIn2, int(id[0]));

        sum = px1[0] + px2[0];
        // Extension: write to image with handle
        _V1::ext::oneapi::write_image<float4>(imgOut, int(id[0]),
                                              float4(sum));
      });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  _V1::ext::oneapi::copy_image(ctxt, out.data(), device_ptr3, desc,
                               _V1::ext::oneapi::image_copy_flags::DtoH);

  // Cleanup
  try {
    _V1::ext::oneapi::destroy_image_handle(ctxt, imgIn1);
    _V1::ext::oneapi::destroy_image_handle(ctxt, imgIn2);
    _V1::ext::oneapi::destroy_image_handle(ctxt, imgOut);
    _V1::ext::oneapi::free_image(ctxt, device_ptr1);
    _V1::ext::oneapi::free_image(ctxt, device_ptr2);
    _V1::ext::oneapi::free_image(ctxt, device_ptr3);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }

  // collect and validate output
  // we use float4s but only take the first element
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
