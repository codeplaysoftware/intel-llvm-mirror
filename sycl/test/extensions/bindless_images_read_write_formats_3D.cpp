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
  size_t height = 4;
  size_t width = 4;
  size_t depth = 4;
  size_t N = height * width * depth;
  std::vector<float2> out(N);
  std::vector<float> expected(N);
  std::vector<float2> dataIn1(N);
  std::vector<float2> dataIn2(N);
  for (int i = 0; i < height; i++) {  // row
    for (int j = 0; j < width; j++) { // column
      for (int k = 0; k < depth; k++) // depth
      {
        expected[i + height * (j + width * k)] = j * 3;
        dataIn1[i + height * (j + width * k)] = {j, j};
      dataIn2[i + height * (j + width * k)] = {j * 2, j * 2};
      }
    }
  }

  // Image descriptor - can use the same for both images
  _V1::ext::oneapi::image_descriptor desc({width, height, depth},
                                          image_channel_order::rg,
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
      cgh.parallel_for<image_addition>(
          nd_range<3>{{width, height, depth}, {width, height, depth}},
          [=](nd_item<3> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            size_t dim2 = it.get_local_id(2);
            float sum = 0;
            // Extension: read image data from handle
            float2 px1 = _V1::ext::oneapi::read_image<float2>(
                imgIn1, int4(dim0, dim1, dim2, 0));
            float2 px2 = _V1::ext::oneapi::read_image<float2>(
                imgIn2, int4(dim0, dim1, dim2, 0));

            sum = px1[0] + px2[0];
            // Extension: write to image with handle
            _V1::ext::oneapi::write_image<float2>(
                imgOut, int4(dim0, dim1, dim2, 0), float2(sum));
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
