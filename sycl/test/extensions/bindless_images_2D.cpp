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
  // data, calculation
  // A:   0,1,2,3   B: 0,2,4,6
  //      0,1,2,3      0,2,4,6
  //      0,1,2,3      0,2,4,6
  //      0,1,2,3      0,2,4,6
  //
  // A+B: 0,3,6,9
  //      0,3,6,9
  //      0,3,6,9
  //      0,3,6,9
  size_t height = 4;
  size_t width = 4;
  size_t N = height * width;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N);
  for (int i = 0; i < height; i++) {  // row
    for (int j = 0; j < width; j++) { // column
      expected[j + (i * width)] = j * 3;
      dataIn1[j + (i * width)] = {j, j, j, j};
      dataIn2[j + (i * width)] = {j * 2, j * 2, j * 2, j * 2};
    }
  }

  // TODO: Extension: implement image descriptor to replace duplication

  // Extension: returns the device pointer to the allocated memory
  uint64_t device_ptr1 = _V1::ext::oneapi::allocate_image(
      {width, height, 0}, image_channel_order::rgba, image_channel_type::fp32,
      ctxt);

  uint64_t device_ptr2 = _V1::ext::oneapi::allocate_image(
      {width, height, 0}, image_channel_order::rgba, image_channel_type::fp32,
      ctxt);

  if (device_ptr1 == 0 || device_ptr2 == 0) {
    std::cout << "Error allocating images!" << std::endl;
    return 1;
  }

  // Extension: copy over data to device
  // 0 -- hostToDevice -- TODO: expand on this enum
  _V1::ext::oneapi::copy_image(device_ptr1, dataIn1.data(), {width, height, 0},
                               0, ctxt);
  _V1::ext::oneapi::copy_image(device_ptr2, dataIn2.data(), {width, height, 0},
                               0, ctxt);

  // Extension: create the image and return the handle
  _V1::ext::oneapi::image_handle imgHandle1 = _V1::ext::oneapi::create_image(
      device_ptr1, dataIn1.data(), image_channel_order::rgba,
      image_channel_type::fp32, {width, height, 0}, ctxt);
  _V1::ext::oneapi::image_handle imgHandle2 = _V1::ext::oneapi::create_image(
      device_ptr2, dataIn2.data(), image_channel_order::rgba,
      image_channel_type::fp32, {width, height, 0}, ctxt);

  try {
    buffer<float, 2> buf((float *)out.data(), range<2>{width, height});
    q.submit([&](handler &cgh) {
      auto outAcc =
          buf.get_access<access_mode::write>(cgh, range<2>{width, height});

      cgh.parallel_for<image_addition>(
          nd_range<2>{{width, height}, {width, height}}, [=](nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            float sum = 0;
            float4 px1 = _V1::ext::oneapi::read_image<float4>(imgHandle1,
                                                              int2(dim1, dim0));
            float4 px2 = _V1::ext::oneapi::read_image<float4>(imgHandle2,
                                                              int2(dim1, dim0));

            sum = px1[0] + px2[0];
            outAcc[id<2>{dim0, dim1}] = sum;
          });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  // collect and validate output
  // we use float4s but only take the first element
  // data, calculation
  // A:   0,1,2,3   B: 0,2,4,6
  //      0,1,2,3      0,2,4,6
  //      0,1,2,3      0,2,4,6
  //      0,1,2,3      0,2,4,6
  //
  // A+B: 0,3,6,9
  //      0,3,6,9
  //      0,3,6,9
  //      0,3,6,9
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
