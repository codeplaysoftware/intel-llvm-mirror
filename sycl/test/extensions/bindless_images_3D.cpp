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
  //            0,1,2,3            0,2,4,6
  //          0,1,2,3            0,2,4,6
  //        0,1,2,3            0,2,4,6
  // A:   0,1,2,3         B: 0,2,4,6
  //      0,1,2,3            0,2,4,6
  //      0,1,2,3            0,2,4,6
  //      0,1,2,3            0,2,4,6
  //
  //            0,3,6,9
  //          0,3,6,9
  //        0,3,6,9
  // A+B: 0,3,6,9
  //      0,3,6,9
  //      0,3,6,9
  //      0,3,6,9
  size_t height = 4;
  size_t width = 4;
  size_t depth = 4;
  size_t N = height * width * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn1(N);
  std::vector<float4> dataIn2(N);
  for (int i = 0; i < height; i++) {  // row
    for (int j = 0; j < width; j++) { // column
      for (int k = 0; k < depth; k++) // depth
      {
        expected[i + width * (j + depth * k)] = j * 3;
        dataIn1[i + width * (j + depth * k)] = {j, j, j, j};
        dataIn2[i + width * (j + depth * k)] = {j * 2, j * 2, j * 2, j * 2};
      }
    }
  }

  // Image descriptor - can use the same for both images
  _V1::ext::oneapi::image_descriptor desc({width, height, depth},
                                          image_format::r32g32b32a32_sfloat);


  // Extension: returns the device pointer to the allocated memory
  auto device_ptr1 = _V1::ext::oneapi::allocate_image(ctxt, desc);
  auto device_ptr2 = _V1::ext::oneapi::allocate_image(ctxt, desc);

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
  _V1::ext::oneapi::image_handle imgHandle1 =
      _V1::ext::oneapi::create_image(ctxt, device_ptr1);
  _V1::ext::oneapi::image_handle imgHandle2 =
      _V1::ext::oneapi::create_image(ctxt, device_ptr2);

  try {
    buffer<float, 3> buf((float *)out.data(), range<3>{width, height, depth});
    q.submit([&](handler &cgh) {
      auto outAcc = buf.get_access<access_mode::write>(
          cgh, range<3>{width, height, depth});

      cgh.parallel_for<image_addition>(
          nd_range<3>{{width, height, depth}, {width, height, depth}},
          [=](nd_item<3> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            size_t dim2 = it.get_local_id(2);
            float sum = 0;
            // Extension: read image data from handle
            float4 px1 = _V1::ext::oneapi::read_image<float4>(
                imgHandle1, int4(dim2, dim1, dim0, 0));
            float4 px2 = _V1::ext::oneapi::read_image<float4>(
                imgHandle2, int4(dim2, dim1, dim0, 0));

            sum = px1[0] + px2[0];
            outAcc[id<3>{dim0, dim1, dim2}] = sum;
          });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  // Cleanup
  try {
    _V1::ext::oneapi::destroy_image_handle(ctxt, imgHandle1);
    _V1::ext::oneapi::destroy_image_handle(ctxt, imgHandle2);
  } catch (...) {
    std::cerr << "Failed to destroy image handle." << std::endl;
    assert(false);
  }

  // collect and validate output
  // we use float4s but only take the first element
  // data, calculation
  //            0,1,2,3            0,2,4,6
  //          0,1,2,3            0,2,4,6
  //        0,1,2,3            0,2,4,6
  // A:   0,1,2,3         B: 0,2,4,6
  //      0,1,2,3            0,2,4,6
  //      0,1,2,3            0,2,4,6
  //      0,1,2,3            0,2,4,6
  //
  //            0,3,6,9
  //          0,3,6,9
  //        0,3,6,9
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
