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
  size_t numImages = 5;
  size_t width = 7;
  size_t height = 3;
  size_t N = width * height;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float4> dataIn(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[j + (height * i)] = (j + (height * i)) * numImages;
      dataIn[j + (height * i)] = {j + (height * i), 0, 0, 0};
    }
  }

  // Image descriptor - can use the same for all images
  sycl::ext::oneapi::image_descriptor desc(
      {width, height}, image_channel_order::rgba, image_channel_type::fp32);

  // Allocate each image and save the device ptrs
  std::vector<void *> imgAllocations;
  for (int i = 0; i < numImages; i++) {
    // Extension: returns the device pointer to the allocated memory
    auto device_ptr = sycl::ext::oneapi::allocate_image(ctxt, desc);
    if (device_ptr == nullptr) {
      std::cout << "Error allocating image!" << std::endl;
      return 1;
    }
    imgAllocations.push_back(device_ptr);
  }

  // Copy over data to device for each image
  for (int i = 0; i < numImages; i++) {
    // Extension: copy over data to device
    q.ext_image_memcpy(imgAllocations[i], dataIn.data(), desc,
                       sycl::ext::oneapi::image_copy_flags::HtoD);
  }
  q.wait();

  // Create the images and return the handles
  std::vector<sycl::ext::oneapi::unsampled_image_handle> imgHandles;
  for (int i = 0; i < numImages; i++) {
    // Extension: create the image and return the handle
    sycl::ext::oneapi::unsampled_image_handle imgHandle =
        sycl::ext::oneapi::create_image(ctxt, imgAllocations[i], desc);
    imgHandles.push_back(imgHandle);
  }

  try {
    // Cuda stores data in column-major fashion
    // SYCL deals with indexing in row-major fashion
    // Reverse output buffer dimensions and access to convert
    // the cuda column-major data back to row-major
    buffer<float, 2> buf(out.data(), range<2>{height, width});
    buffer imgHandlesBuf{imgHandles};
    q.submit([&](handler &cgh) {
      accessor outAcc{buf, cgh, write_only};
      accessor imgHandleAcc{imgHandlesBuf, cgh, read_only};

      cgh.parallel_for<image_addition>(
          nd_range<2>{{width, height}, {width, height}}, [=](nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Sum each image by reading their handle
            float sum = 0;
            for (int i = 0; i < numImages; i++) {
              // Extension: read image data from handle
              sum += (sycl::ext::oneapi::read_image<float4>(
                  imgHandleAcc[i], int2(dim0, dim1)))[0];
            }
            outAcc[id<2>{dim1, dim0}] = sum;
          });
    });

    // Using image handles requires manual synchronization
    q.wait_and_throw();

  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    assert(false);
  }

  // Cleanup
  try {
    for (int i = 0; i < numImages; i++) {
      sycl::ext::oneapi::destroy_image_handle(ctxt, imgHandles[i]);
      sycl::ext::oneapi::free_image(ctxt, imgAllocations[i]);
    }
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
