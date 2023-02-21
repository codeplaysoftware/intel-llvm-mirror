// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {

  device dev;
  queue q(dev);
  auto ctxt = q.get_context();

  size_t height = 13;
  size_t width = 7;
  size_t depth = 11;

  // Submit dummy kernel to let the runtime decide the backend (CUDA)
  // Without this, the default Level Zero backend is active
  q.submit([&](handler &cgh) { cgh.single_task([](){}); });

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::image_descriptor desc({width, height, depth},
                                           image_channel_order::r,
                                           image_channel_type::signed_int32);

  // Extension: returns the device pointer to the allocated memory
  // Input images memory
  auto img_mem_0 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (img_mem_0.value == nullptr) {
    std::cout << "Error allocating image!" << std::endl;
    return 1;
  }

  bool validated = true;

  auto range = sycl::ext::oneapi::get_image_range(ctxt, img_mem_0);
  if (range[0] == width) {
    std::cout << "width is correct!\n";
  } else {
    std::cout << "width is NOT correct!\n";
    validated = false;
  }
  if (range[1] == height) {
    std::cout << "height is correct!\n";
  } else {
    std::cout << "height is NOT correct!\n";
    validated = false;
  }
  if (range[2] == depth) {
    std::cout << "depth is correct!\n";
  } else {
    std::cout << "depth is NOT correct!\n";
    validated = false;
  }

  auto flags = sycl::ext::oneapi::get_image_flags(ctxt, img_mem_0);
  if (flags == 0) {
    std::cout << "flags is correct!\n";
  } else {
    std::cout << "flags is NOT correct!\n";
    validated = false;
  }

  auto ctype = sycl::ext::oneapi::get_image_channel_type(ctxt, img_mem_0);
  if (ctype == image_channel_type::signed_int32) {
    std::cout << "channel type is correct!\n";
  } else {
    std::cout << "channel type is NOT correct!\n";
    validated = false;
  }

  auto numchannels = sycl::ext::oneapi::get_image_num_channels(ctxt, img_mem_0);
  if (numchannels == 1) {
    std::cout << "num channels is correct!\n";
  } else {
    std::cout << "num channels is NOT correct!\n";
    validated = false;
  }

  return validated ? 0 : 1;
}
