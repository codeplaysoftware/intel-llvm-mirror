//==------ bindless_image_descriptor.hpp --- SYCL bindless textures --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/image.hpp>
#include <sycl/range.hpp>
#include <cstdint>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {

// Handle to image memory allocated using `allocate_image`
struct image_mem_handle {
  void *value;
};

/// A class to describe the properties of an image.
struct image_descriptor {
  unsigned int width;
  unsigned int height;
  unsigned int depth;
  image_channel_type channel_type;
  image_channel_order channel_order;
  size_t row_pitch;

  image_descriptor(range<1> dims, image_channel_order order,
                   image_channel_type type, size_t pitch = 0)
      : width(dims[0]), height(0), depth(0), channel_type(type),
        channel_order(order), row_pitch(pitch) {}

  image_descriptor(range<2> dims, image_channel_order order,
                   image_channel_type type, size_t pitch = 0)
      : width(dims[0]), height(dims[1]), depth(0), channel_type(type),
        channel_order(order), row_pitch(pitch) {}

  image_descriptor(range<3> dims, image_channel_order order,
                   image_channel_type type, size_t pitch = 0)
      : width(dims[0]), height(dims[1]), depth(dims[2]), channel_type(type),
        channel_order(order), row_pitch(pitch) {}
};

/// Direction to copy data from bindless image handle
/// (Host -> Device) (Device -> Host) etc.
enum image_copy_flags : unsigned int {
  HtoD = 0,
  DtoH = 1,
  DtoD = 2,
};

} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
