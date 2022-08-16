//==----------- bindless_textures.hpp --- SYCL bindless textures -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/range.hpp>

#include <cstdint>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

/// A class to describe the properties of an image.
class image_descriptor {
public:
  /// Prototype does not allow choice of image format, and is 1D only.
  template <int Dims>
  image_descriptor(sycl::range<Dims> imgDims)
      : m_num_dimensions{Dims}, m_dimensions{imgDims[0]} {
    static_asset(Dims == 1, "Prototype only supports 1D textures.");
  }

  inline size_t get(int dimension) const { return m_dimensions[dimension]; }
  size_t &operator[](int dimension) { return m_dimensions[dimension]; }
  size_t operator[](int dimension) const { return m_dimensions[dimension]; }

  inline size_t dimensions() const { return m_num_dimensions; }

private:
  int m_num_dimensions;
  sycl::range<1> m_dimensions;
};

/// Opaque image handle type.
using image_handle = uint64_t;

/** Create an image handle.
 *  @param imageDesc Image meta-data.
 *  @param usmAllocation A pointer to a USM allocation large enough to
 *  store the image described by imageDesc.
 *  @param syclContext The context on which the handle is valid.
 *  @return An image handle
 **/
image_handle create_image_handle(image_descriptor imageDesc,
                                 void *usmAllocation,
                                 sycl::context &syclContext);

/** Destroy an image handle. Does not free memory backing the handle.
 *  @param imageHandle The handle to destroy.
 *  @param syclContext The context the handle is valid in.
 **/
void destroy_image_handle(image_handle &imageHandle,
                          const sycl::context &syclContext);

} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
