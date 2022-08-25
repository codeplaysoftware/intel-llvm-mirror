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

/** Read an image using its handle.
 *  @tparam DataT is the type of data to return. 
 *  @tparam CoordT is the input coordinate type. Float, float2, or float4 for 1D, 2D and 3D respectively.
 *  @param imageHandle is the image's handle.
 *  @param coords is the coordinate at which to get image data.
 *  @return data from the image.
 **/

template <typename DataT, typename CoordT>
DataT read(const image_handle& imageHandle, const CoordT &coords) const noexcept {
	constexpr size_t coordSize = coords.size();
	if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4){
#ifdef __SYCL_DEVICE_ONLY__
    return __invoke__ImageRead<DataT, image_handle, CoordT>(
        imageHandle, coords);
#else
    static_assert(false, "Bindless images not yet implemented on host.");
#endif 
	} else {
		static_assert(false, "Expected input coordinate to be have 1, 2, or 4 components for 1D, 2D and 3D images respectively.");
	}
}


} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
