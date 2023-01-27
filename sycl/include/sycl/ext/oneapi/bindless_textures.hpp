//==----------- bindless_textures.hpp --- SYCL bindless textures -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/image_ocl_types.hpp>
#include <sycl/image.hpp>
#include <sycl/range.hpp>
#include <sycl/sampler.hpp>
#include <sycl/queue.hpp>

#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/ext/oneapi/bindless_textures.hpp>

#include <cstdint>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {

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

/// Opaque unsampled image handle type.
typedef struct {
  void *value;
} unsampled_image_handle;

/// Opaque sampled image handle type.
typedef struct {
  void *value;
} sampled_image_handle;

#ifdef __SYCL_DEVICE_ONLY__
using OCLImageTy = typename sycl::detail::opencl_image_type<1, sycl::access::mode::read,
                                                        sycl::access::target::image>::type;
#endif

/**
 *  @brief   Allocate image memory based on image_descriptor
 *  @param   syclContext The context in which we create our handle
 *  @param   desc The image descriptor
 *  @returns Handle to allocated memory on the GPU
 */
__SYCL_EXPORT void *allocate_image(const sycl::context &syclContext,
                                   image_descriptor desc);

/**
 *  @brief   Create an image and return the device handle
 *  @param   syclContext The context in which we create our handle
 *  @param   devPtr Device memory handle for created image
 *  @returns Handle to created image object on the GPU
 */
__SYCL_EXPORT unsampled_image_handle create_image(
    const sycl::context &syclContext, void *devPtr, image_descriptor desc);

/**
 *  @brief   Create a sampled image and return the device handle
 *  @param   syclContext The context in which we create our handle
 *  @param   devPtr Device memory handle for created image
 *  @param   sampler SYCL sampler to sample the image
 *  @returns Handle to created image object on the GPU
 */
__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, void *devPtr,
                     sampler &sampler, image_descriptor desc);

/**
 *  @brief   Copy image data between device and host
 *  @param   syclQueue The queue in which we copy our image
 *  @param   dst_ptr Destination memory handle/pointer
 *  @param   src_ptr Source memory handle/pointer
 *  @param   desc Image descriptor
 *  @param   flags Image copy flags for copy direction
 */
__SYCL_EXPORT void copy_image(const sycl::queue &syclQueue, void *dst_ptr,
                              void *src_ptr, image_descriptor desc,
                              image_copy_flags flags);

/**
 *  @brief   Free image memory
 *  @param   syclContext The context in which we create our handle
 *  @param   memory_handle The image memory handle
 */
__SYCL_EXPORT void free_image(const sycl::context &syclContext,
                              void *memory_handle);

/**
 *  @brief Destroy an image handle. Does not free memory backing the handle.
 *  @param imageHandle The handle to destroy.
 *  @param syclContext The context the handle is valid in.
 **/
__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        unsampled_image_handle &imageHandle);

/**
 *  @brief Destroy an image handle. Does not free memory backing the handle.
 *  @param imageHandle The handle to destroy.
 *  @param syclContext The context the handle is valid in.
 **/
__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        sampled_image_handle &imageHandle);

__SYCL_EXPORT void *
pitched_alloc_device(size_t *result_pitch, size_t width_in_bytes, size_t height,
                     unsigned int element_size_bytes,
                     const queue &q);

namespace detail {
template<typename CoordT>
constexpr size_t coord_size(){
  if constexpr (std::is_scalar<CoordT>::value) {
    return 1;
  } else {
    return CoordT::size();
  }
}
}

/** Read an unsampled image using its handle.
 *  @tparam DataT is the type of data to return.
 *  @tparam CoordT is the input coordinate type. Float, float2, or float4 for
 *1D, 2D and 3D respectively.
 *  @param imageHandle is the image's handle.
 *  @param coords is the coordinate at which to get image data.
 *  @return data from the image.
 *
 * __NVPTX__: Name mangling info
 *            Cuda surfaces require integer coordinates (by bytes)
 *            Cuda textures require float coordinates (by index or normalized)
 *            The name mangling should therefore not interfere with one another
 **/
template <typename DataT, typename CoordT, typename Handle>
DataT read_image(const Handle &imageHandle, const CoordT &coords) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    return __invoke__ImageRead<DataT, uint64_t, CoordT>(
        (uint64_t)imageHandle.value, coords);
#else
    return __invoke__ImageRead<DataT, OCLImageTy, int>(
        __spirv_ConvertUToImageNV<OCLImageTy>(imageHandle.value), (int)coords);
#endif
#else
    assert(false); // Bindless images not yet implemented on host.
#endif
  } else {
    static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                  "Expected input coordinate to be have 1, 2, or 4 components "
                  "for 1D, 2D and 3D images respectively.");
  }
}

template <typename DataT, typename CoordT>
void write_image(const unsampled_image_handle &imageHandle,
                 const CoordT &Coords, const DataT &Color) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    __invoke__ImageWrite<uint64_t, CoordT, DataT>((uint64_t)imageHandle.value,
                                                  Coords, Color);
#else
    __invoke__ImageWrite<OCLImageTy, CoordT, DataT>(
        __spirv_ConvertUToImageNV<OCLImageTy>(imageHandle.value), Coords,
        Color);
#endif
#else
    assert(false); // Bindless images not yet implemented on host.
#endif
  } else {
    static_assert(coordSize == 1 || coordSize == 2 || coordSize == 4,
                  "Expected input coordinate to be have 1, 2, or 4 components "
                  "for 1D, 2D and 3D images respectively.");
  }
}

} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
