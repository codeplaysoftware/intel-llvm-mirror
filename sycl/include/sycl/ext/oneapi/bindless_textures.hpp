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

#include <sycl/ext/oneapi/bindless_image_descriptor.hpp>

#include <cstdint>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {

/// Opaque unsampled image handle type.
struct unsampled_image_handle {
  unsigned long value;
};

/// Opaque sampled image handle type.
struct sampled_image_handle {
  unsigned long value;
};

// SPIR-V Image Types
#ifdef __SYCL_DEVICE_ONLY__
#define img_type(Dim, AMSuffix) __ocl_image##Dim##d_##AMSuffix##_t
#define smp_type(Dim, AMSuffix) __ocl_sampled_image##Dim##d_##AMSuffix##_t
#endif

/**
 *  @brief   Allocate image memory based on image_descriptor
 *  @param   syclContext The context in which we create our handle
 *  @param   desc The image descriptor
 *  @returns Handle to allocated memory on the GPU
 */
__SYCL_EXPORT image_mem_handle allocate_image(const sycl::context &syclContext,
                                              image_descriptor desc);

/**
 *  @brief   Create an image and return the device handle
 *  @param   syclContext The context in which we create our handle
 *  @param   devPtr Device memory handle for created image
 *  @returns Handle to created image object on the GPU
 */
__SYCL_EXPORT unsampled_image_handle create_image(
    const sycl::context &syclContext, void *devPtr, image_descriptor desc);

__SYCL_EXPORT unsampled_image_handle
create_image(const sycl::context &syclContext, image_mem_handle memHandle,
             image_descriptor desc);

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

__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, image_mem_handle memHandle,
             sampler &sampler, image_descriptor desc);

/**
 *  @brief   Free image memory
 *  @param   syclContext The context in which we create our handle
 *  @param   memory_handle The image memory handle
 */
__SYCL_EXPORT void free_image(const sycl::context &syclContext,
                              image_mem_handle memoryHandle);

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

__SYCL_EXPORT void *pitched_alloc_device(size_t *result_pitch,
                                         size_t width_in_bytes, size_t height,
                                         unsigned int element_size_bytes,
                                         const queue &q);

namespace detail {
template <typename CoordT> constexpr size_t coord_size() {
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
template <typename DataT, typename CoordT>
DataT read_image(const unsampled_image_handle &imageHandle,
                 const CoordT &coords) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    return __invoke__ImageRead<DataT, uint64_t, CoordT>(imageHandle.value,
                                                        coords);
#else
    if constexpr (coordSize == 1) {
      return __invoke__ImageRead<DataT, img_type(1, ro), CoordT>(
          __spirv_ConvertUToImageNV<img_type(1, ro)>(imageHandle.value),
          coords);
    }
    if constexpr (coordSize == 2) {
      return __invoke__ImageRead<DataT, img_type(2, ro), CoordT>(
          __spirv_ConvertUToImageNV<img_type(2, ro)>(imageHandle.value),
          coords);
    }
    if constexpr (coordSize == 4) {
      return __invoke__ImageRead<DataT, img_type(3, ro), CoordT>(
          __spirv_ConvertUToImageNV<img_type(3, ro)>(imageHandle.value),
          coords);
    }
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
DataT read_image(const sampled_image_handle &imageHandle,
                 const CoordT &coords) {
  constexpr size_t coordSize = detail::coord_size<CoordT>();
  if constexpr (coordSize == 1 || coordSize == 2 || coordSize == 4) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
    return __invoke__ImageRead<DataT, uint64_t, CoordT>(imageHandle.value,
                                                        coords);
#else
    if constexpr (coordSize == 1) {
      return __invoke__ImageReadExpSampler<DataT, smp_type(1, ro), CoordT>(
          __spirv_ConvertUToSampledImageNV<smp_type(1, ro)>(imageHandle.value),
          coords);
    }
    if constexpr (coordSize == 2) {
      return __invoke__ImageReadExpSampler<DataT, smp_type(2, ro), CoordT>(
          __spirv_ConvertUToSampledImageNV<smp_type(2, ro)>(imageHandle.value),
          coords);
    }
    if constexpr (coordSize == 4) {
      return __invoke__ImageReadExpSampler<DataT, smp_type(3, ro), CoordT>(
          __spirv_ConvertUToSampledImageNV<smp_type(3, ro)>(imageHandle.value),
          coords);
    }
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
    if constexpr (coordSize == 1) {
      __invoke__ImageWrite<img_type(1, wo), CoordT, DataT>(
          __spirv_ConvertUToImageNV<img_type(1, wo)>(imageHandle.value), Coords,
          Color);
    }
    if constexpr (coordSize == 2) {
      __invoke__ImageWrite<img_type(2, wo), CoordT, DataT>(
          __spirv_ConvertUToImageNV<img_type(2, wo)>(imageHandle.value), Coords,
          Color);
    }
    if constexpr (coordSize == 4) {
      __invoke__ImageWrite<img_type(3, wo), CoordT, DataT>(
          __spirv_ConvertUToImageNV<img_type(3, wo)>(imageHandle.value), Coords,
          Color);
    }
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
