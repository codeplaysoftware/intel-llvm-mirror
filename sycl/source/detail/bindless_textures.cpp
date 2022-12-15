//==----------- bindless_textures.hpp --- SYCL bindless textures -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/oneapi/bindless_textures.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>

#include <detail/context_impl.hpp>
#include <detail/plugin_printers.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {

/** Create an image handle.
 *  @param imageDesc Image meta-data.
 *  @param usmAllocation A pointer to a USM allocation large enough to
 *  store the image described by imageDesc.
 *  @param syclContext The context on which the handle is valid.
 *  @return An image handle
 **/
__SYCL_EXPORT image_handle create_image_handle(image_descriptor imageDesc,
                                 void *usmAllocation,
                                 sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc;
  piDesc.image_width = imageDesc[0];
  // Unused properties in MVP
  piDesc.image_height = imageDesc.dimensions() > 1 ? imageDesc[1] : 0;
  piDesc.image_depth = imageDesc.dimensions() > 2 ? imageDesc[2] : 0;
  piDesc.image_type = PI_MEM_TYPE_IMAGE1D;
  piDesc.image_array_size = 0;
  piDesc.image_row_pitch = 0;
  piDesc.image_slice_pitch = 0;
  piDesc.num_mip_levels = 0;
  piDesc.num_samples = 0;
  piDesc.buffer = nullptr;

  // MVP assumes following format:
  pi_image_format piFormat;
  piFormat.image_channel_data_type = PI_IMAGE_CHANNEL_TYPE_FLOAT;
  piFormat.image_channel_order = PI_IMAGE_CHANNEL_ORDER_RGBA;

  // Call impl.
  pi_image_handle piImageHandle;
  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextImgHandleCreate>(
      &piImageHandle, C, &piDesc, &piFormat, usmAllocation);

  if (Error != PI_SUCCESS) {
    return image_handle(nullptr);
  }
  image_handle ImageHandle{piImageHandle};
  return ImageHandle;
}

/** Destroy an image handle. Does not free memory backing the handle.
 *  @param imageHandle The handle to destroy.
 *  @param syclContext The context the handle is valid in.
 **/
__SYCL_EXPORT void destroy_image_handle(image_handle &imageHandle,
                          const sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;
  pi_image_handle piImageHandle = imageHandle;

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextImgHandleDestroy>(
      C, &piImageHandle);
  imageHandle = piImageHandle;
}

/**
 *
 *TODO: Add briefs
 */
__SYCL_EXPORT uint64_t allocate_image(sycl::range<3> range,
                                      image_channel_order order,
                                      image_channel_type type,
                                      sycl::context &syclContext) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc;
  piDesc.image_width = range[0];
  // Unused properties in MVP
  piDesc.image_height = range.size() > 1 ? range[1] : 0;
  piDesc.image_depth = range.size() > 2 ? range[2] : 0;
  piDesc.image_type = PI_MEM_TYPE_IMAGE1D;
  piDesc.image_array_size = 0;
  piDesc.image_row_pitch = 0;
  piDesc.image_slice_pitch = 0;
  piDesc.num_mip_levels = 0;
  piDesc.num_samples = 0;
  piDesc.buffer = nullptr;

  // MVP assumes following format:
  pi_image_format piFormat;
  piFormat.image_channel_data_type = PI_IMAGE_CHANNEL_TYPE_FLOAT;
  piFormat.image_channel_order = PI_IMAGE_CHANNEL_ORDER_RGBA;

  // Call impl.
  // TODO: replace 1 with flags
  uint64_t *devPtr = new uint64_t;
  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageAllocate>(
      C, 1, &piFormat, &piDesc, devPtr);

  if (Error != PI_SUCCESS)
  {
    return 0;  // nullptr
  }

  return *devPtr;
}

/**
 * @brief
 *
 */
__SYCL_EXPORT image_handle create_image(uint64_t devPtr, void *data,
                                        image_channel_order order,
                                        image_channel_type type,
                                        sycl::range<3> range /*sampler, pitch*/,
                                        sycl::context &syclContext) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc;
  piDesc.image_width = range[0];
  // Unused properties in MVP
  piDesc.image_height = range.size() > 1 ? range[1] : 0;
  piDesc.image_depth = range.size() > 2 ? range[2] : 0;
  piDesc.image_type = PI_MEM_TYPE_IMAGE1D;
  piDesc.image_array_size = 0;
  piDesc.image_row_pitch = 0;
  piDesc.image_slice_pitch = 0;
  piDesc.num_mip_levels = 0;
  piDesc.num_samples = 0;
  piDesc.buffer = nullptr;

  // Call impl.
  pi_image_handle piImageHandle;
  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageCreate>(
      C, devPtr, &piImageHandle);

  if (Error != PI_SUCCESS) {
    return image_handle(nullptr);
  }
  image_handle ImageHandle{piImageHandle};
  return ImageHandle;
}

/**
 * @brief
 *
 */
__SYCL_EXPORT void copy_image(uint64_t devPtr, void *data, sycl::range<3> range,
                              uint64_t direction, sycl::context &syclContext) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc;
  piDesc.image_width = range[0];
  // Unused properties in MVP
  piDesc.image_height = range.size() > 1 ? range[1] : 0;
  piDesc.image_depth = range.size() > 2 ? range[2] : 0;
  piDesc.image_type =
      range.size() > 2
          ? PI_MEM_TYPE_IMAGE3D
          : (range.size() > 1 ? PI_MEM_TYPE_IMAGE2D : PI_MEM_TYPE_IMAGE1D);
  piDesc.image_array_size = 0;
  piDesc.image_row_pitch = 0;
  piDesc.image_slice_pitch = 0;
  piDesc.num_mip_levels = 0;
  piDesc.num_samples = 0;
  piDesc.buffer = nullptr;

  pi_image_format piFormat;
  piFormat.image_channel_data_type = PI_IMAGE_CHANNEL_TYPE_FLOAT;
  piFormat.image_channel_order = PI_IMAGE_CHANNEL_ORDER_RGBA;

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageCopy>(
      C, devPtr, data, &piFormat, &piDesc, direction);
}
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
