//==----------- bindless_textures.hpp --- SYCL bindless textures -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/ext/oneapi/bindless_textures.hpp>
#include <sycl/sampler.hpp>

#include <detail/context_impl.hpp>
#include <detail/image_impl.hpp>
#include <detail/plugin_printers.hpp>
#include <detail/sampler_impl.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {

__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        image_handle &imageHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;
  pi_image_handle piImageHandle = imageHandle.value;

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageHandleDestroy>(
      C, piImageHandle);

  if (Error != PI_SUCCESS) {
      throw std::invalid_argument("Failed to destroy image_handle");
  }

  /// TODO: would be nice to have overloaded assignment operator for this
  imageHandle.value = piImageHandle;
}

__SYCL_EXPORT void *allocate_image(const sycl::context &syclContext,
                                      image_descriptor desc) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc;
  piDesc.image_width = desc.width;
  piDesc.image_height = desc.height;
  piDesc.image_depth = desc.depth;
  piDesc.image_type =
      desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                   : (desc.height > 0 ? PI_MEM_TYPE_IMAGE2D : PI_MEM_TYPE_IMAGE1D);
  // Unused properties in MVP
  piDesc.image_array_size = 0;
  piDesc.image_row_pitch = 0;
  piDesc.image_slice_pitch = 0;
  piDesc.num_mip_levels = 0;
  piDesc.num_samples = 0;
  piDesc.buffer = nullptr;

  // MVP assumes following format:
  pi_image_format piFormat;
  piFormat.image_channel_data_type =
      sycl::_V1::detail::convertImageFormat(desc.format).image_channel_data_type;
  piFormat.image_channel_order =
      sycl::_V1::detail::convertImageFormat(desc.format).image_channel_order;

  // Call impl.
  // TODO: replace 1 with flags
  void *devPtr;
  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageAllocate>(
      C, 1, &piFormat, &piDesc, &devPtr);

  if (Error != PI_SUCCESS) {
    return nullptr;
  }

  return devPtr;
}

__SYCL_EXPORT image_handle create_image(const sycl::context &syclContext,
                                        void *devPtr) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  // Call impl.
  pi_image_handle piImageHandle;
  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageCreate>(
      C, devPtr, &piImageHandle);

  if (Error != PI_SUCCESS) {
    return image_handle{nullptr};
  }
  return image_handle{piImageHandle};
}

__SYCL_EXPORT image_handle create_image(const sycl::context &syclContext,
                                        void *devPtr, sampler &sampler) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  // Obtain pi_sampler
  std::shared_ptr<sycl::detail::sampler_impl> SamplerImpl =
      sycl::detail::getSyclObjImpl(sampler);
  pi_sampler piSampler = SamplerImpl->getOrCreateSampler(syclContext);

  // Call impl.
  pi_image_handle piImageHandle;
  Error =
      Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemSampledImageCreate>(
          C, piSampler, devPtr, &piImageHandle);

  if (Error != PI_SUCCESS) {
    return image_handle{nullptr};
  }
  return image_handle{piImageHandle};
}

__SYCL_EXPORT void copy_image(const sycl::context &syclContext, void *devPtr,
                              void *data, image_descriptor desc,
                              image_copy_flags flags) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc;
  piDesc.image_width = desc.width;
  piDesc.image_height = desc.height;
  piDesc.image_depth = desc.depth;
  piDesc.image_type =
      desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                   : (desc.height > 0 ? PI_MEM_TYPE_IMAGE2D : PI_MEM_TYPE_IMAGE1D);
  // Unused properties in MVP
  piDesc.image_array_size = 0;
  piDesc.image_row_pitch = 0;
  piDesc.image_slice_pitch = 0;
  piDesc.num_mip_levels = 0;
  piDesc.num_samples = 0;
  piDesc.buffer = nullptr;

  pi_image_format piFormat;
  piFormat.image_channel_data_type =
      sycl::_V1::detail::convertImageFormat(desc.format).image_channel_data_type;
  piFormat.image_channel_order =
      sycl::_V1::detail::convertImageFormat(desc.format).image_channel_order;

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageCopy>(
      C, devPtr, data, &piFormat, &piDesc, flags);

  if (Error != PI_SUCCESS) {
      throw std::invalid_argument("Failed to copy image");
  }
}
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
