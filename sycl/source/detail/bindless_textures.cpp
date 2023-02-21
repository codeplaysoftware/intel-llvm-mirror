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
#include <detail/queue_impl.hpp>
#include <detail/sampler_impl.hpp>

#include <memory>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {

__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        unsampled_image_handle &imageHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;
  pi_image_handle piImageHandle = imageHandle.value;

  Error = Plugin.call_nocheck<
      sycl::detail::PiApiKind::piextMemUnsampledImageHandleDestroy>(
      C, &piImageHandle);

  if (Error != PI_SUCCESS) {
    throw std::invalid_argument("Failed to destroy image_handle");
  }

  /// TODO: would be nice to have overloaded assignment operator for this
  imageHandle.value = piImageHandle;
}

__SYCL_EXPORT void destroy_image_handle(const sycl::context &syclContext,
                                        sampled_image_handle &imageHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;
  pi_image_handle piImageHandle = imageHandle.value;

  Error = Plugin.call_nocheck<
      sycl::detail::PiApiKind::piextMemSampledImageHandleDestroy>(
      C, &piImageHandle);

  if (Error != PI_SUCCESS) {
      throw std::invalid_argument("Failed to destroy image_handle");
  }

  /// TODO: would be nice to have overloaded assignment operator for this
  imageHandle.value = piImageHandle;
}

__SYCL_EXPORT image_mem_handle allocate_image(const sycl::context &syclContext,
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
  piDesc.image_type = desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);
  piDesc.image_array_size = 0;
  piDesc.image_row_pitch = 0;
  piDesc.image_slice_pitch = 0;
  piDesc.num_mip_levels = 0;
  piDesc.num_samples = 0;
  piDesc.buffer = nullptr;

  pi_image_format piFormat;
  piFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(desc.channel_type);
  piFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(desc.channel_order);

  // Call impl.
  // TODO: replace 1 with flags
  image_mem_handle devPtr;
  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageAllocate>(
      C, 1, &piFormat, &piDesc, &devPtr.value);

  if (Error != PI_SUCCESS) {
    return image_mem_handle{nullptr};
  }

  return devPtr;
}

__SYCL_EXPORT void free_image(const sycl::context &syclContext,
                              image_mem_handle memoryHandle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();

Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageFree>(
    C, memoryHandle.value);

return;
}

__SYCL_EXPORT unsampled_image_handle
create_image(const sycl::context &syclContext, image_mem_handle memHandle,
             image_descriptor desc) {
  return create_image(syclContext, memHandle.value, desc);
}

__SYCL_EXPORT unsampled_image_handle create_image(
    const sycl::context &syclContext, void *devPtr, image_descriptor desc) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_desc piDesc = {};
  piDesc.image_type = desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);
  piDesc.image_width = desc.width;
  piDesc.image_height = desc.height;
  piDesc.image_depth = desc.depth;
  piDesc.image_row_pitch = desc.row_pitch;

  pi_image_format piFormat;
  piFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(desc.channel_type);
  piFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(desc.channel_order);

  // Call impl.
  pi_image_handle piImageHandle;
  Error =
      Plugin
          .call_nocheck<sycl::detail::PiApiKind::piextMemUnsampledImageCreate>(
              C, devPtr, &piFormat, &piDesc, &piImageHandle);

  if (Error != PI_SUCCESS) {
    return unsampled_image_handle{0};
  }
  return unsampled_image_handle{piImageHandle};
}

__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, image_mem_handle memHandle,
             sampler &sampler, image_descriptor desc) {
  return create_image(syclContext, memHandle.value, sampler, desc);
}

__SYCL_EXPORT sampled_image_handle
create_image(const sycl::context &syclContext, void *devPtr,
                     sampler &sampler, image_descriptor desc) {

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  // Obtain pi_sampler
  std::shared_ptr<sycl::detail::sampler_impl> SamplerImpl =
      sycl::detail::getSyclObjImpl(sampler);
  pi_sampler piSampler = SamplerImpl->getOrCreateSampler(syclContext);

  pi_image_desc piDesc = {};
  piDesc.image_type = desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);
  piDesc.image_width = desc.width;
  piDesc.image_height = desc.height;
  piDesc.image_depth = desc.depth;
  piDesc.image_row_pitch = desc.row_pitch;

  pi_image_format piFormat;
  piFormat.image_channel_data_type =
      sycl::_V1::detail::convertChannelType(desc.channel_type);
  piFormat.image_channel_order =
      sycl::_V1::detail::convertChannelOrder(desc.channel_order);

  // Call impl.
  pi_image_handle piImageHandle;
  Error =
      Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemSampledImageCreate>(
          C, devPtr, &piFormat, &piDesc, piSampler, &piImageHandle);

  if (Error != PI_SUCCESS) {
    return sampled_image_handle{0};
  }
  return sampled_image_handle{piImageHandle};
}

__SYCL_EXPORT void *pitched_alloc_device(size_t *ResultPitch,
                                         size_t WidthInBytes, size_t Height,
                                         unsigned int ElementSizeBytes,
                                         const queue &Q) {
  void *RetVal = nullptr;
  if (WidthInBytes == 0 || Height == 0 || ElementSizeBytes == 0)
    return nullptr;

  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(Q.get_context());
  if (CtxImpl->is_host()) {
    return nullptr;
  }

  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;
  pi_device Id;

  Id = sycl::detail::getSyclObjImpl(Q.get_device())->getHandleRef();

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextUSMPitchedAlloc>(
      &RetVal, ResultPitch, C, Id, nullptr, WidthInBytes, Height,
      ElementSizeBytes);

  (void)Error;

  return RetVal;
}

__SYCL_EXPORT sycl::range<3>
get_image_range(const sycl::context &syclContext,
                const image_mem_handle mem_handle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  size_t Width, Height, Depth;

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      mem_handle.value, PI_IMAGE_INFO_WIDTH, &Width, nullptr);
  if (Error != PI_SUCCESS)
    return {0, 0, 0};

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      mem_handle.value, PI_IMAGE_INFO_HEIGHT, &Height, nullptr);
  if (Error != PI_SUCCESS)
    return {0, 0, 0};

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      mem_handle.value, PI_IMAGE_INFO_DEPTH, &Depth, nullptr);
  if (Error != PI_SUCCESS)
    return {0, 0, 0};

  return {Width, Height, Depth};
}

__SYCL_EXPORT unsigned int get_image_flags(const sycl::context &syclContext,
                                           const image_mem_handle mem_handle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  unsigned int Flags;

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      mem_handle.value, PI_IMAGE_INFO_FLAGS, &Flags, nullptr);
  if (Error != PI_SUCCESS)
    return ~(unsigned int)0;

  return Flags;
}

__SYCL_EXPORT sycl::image_channel_type
get_image_channel_type(const sycl::context &syclContext,
                       const image_mem_handle mem_handle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_format PIFormat;

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      mem_handle.value, PI_IMAGE_INFO_FORMAT, &PIFormat, nullptr);
  if (Error != PI_SUCCESS)
    return (sycl::image_channel_type)~0;

  image_channel_type ChannelType =
      sycl::detail::convertChannelType(PIFormat.image_channel_data_type);

  return ChannelType;
}

__SYCL_EXPORT unsigned int
get_image_num_channels(const sycl::context &syclContext,
                       const image_mem_handle mem_handle) {
  std::shared_ptr<sycl::detail::context_impl> CtxImpl =
      sycl::detail::getSyclObjImpl(syclContext);
  pi_context C = CtxImpl->getHandleRef();
  const sycl::detail::plugin &Plugin = CtxImpl->getPlugin();
  pi_result Error;

  pi_image_format PIFormat;

  Error = Plugin.call_nocheck<sycl::detail::PiApiKind::piextMemImageGetInfo>(
      mem_handle.value, PI_IMAGE_INFO_FORMAT, &PIFormat, nullptr);
  if (Error != PI_SUCCESS)
    return 0;

  image_channel_order Order =
      sycl::detail::convertChannelOrder(PIFormat.image_channel_order);
  switch (Order) {
  case image_channel_order::a:
  case image_channel_order::r:
    return 1;
  case image_channel_order::rx:
  case image_channel_order::rg:
  case image_channel_order::ra:
    return 2;
  case image_channel_order::rgx:
  case image_channel_order::rgb:
    return 3;
  case image_channel_order::rgbx:
  case image_channel_order::rgba:
  case image_channel_order::argb:
  case image_channel_order::bgra:
  case image_channel_order::abgr:
    return 4;
  // Unsupported channel types
  case image_channel_order::intensity:
  case image_channel_order::luminance:
  case image_channel_order::ext_oneapi_srgba:
  default:
    return 0;
  }
}

} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
