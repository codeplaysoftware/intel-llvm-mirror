//==-------------- queue.cpp -----------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/backend_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/image_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/event.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/ext/codeplay/experimental/fusion_properties.hpp>
#include <sycl/handler.hpp>
#include <sycl/queue.hpp>
#include <sycl/stl.hpp>
#include <sycl/usm.hpp>

#include <algorithm>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

queue::queue(const context &SyclContext, const device_selector &DeviceSelector,
             const async_handler &AsyncHandler, const property_list &PropList) {

  const std::vector<device> Devs = SyclContext.get_devices();

  auto Comp = [&DeviceSelector](const device &d1, const device &d2) {
    return DeviceSelector(d1) < DeviceSelector(d2);
  };

  const device &SyclDevice = *std::max_element(Devs.begin(), Devs.end(), Comp);

  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), detail::getSyclObjImpl(SyclContext),
      AsyncHandler, PropList);
}

queue::queue(const context &SyclContext, const device &SyclDevice,
             const async_handler &AsyncHandler, const property_list &PropList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), detail::getSyclObjImpl(SyclContext),
      AsyncHandler, PropList);
}

queue::queue(const device &SyclDevice, const async_handler &AsyncHandler,
             const property_list &PropList) {
  impl = std::make_shared<detail::queue_impl>(
      detail::getSyclObjImpl(SyclDevice), AsyncHandler, PropList);
}

queue::queue(cl_command_queue clQueue, const context &SyclContext,
             const async_handler &AsyncHandler) {
  impl = std::make_shared<detail::queue_impl>(
      reinterpret_cast<RT::PiQueue>(clQueue),
      detail::getSyclObjImpl(SyclContext), AsyncHandler);
}

queue::queue(const context &SyclContext, const device_selector &deviceSelector,
             const property_list &PropList)
    : queue(SyclContext, deviceSelector,
            detail::getSyclObjImpl(SyclContext)->get_async_handler(),
            PropList) {}

queue::queue(const context &SyclContext, const device &SyclDevice,
             const property_list &PropList)
    : queue(SyclContext, SyclDevice,
            detail::getSyclObjImpl(SyclContext)->get_async_handler(),
            PropList) {}

cl_command_queue queue::get() const { return impl->get(); }

context queue::get_context() const { return impl->get_context(); }

device queue::get_device() const { return impl->get_device(); }

bool queue::is_host() const {
  bool IsHost = impl->is_host();
  assert(!IsHost && "queue::is_host should not be called in implementation.");
  return IsHost;
}

void queue::throw_asynchronous() { impl->throw_asynchronous(); }

event queue::memset(void *Ptr, int Value, size_t Count) {
  return impl->memset(impl, Ptr, Value, Count, {});
}

event queue::memset(void *Ptr, int Value, size_t Count, event DepEvent) {
  return impl->memset(impl, Ptr, Value, Count, {DepEvent});
}

event queue::memset(void *Ptr, int Value, size_t Count,
                    const std::vector<event> &DepEvents) {
  return impl->memset(impl, Ptr, Value, Count, DepEvents);
}

event queue::memcpy(void *Dest, const void *Src, size_t Count) {
  return impl->memcpy(impl, Dest, Src, Count, {});
}

event queue::memcpy(void *Dest, const void *Src, size_t Count, event DepEvent) {
  return impl->memcpy(impl, Dest, Src, Count, {DepEvent});
}

event queue::memcpy(void *Dest, const void *Src, size_t Count,
                    const std::vector<event> &DepEvents) {
  return impl->memcpy(impl, Dest, Src, Count, DepEvents);
}

event queue::mem_advise(const void *Ptr, size_t Length, pi_mem_advice Advice) {
  return mem_advise(Ptr, Length, int(Advice));
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice) {
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), {});
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        event DepEvent) {
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), {DepEvent});
}

event queue::mem_advise(const void *Ptr, size_t Length, int Advice,
                        const std::vector<event> &DepEvents) {
  return impl->mem_advise(impl, Ptr, Length, pi_mem_advice(Advice), DepEvents);
}

event queue::ext_image_memcpy(ext::oneapi::image_mem_handle Dest, void *Src,
                              const ext::oneapi::image_descriptor &Desc) {
  return this->ext_image_memcpy(Dest, Src, Desc, std::vector<event>{});
}

event queue::ext_image_memcpy(ext::oneapi::image_mem_handle Dest, void *Src,
                              const ext::oneapi::image_descriptor &Desc,
                              event DepEvent) {
  return this->ext_image_memcpy(Dest, Src, Desc, {DepEvent});
}

event queue::ext_image_memcpy(ext::oneapi::image_mem_handle Dest, void *Src,
                              const ext::oneapi::image_descriptor &Desc,
                              const std::vector<event> &DepEvents) {
  RT::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_row_pitch = Desc.row_pitch;
  PiDesc.image_type = Desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);
  RT::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order =
      detail::convertChannelOrder(Desc.channel_order);

  return impl->ext_image_memcpy(impl, Dest.value, Src, PiDesc, PiFormat,
                                pi_image_copy_flags::PI_IMAGE_COPY_HTOD,
                                DepEvents);
}

event queue::ext_image_memcpy(void *Dest, ext::oneapi::image_mem_handle Src,
                              const ext::oneapi::image_descriptor &Desc) {
  return this->ext_image_memcpy(Dest, Src, Desc, std::vector<event>{});
}

event queue::ext_image_memcpy(void *Dest, ext::oneapi::image_mem_handle Src,
                              const ext::oneapi::image_descriptor &Desc,
                              event DepEvent) {
  return this->ext_image_memcpy(Dest, Src, Desc, {DepEvent});
}

event queue::ext_image_memcpy(void *Dest, ext::oneapi::image_mem_handle Src,
                              const ext::oneapi::image_descriptor &Desc,
                              const std::vector<event> &DepEvents) {
  RT::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_row_pitch = Desc.row_pitch;
  PiDesc.image_type = Desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);
  RT::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order =
      detail::convertChannelOrder(Desc.channel_order);

  return impl->ext_image_memcpy(impl, Dest, Src.value, PiDesc, PiFormat,
                                pi_image_copy_flags::PI_IMAGE_COPY_DTOH,
                                DepEvents);
}

event queue::ext_image_memcpy(void *Dest, void *Src,
                              const ext::oneapi::image_descriptor &Desc) {
  return this->ext_image_memcpy(Dest, Src, Desc, std::vector<event>{});
}

event queue::ext_image_memcpy(void *Dest, void *Src,
                              const ext::oneapi::image_descriptor &Desc,
                              event DepEvent) {
  return this->ext_image_memcpy(Dest, Src, Desc, {DepEvent});
}

event queue::ext_image_memcpy(void *Dest, void *Src,
                              const ext::oneapi::image_descriptor &Desc,
                              const std::vector<event> &DepEvents) {
  RT::PiMemImageDesc PiDesc = {};
  PiDesc.image_width = Desc.width;
  PiDesc.image_height = Desc.height;
  PiDesc.image_depth = Desc.depth;
  PiDesc.image_row_pitch = Desc.row_pitch;
  PiDesc.image_type = Desc.depth > 0 ? PI_MEM_TYPE_IMAGE3D
                                     : (Desc.height > 0 ? PI_MEM_TYPE_IMAGE2D
                                                        : PI_MEM_TYPE_IMAGE1D);
  RT::PiMemImageFormat PiFormat;
  PiFormat.image_channel_data_type =
      detail::convertChannelType(Desc.channel_type);
  PiFormat.image_channel_order =
      detail::convertChannelOrder(Desc.channel_order);

  // Flags
  pi_image_copy_flags copy_flags;
  usm::alloc dest_type = get_pointer_type(Dest, impl->get_context());
  usm::alloc src_type = get_pointer_type(Src, impl->get_context());
  if (dest_type == usm::alloc::device) {
    // Dest is on device
    if (src_type == usm::alloc::device) {
      copy_flags = pi_image_copy_flags::PI_IMAGE_COPY_DTOD;
    } else if (src_type == usm::alloc::host ||
               src_type == usm::alloc::unknown) {
      copy_flags = pi_image_copy_flags::PI_IMAGE_COPY_HTOD;
    } else {
      assert(false && "Unknown copy source location");
    }
  } else if (dest_type == usm::alloc::host ||
             dest_type == usm::alloc::unknown) {
    // Dest is on host
    if (src_type == usm::alloc::device) {
      copy_flags = pi_image_copy_flags::PI_IMAGE_COPY_DTOH;
    } else if (src_type == usm::alloc::host ||
               src_type == usm::alloc::unknown) {
      assert(false && "Cannot copy image from host to host");
    } else {
      assert(false && "Unknown copy source location");
    }
  } else {
    assert(false && "Unknown copy destination location");
  }

  return impl->ext_image_memcpy(impl, Dest, Src, PiDesc, PiFormat, copy_flags,
                                DepEvents);
}

event queue::discard_or_return(const event &Event) {
  if (!(impl->MDiscardEvents))
    return Event;
  using detail::event_impl;
  auto Impl = std::make_shared<event_impl>(event_impl::HES_Discarded);
  return detail::createSyclObjFromImpl<event>(Impl);
}

event queue::submit_impl(std::function<void(handler &)> CGH,
                         const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, CodeLoc);
}

event queue::submit_impl(std::function<void(handler &)> CGH, queue SecondQueue,
                         const detail::code_location &CodeLoc) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc);
}

event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, const detail::code_location &CodeLoc,
    const SubmitPostProcessF &PostProcess) {
  return impl->submit(CGH, impl, CodeLoc, &PostProcess);
}

event queue::submit_impl_and_postprocess(
    std::function<void(handler &)> CGH, queue SecondQueue,
    const detail::code_location &CodeLoc,
    const SubmitPostProcessF &PostProcess) {
  return impl->submit(CGH, impl, SecondQueue.impl, CodeLoc, &PostProcess);
}

void queue::wait_proxy(const detail::code_location &CodeLoc) {
  impl->wait(CodeLoc);
}

void queue::wait_and_throw_proxy(const detail::code_location &CodeLoc) {
  impl->wait_and_throw(CodeLoc);
}

template <typename Param>
typename detail::is_queue_info_desc<Param>::return_type
queue::get_info() const {
  return impl->get_info<Param>();
}

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, Picode)              \
  template __SYCL_EXPORT ReturnT queue::get_info<info::queue::Desc>() const;

#include <sycl/info/queue_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

template <typename PropertyT> bool queue::has_property() const noexcept {
  return impl->has_property<PropertyT>();
}

template <typename PropertyT> PropertyT queue::get_property() const {
  return impl->get_property<PropertyT>();
}

template __SYCL_EXPORT bool
queue::has_property<property::queue::enable_profiling>() const noexcept;
template __SYCL_EXPORT property::queue::enable_profiling
queue::get_property<property::queue::enable_profiling>() const;

template __SYCL_EXPORT bool
queue::has_property<property::queue::in_order>() const;
template __SYCL_EXPORT property::queue::in_order
queue::get_property<property::queue::in_order>() const;

bool queue::is_in_order() const {
  return impl->has_property<property::queue::in_order>();
}

backend queue::get_backend() const noexcept { return getImplBackend(impl); }

bool queue::ext_oneapi_empty() const { return impl->ext_oneapi_empty(); }

pi_native_handle queue::getNative() const { return impl->getNative(); }

buffer<detail::AssertHappened, 1> &queue::getAssertHappenedBuffer() {
  return impl->getAssertHappenedBuffer();
}

bool queue::device_has(aspect Aspect) const {
  // avoid creating sycl object from impl
  return impl->getDeviceImplPtr()->has(Aspect);
}

bool queue::ext_codeplay_supports_fusion() const {
  return impl->has_property<
      ext::codeplay::experimental::property::queue::enable_fusion>();
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
