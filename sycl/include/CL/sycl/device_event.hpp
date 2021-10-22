//==----------- device_event.hpp --- SYCL device event ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace detail {
template <typename G> struct group_execution_scope {};
template <> struct group_execution_scope<sycl::ext::oneapi::sub_group> {
  constexpr static auto Scope = __spv::Scope::Subgroup;
};
template <int D> struct group_execution_scope<sycl::group<D>> {
  constexpr static auto Scope = __spv::Scope::Workgroup;
};
} // namespace detail

/// Encapsulates a single SYCL device event which is available only within SYCL
/// kernel functions and can be used to wait for asynchronous operations within
/// a kernel function to complete.
///
/// \ingroup sycl_api
class device_event {
private:
  __ocl_event_t *m_Event;

public:
  device_event(const device_event &rhs) = default;
  device_event(device_event &&rhs) = default;
  device_event &operator=(const device_event &rhs) = default;
  device_event &operator=(device_event &&rhs) = default;

  device_event(__ocl_event_t *Event) : m_Event(Event) {}

  template <typename Group>
  void wait(Group) {
    __spirv_GroupWaitEvents(detail::group_execution_scope<Group>::Scope, 1, m_Event);
  }
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
