//==------ native_macros.hpp - SYCL math built-ins with native macros ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/builtins.hpp>
#include <CL/sycl/detail/generic_type_lists.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>


__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

using floating_ex_float_list =
    type_list<detail::gtl::half_list, detail::gtl::double_list>;

template <typename T>
using is_genfloatxf = is_contained<T, floating_ex_float_list>;

// genfloat cos (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> cos(T x) __NOEXC {
  return __sycl_std::__invoke_cos<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> cos(T x) __NOEXC {
#ifdef SYCL_NATIVE_COS
  return __sycl_std::__invoke_native_cos<T>(x);
#else
  return __sycl_std::__invoke_cos<T>(x);
#endif
}

// genfloat exp (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> exp(T x) __NOEXC {
  return __sycl_std::__invoke_exp<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp(T x) __NOEXC {
#ifdef SYCL_NATIVE_EXP
  return __sycl_std::__invoke_native_exp<T>(x);
#else
  return __sycl_std::__invoke_exp<T>(x);
#endif
}

// genfloat exp2 (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> exp2(T x) __NOEXC {
  return __sycl_std::__invoke_exp2<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp2(T x) __NOEXC {
#ifdef SYCL_NATIVE_EXP2
  return __sycl_std::__invoke_native_exp2<T>(x);
#else
  return __sycl_std::__invoke_exp2<T>(x);
#endif
}

// genfloat exp10 (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> exp10(T x) __NOEXC {
  return __sycl_std::__invoke_exp10<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> exp10(T x) __NOEXC {
#ifdef SYCL_NATIVE_EXP10
  return __sycl_std::__invoke_native_exp10<T>(x);
#else
  return __sycl_std::__invoke_exp10<T>(x);
#endif
}

// genfloat log (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> log(T x) __NOEXC {
  return __sycl_std::__invoke_log<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log(T x) __NOEXC {
#ifdef SYCL_NATIVE_LOG
  return __sycl_std::__invoke_native_log<T>(x);
#else
  return __sycl_std::__invoke_log<T>(x);
#endif
}

// genfloat log2 (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> log2(T x) __NOEXC {
  return __sycl_std::__invoke_log2<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log2(T x) __NOEXC {
#ifdef SYCL_NATIVE_LOG2
  return __sycl_std::__invoke_native_log2<T>(x);
#else
  return __sycl_std::__invoke_log2<T>(x);
#endif
}

// genfloat log10 (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> log10(T x) __NOEXC {
  return __sycl_std::__invoke_log10<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> log10(T x) __NOEXC {
#ifdef SYCL_NATIVE_LOG10
  return __sycl_std::__invoke_native_log10<T>(x);
#else
  return __sycl_std::__invoke_log10<T>(x);
#endif
}

// genfloat rsqrt (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> rsqrt(T x) __NOEXC {
  return __sycl_std::__invoke_rsqrt<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> rsqrt(T x) __NOEXC {
#ifdef SYCL_NATIVE_RSQRT
  return __sycl_std::__invoke_native_rsqrt<T>(x);
#else
  return __sycl_std::__invoke_rsqrt<T>(x);
#endif
}

// genfloat sin (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> sin(T x) __NOEXC {
  return __sycl_std::__invoke_sin<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> sin(T x) __NOEXC {
#ifdef SYCL_NATIVE_SIN
  return __sycl_std::__invoke_native_sin<T>(x);
#else
  return __sycl_std::__invoke_sin<T>(x);
#endif
}

// genfloat sqrt (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> sqrt(T x) __NOEXC {
  return __sycl_std::__invoke_sqrt<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> sqrt(T x) __NOEXC {
#ifdef SYCL_NATIVE_SQRT
  return __sycl_std::__invoke_native_sqrt<T>(x);
#else
  return __sycl_std::__invoke_sqrt<T>(x);
#endif
}

// genfloat tan (genfloat x)
template <typename T>
detail::enable_if_t<is_genfloatxf<T>::value, T> tan(T x) __NOEXC {
  return __sycl_std::__invoke_tan<T>(x);
}

template <typename T>
detail::enable_if_t<detail::is_genfloatf<T>::value, T> tan(T x) __NOEXC {
#ifdef SYCL_NATIVE_TAN
  return __sycl_std::__invoke_native_tan<T>(x);
#else
  return __sycl_std::__invoke_tan<T>(x);
#endif
}
}

} // namespace experimental
} // namespace oneapi
} // namespace ext

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

