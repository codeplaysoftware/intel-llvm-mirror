#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
// namespace detail {

namespace experimental::matrix {

enum class matrix_type { a, b, c };

enum class matrix_layout { row_major, col_major, packed };

// todo deal with Conditional enable_ifs for each case if appropriate.
template <typename Group, typename T, matrix_type MT,
          size_t Rows = sycl::dynamic_extent,
          size_t Cols = sycl::dynamic_extent,
          matrix_layout Layout = matrix_layout::row_major>
struct joint_matrix {
  joint_matrix(Group g) {}
};

template <matrix_layout Layout>
struct joint_matrix<sub_group, int8_t, matrix_type::a, 16, 16, Layout> {
  int32_t data[2]; // "s8" : "i32" and "m16n16k16:a:s8" : 2
};

template <matrix_layout Layout>
struct joint_matrix<sub_group, int8_t, matrix_type::b, 16, 16, Layout> {
  int32_t data[2]; // "s8" : "i32" and "m16n16k16:b:s8" : 2
};

template <matrix_layout Layout>
struct joint_matrix<sub_group, int, matrix_type::c, 16, 16, Layout> {
  int32_t data[8]; // "s32" : "i32" and "m16n16k16:c:s32" : 8
};

template <typename Group, typename T, matrix_type MT, size_t NumRows,
          size_t NumCols, matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl {
  void load(joint_matrix<Group, T, MT, NumRows, NumCols, Layout> &res,
            multi_ptr<T, Space> src, size_t stride);
};

template <matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl<sub_group, int8_t, matrix_type::a, 16, 16, Layout,
                              Space> {
  void
  load(joint_matrix<sub_group, int8_t, matrix_type::a, 16, 16, Layout> &res,
       multi_ptr<int8_t, Space> src, size_t stride) {
    __imma_m16n16k16_ld_a_s8(res.data, src, stride, 0);
  }
};

template <matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl<sub_group, int8_t, matrix_type::b, 16, 16, Layout,
                              Space> {
  void
  load(joint_matrix<sub_group, int8_t, matrix_type::b, 16, 16, Layout> &res,
       multi_ptr<int8_t, Space> src, size_t stride) {
    __imma_m16n16k16_ld_b_s8(res.data, src, stride, 0);
  }
};

// get_layout_int<Layout>() replaced with 0 in final buildin argument;
template <matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl<sub_group, int32_t, matrix_type::c, 16, 16,
                              Layout, Space> {
  void
  load(joint_matrix<sub_group, int32_t, matrix_type::c, 16, 16, Layout> &res,
       multi_ptr<int32_t, Space> src, size_t stride) {
    __imma_m16n16k16_ld_c(res.data, src, stride, 0);
  }
};

template <typename Group, typename T, matrix_type MT, size_t NumRows,
          size_t NumCols, matrix_layout Layout, access::address_space Space>
void joint_matrix_load(
    Group sg, joint_matrix<Group, T, MT, NumRows, NumCols, Layout> &res,
    multi_ptr<T, Space> src, size_t stride,
    matrix_layout layout = matrix_layout::row_major) {
  // Should be a requirement for the CUDA backend
  assert(layout == Layout);
  joint_matrix_load_impl<Group, T, MT, NumRows, NumCols, Layout, Space>{}.load(
      res, src, stride);
}

} // namespace experimental::matrix
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
