#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {

namespace experimental::matrix {

enum class matrix_type { a, b, c };

enum class matrix_layout { row_major, col_major, packed };

// TODO deal with Conditional enable_ifs for each case if appropriate.
template <typename Group, matrix_type MT, size_t Rows = sycl::dynamic_extent,
          size_t Cols = sycl::dynamic_extent,
          matrix_layout Layout = matrix_layout::row_major>
struct joint_matrix {
  joint_matrix(Group g) {}
};

// TODO are A/B really identical in this case?
template <matrix_layout Layout>
struct joint_matrix<sub_group, matrix_type::a, 16, 16, Layout> {
  int32_t data[2]; // "s8" : "i32" and "m16n16k16:a:s8" : 2 
};

template <matrix_layout Layout>
struct joint_matrix<sub_group, matrix_type::b, 16, 16, Layout> {
  int32_t data[2]; // "s8" : "i32" and "m16n16k16:b:s8" : 2
};

template <matrix_layout Layout>
struct joint_matrix<sub_group, matrix_type::c, 16, 16, Layout> {
  int32_t data[8]; // "s32" : "i32" and "m16n16k16:c:s32" : 8
};

} // namespace experimental::matrix

namespace detail {
using namespace experimental;

template <typename Group, matrix::matrix_type MT, size_t NumRows,
          size_t NumCols, matrix::matrix_layout Layout,
          access::address_space Space>
struct joint_matrix_load_impl {
  void load(matrix::joint_matrix<Group, MT, NumRows, NumCols, Layout> &res,
            multi_ptr<int32_t, Space> src, size_t stride);
};

// Helper
template <matrix::matrix_layout Layout> const int get_layout_int() {
  static_assert(Layout == matrix::matrix_layout::row_major ||
                Layout == matrix::matrix_layout::col_major);
  return Layout == matrix::matrix_layout::row_major ? 0 : 1;
}

// TODO need to add a conditional using matrix layout.
// std::enable_if_t<Layout == matrix_layout::row_major || Layout ==
// matrix_layout::col_major, int> = 0>
template <matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl<sub_group, matrix::matrix_type::a, 16, 16, Layout,
                              Space> {
  void load(matrix::joint_matrix<sub_group, matrix::matrix_type::a, 16, 16,
                                 Layout> &res,
            multi_ptr<int32_t, Space> src, size_t stride) {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __imma_m16n16k16_ld_a_s8(res.data, src.get(), stride, 0);
#endif
#endif
  }
};

template <matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl<sub_group, matrix::matrix_type::b, 16, 16, Layout,
                              Space> {
  void load(matrix::joint_matrix<sub_group, matrix::matrix_type::b, 16, 16,
                                 Layout> &res,
            multi_ptr<int32_t, Space> src, size_t stride) {
    // const int i = 0;
    // const int* ll;
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __imma_m16n16k16_ld_b_s8(res.data, src.get(), stride, 1);
#endif
#endif  
  }
};

// get_layout_int<Layout>() replaced with 0 in final buildin argument;
template <matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl<sub_group, matrix::matrix_type::c, 16, 16, Layout,
                              Space> {
  void load(matrix::joint_matrix<sub_group, matrix::matrix_type::c, 16, 16,
                                 Layout> &res,
            multi_ptr<int32_t, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __imma_m16n16k16_ld_c(res.data, src.get(), stride, 0);
#endif
#endif
    
  }
};

template <typename Group, size_t NumRows, size_t NumCols,
          matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_store_impl {
  void store(matrix::joint_matrix<Group, matrix::matrix_type::c, NumRows,
                                  NumCols, Layout> &src,
             multi_ptr<int32_t, Space> dst, size_t stride);
};

// TODO currently assuming row order.
template <matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_store_impl<sub_group, 16, 16, Layout, Space> {
  void store(matrix::joint_matrix<sub_group, matrix::matrix_type::c, 16, 16,
                                  Layout> &src,
             multi_ptr<int32_t, Space> dst, size_t stride) {
    
            #ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __imma_m16n16k16_st_c_i32(dst, src.data, stride, 0);
#endif
#endif
  }
};
template <typename Group, std::size_t M, std::size_t K, std::size_t N,
          matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
          matrix::matrix_layout LayoutC>
struct joint_matrix_mma_impl {
  matrix::joint_matrix<Group, matrix::matrix_type::c, M, N, LayoutC>
  mma(Group sg,
      matrix::joint_matrix<Group, matrix::matrix_type::a, M, K, LayoutA> A,
      matrix::joint_matrix<Group, matrix::matrix_type::b, K, N, LayoutB> B,
      matrix::joint_matrix<Group, matrix::matrix_type::c, M, N, LayoutC> C);
};

// Helper
template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB>
int get_layout_comb_int() {
  static_assert(LayoutA == matrix::matrix_layout::row_major ||
                LayoutA == matrix::matrix_layout::col_major);
  static_assert(LayoutB == matrix::matrix_layout::row_major ||
                LayoutB == matrix::matrix_layout::col_major);
  if (LayoutA == matrix::matrix_layout::row_major)
    return LayoutB == matrix::matrix_layout::row_major ? 0 : 1;
  // LayoutA == matrix_layout::col_major must hold here
  return LayoutB == matrix::matrix_layout::row_major ? 2 : 3;
}

template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
          matrix::matrix_layout LayoutC>
struct joint_matrix_mma_impl<sub_group, 16, 16, 16, LayoutA, LayoutB, LayoutC> {
  matrix::joint_matrix<sub_group, matrix::matrix_type::c, 16, 16, LayoutC>
  mma(sub_group sg,
      matrix::joint_matrix<sub_group, matrix::matrix_type::a, 16, 16, LayoutA>
          A,
      matrix::joint_matrix<sub_group, matrix::matrix_type::b, 16, 16, LayoutB>
          B,
      matrix::joint_matrix<sub_group, matrix::matrix_type::c, 16, 16, LayoutC>
          C) {
    matrix::joint_matrix<sub_group, matrix::matrix_type::c, 16, 16, LayoutC> D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __imma_m16n16k16_mma_s8(D.data, A.data, B.data, C.data, 1, 0);
#endif
#endif
    
    return D;
  }
};

} // namespace detail

namespace experimental::matrix {
template <typename Group, matrix_type MT, size_t NumRows, size_t NumCols,
          matrix_layout Layout, access::address_space Space>
void joint_matrix_load(Group sg,
                       joint_matrix<Group, MT, NumRows, NumCols, Layout> &res,
                       multi_ptr<int32_t, Space> src, size_t stride,
                       matrix_layout layout = matrix_layout::row_major) {
  // Should be a requirement for the CUDA backend
  assert(layout == Layout);
  detail::joint_matrix_load_impl<Group, MT, NumRows, NumCols, Layout, Space>{}
      .load(res, src, stride);
}

template <typename Group, size_t NumRows, size_t NumCols, matrix_layout Layout,
          access::address_space Space>
void joint_matrix_store(
    Group sg,
    joint_matrix<Group, matrix_type::c, NumRows, NumCols, Layout> &src,
    multi_ptr<int32_t, Space> dst, size_t stride,
    matrix_layout layout = matrix_layout::row_major) {
  // Should be a requirement for the CUDA backend
  assert(layout == Layout);
  detail::joint_matrix_store_impl<Group, NumRows, NumCols, Layout, Space>{}
      .store(src, dst, stride);
}

template <typename Group, std::size_t M, std::size_t K, std::size_t N,
          matrix_layout LayoutA, matrix_layout LayoutB, matrix_layout LayoutC>
joint_matrix<Group, matrix_type::c, M, N, LayoutC>
joint_matrix_mad(Group sg, joint_matrix<Group, matrix_type::a, M, K, LayoutA> A,
                 joint_matrix<Group, matrix_type::b, K, N, LayoutB> B,
                 joint_matrix<Group, matrix_type::c, M, N, LayoutC> C) {
  return detail::joint_matrix_mma_impl<Group, M, K, N, LayoutA, LayoutB,
                                       LayoutC>{}
      .mma(sg, A, B, C);
}
} // namespace experimental::matrix

} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
