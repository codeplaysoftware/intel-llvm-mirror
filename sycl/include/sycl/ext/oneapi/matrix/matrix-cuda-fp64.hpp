#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {

namespace experimental::matrix {

enum class matrix_type { a, b, c};

//packed layout not yet implemented. // col_major works fine via builtins but enum usage leads to strange error.
enum class matrix_layout { row_major, col_major, packed };

// TODO deal with Conditional enable_ifs for each case if appropriate: packed type is fundamentally different.
// Note that the Group typename is never used (it is only used in the JIT AMX implementation)
template <typename Group, matrix_type MT, matrix_layout Layout, size_t Rows = sycl::dynamic_extent,
          size_t Cols = sycl::dynamic_extent>
struct joint_matrix {
  joint_matrix(Group g){}
};

//TODO need to assert layout is col or row type.
template <matrix_layout Layout>
struct joint_matrix<sub_group, matrix_type::a, Layout, 8, 4> {
  double data[1]; 
};

template <matrix_layout Layout>
struct joint_matrix<sub_group, matrix_type::b, Layout, 4, 8> {
  double data[1]; 
};

template <matrix_layout Layout>
struct joint_matrix<sub_group, matrix_type::c, Layout, 8, 8> {
  double data[2]; 
};

} // namespace experimental::matrix

namespace detail {
using namespace experimental;

template <typename Group, matrix::matrix_type MT, size_t NumRows,
          size_t NumCols, matrix::matrix_layout Layout,
          access::address_space Space>
struct joint_matrix_load_impl {
  void load(matrix::joint_matrix<Group, MT, Layout, NumRows, NumCols> &res,
            multi_ptr<double, Space> src, size_t stride);
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
struct joint_matrix_load_impl<sub_group, matrix::matrix_type::a, 8, 4, Layout,
                              Space> {
  void load(matrix::joint_matrix<sub_group, matrix::matrix_type::a, Layout, 8, 4> &res,
            multi_ptr<double, Space> src, size_t stride) {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
if (Layout == matrix::matrix_layout::row_major)
    __dmma_m8n8k4_ld_a(res.data, src.get(), stride, 0);
   else
     __dmma_m8n8k4_ld_a(res.data, src.get(), stride, 1);
#endif
#endif
  }
};

template <matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl<sub_group, matrix::matrix_type::b, 4, 8, Layout,
                              Space> {
  void load(matrix::joint_matrix<sub_group, matrix::matrix_type::b, Layout, 4, 8
                                 > &res,
            multi_ptr<double, Space> src, size_t stride) {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
if (Layout == matrix::matrix_layout::row_major)
 __dmma_m8n8k4_ld_b(res.data, src.get(), stride, 0);
 else
    __dmma_m8n8k4_ld_b(res.data, src.get(), stride, 1);
#endif
#endif  
  }
};


template <matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_load_impl<sub_group, matrix::matrix_type::c, 8, 8, Layout,
                              Space> {
  void load(matrix::joint_matrix<sub_group, matrix::matrix_type::c, Layout, 8, 8
                                 > &res,
            multi_ptr<double, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
if (Layout == matrix::matrix_layout::row_major)
     __dmma_m8n8k4_ld_c(res.data, src.get(), stride, 0);
     else
     __dmma_m8n8k4_ld_c(res.data, src.get(), stride, 1);
#endif
#endif
    
  }
};

template <typename Group, size_t NumRows, size_t NumCols,
          matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_store_impl {
  void store(matrix::joint_matrix<Group, matrix::matrix_type::c, Layout, NumRows,
                                  NumCols> &src,
             multi_ptr<double, Space> dst, size_t stride);
};

// TODO currently assuming row order.
template <matrix::matrix_layout Layout, access::address_space Space>
struct joint_matrix_store_impl<sub_group, 8, 8, Layout, Space> {
  void store(matrix::joint_matrix<sub_group, matrix::matrix_type::c, Layout, 8, 8
                                  > &src,
             multi_ptr<double, Space> dst, size_t stride) {
    
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
if (Layout == matrix::matrix_layout::row_major)
    __dmma_m8n8k4_st_c_f64(dst.get(), src.data, stride, 0);
    else
    __dmma_m8n8k4_st_c_f64(dst.get(), src.data, stride, 1);
#endif
#endif
  }
};
template <typename Group, std::size_t M, std::size_t K, std::size_t N,
          matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
          matrix::matrix_layout LayoutC>
struct joint_matrix_mma_impl {
  matrix::joint_matrix<Group, matrix::matrix_type::c, LayoutC, M, N>
  mma(Group sg,
      matrix::joint_matrix<Group, matrix::matrix_type::a, LayoutA, M, K> A,
      matrix::joint_matrix<Group, matrix::matrix_type::b, LayoutB, K, N> B,
      matrix::joint_matrix<Group, matrix::matrix_type::c, LayoutC, M, N> C);
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
struct joint_matrix_mma_impl<sub_group, 8, 4, 8, LayoutA, LayoutB, LayoutC> {
  matrix::joint_matrix<sub_group, matrix::matrix_type::c, LayoutC, 8, 8>
  mma(sub_group sg,
      matrix::joint_matrix<sub_group, matrix::matrix_type::a, LayoutA, 8, 4>
          A,
      matrix::joint_matrix<sub_group, matrix::matrix_type::b, LayoutB, 4, 8>
          B,
      matrix::joint_matrix<sub_group, matrix::matrix_type::c, LayoutC, 8, 8>
          C) {
    matrix::joint_matrix<sub_group, matrix::matrix_type::c, LayoutC, 8, 8> D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    //__imma_m16n16k16_mma_s8(D.data, A.data, B.data, C.data, 1, 0);
    if (LayoutA == matrix::matrix_layout::row_major)
    {
      if(LayoutB == matrix::matrix_layout::row_major)
    __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, 0, 0);
    else
    __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, 1, 0);
    }
    else
    {
     if (LayoutB == matrix::matrix_layout::row_major)
     __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, 2, 0);
     else
      __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, 3, 0);
    }
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
                       joint_matrix<Group, MT, Layout, NumRows, NumCols> &res,
                       multi_ptr<double, Space> src, size_t stride,
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
    joint_matrix<Group, matrix_type::c, Layout, NumRows, NumCols> &src,
    multi_ptr<double, Space> dst, size_t stride,
    matrix_layout layout = matrix_layout::row_major) {
  // Should be a requirement for the CUDA backend
  assert(layout == Layout); //todo: improve error message!!
  detail::joint_matrix_store_impl<Group, NumRows, NumCols, Layout, Space>{}
      .store(src, dst, stride);
}

template <typename Group, std::size_t M, std::size_t K, std::size_t N,
          matrix_layout LayoutA, matrix_layout LayoutB, matrix_layout LayoutC>
joint_matrix<Group, matrix_type::c, LayoutC, M, N>
joint_matrix_mad(Group sg, joint_matrix<Group, matrix_type::a, LayoutA, M, K> A,
                 joint_matrix<Group, matrix_type::b, LayoutB, K, N> B,
                 joint_matrix<Group, matrix_type::c, LayoutC, M, N> C) {
  return detail::joint_matrix_mma_impl<Group, M, K, N, LayoutA, LayoutB,
                                       LayoutC>{}.mma(sg, A, B, C);
}
} // namespace experimental::matrix

} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
