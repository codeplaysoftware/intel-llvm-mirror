#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl) {
  namespace sycl {
  namespace ext {
  namespace oneapi {
  namespace experimental::matrix {

  enum class matrix_use { a, b, accumulator };

  enum class matrix_layout { row_major, col_major, packed_a, packed_b };

  enum class matrix_type {
    bf8,
    bf16,
    fp16,
    fp19, // tfloat32
    fp32,
    fp64,
    sint2,
    sint4,
    sint8,
    sint16,
    sint32,
    sint64,
    uint2,
    uint4,
    uint8,
    uint16,
    uint32,
    uint64
  };

  template <matrix_type T, matrix_use MT, size_t Rows = sycl::dynamic_extent,
            size_t Cols = sycl::dynamic_extent,
            matrix_layout Layout = matrix_layout::row_major,
            typename Group = sycl::sub_group, typename Cond = void>
  struct joint_matrix {
    joint_matrix(Group g) {}
  };

  // The enable_if_t usage in this file is used to disable the
  // matrix_layout::packed case which is not compatible with the Nvidia cuda
  // backend.

#define __SYCL_JOINT_MATRIX_OVERLOAD(type, use, M, N, frag_type, frag_size) \
  template <matrix_layout Layout>                                                  \
  struct joint_matrix<                                                             \
      matrix_type::type, matrix_use::use, M, N, Layout, sycl::sub_group,                            \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||              \
                                Layout == matrix_layout::col_major>> {             \
    frag_type data[frag_size];                                                     \
  };

// m8n8k4 double
__SYCL_JOINT_MATRIX_OVERLOAD(fp64, a, 8, 4, double, 1)
__SYCL_JOINT_MATRIX_OVERLOAD(fp64, b, 4, 8, double, 1)
__SYCL_JOINT_MATRIX_OVERLOAD(fp64, accumulator, 8, 8, double, 2)

// m8n32k16 bf16 //todo: will there be bf16 object.. introduced
__SYCL_JOINT_MATRIX_OVERLOAD(bf16, a, 8, 16, int32_t, 2)
__SYCL_JOINT_MATRIX_OVERLOAD(bf16, b, 16, 32, int32_t, 8)
__SYCL_JOINT_MATRIX_OVERLOAD(fp32, accumulator, 8, 32, float, 8)

// m16n16k16
__SYCL_JOINT_MATRIX_OVERLOAD(bf16, a, 16, 16, int32_t, 4)
__SYCL_JOINT_MATRIX_OVERLOAD(bf16, b, 16, 16, int32_t, 4)
__SYCL_JOINT_MATRIX_OVERLOAD(fp32, accumulator, 16, 16, float, 8)

__SYCL_JOINT_MATRIX_OVERLOAD(sint8, a, 16, 16, int32_t, 2)
__SYCL_JOINT_MATRIX_OVERLOAD(sint8, b, 16, 16, int32_t, 2)
__SYCL_JOINT_MATRIX_OVERLOAD(uint8, a, 16, 16, int32_t, 2)
__SYCL_JOINT_MATRIX_OVERLOAD(uint8, b, 16, 16, int32_t, 2)
__SYCL_JOINT_MATRIX_OVERLOAD(sint32, accumulator, 16, 16, int32_t, 8)


#undef __SYCL_JOINT_MATRIX_OVERLOAD
  } // namespace experimental::matrix

  namespace detail {
  using namespace experimental;

  template <typename U, matrix::matrix_type T, matrix::matrix_use MT,
            size_t NumRows, size_t NumCols, matrix::matrix_layout Layout,
            access::address_space Space, typename Cond = void>
  struct joint_matrix_load_impl {
    void load(matrix::joint_matrix<T, MT, NumRows, NumCols, Layout> &res,
              multi_ptr<U, Space> src, size_t stride);
  };

  template <matrix::matrix_layout Layout> constexpr int get_layout_id();

  template <> constexpr int get_layout_id<matrix::matrix_layout::row_major>() {
    return 0;
  }

  template <> constexpr int get_layout_id<matrix::matrix_layout::col_major>() {
    return 1;
  }

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      double, matrix::matrix_type::fp64, matrix::matrix_use::a, 8, 4, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::fp64,
                                   matrix::matrix_use::a, 8, 4, Layout> &res,
              multi_ptr<double, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __dmma_m8n8k4_ld_a(res.data, src.get(), stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

  

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      float, matrix::matrix_type::fp32, matrix::matrix_use::accumulator, 16, 16,
      Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::fp32,
                                   matrix::matrix_use::accumulator, 16, 16,
                                   Layout> &res,
              multi_ptr<float, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __hmma_m16n16k16_ld_c_f32(res.data, src.get(), stride,
                                get_layout_id<Layout>());
#endif
#endif
    }
  };

#ifdef __NVPTX__                                                                        
#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_JOINT_MATRIX_LOAD_A(OP) __##OP##_ld_a(res.data, tileptr, stride, get_layout_id<Layout>())
#endif                                                                                  
#endif

//__SYCL_JOINT_MATRIX_LOAD_A(mma_bf16_m8n32k16)

#define __SYCL_JOINT_MATRIX_LOAD_OVERLOAD(type, use, M, N, element_type, frag_type, OP)             \
template <matrix::matrix_layout Layout, access::address_space Space>                    \
  struct joint_matrix_load_impl<                                                        \
      element_type, matrix::matrix_type::type, matrix::matrix_use::use, M, N, Layout,  \
      Space,                                                                            \
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||           \
                                Layout == matrix::matrix_layout::col_major>> {          \
    void load(matrix::joint_matrix<matrix::matrix_type::type,                           \
                                   matrix::matrix_use::use, M, N, Layout> &res,          \
              multi_ptr<element_type, Space> src, size_t stride) {                    \
frag_type* tileptr = reinterpret_cast<frag_type*>(src.get());                               \
      __SYCL_JOINT_MATRIX_LOAD_A(OP);                          \                                                                              
    }                                                                                   \
  };

 
 #ifdef __NVPTX__                                                                        
#ifdef __SYCL_DEVICE_ONLY__
  __SYCL_JOINT_MATRIX_LOAD_OVERLOAD(bf16, a, 8, 16, unsigned short, int32_t, mma_bf16_m8n32k16)
  
    #endif                                                                                  
#endif
  #undef __SYCL_JOINT_MATRIX_OVERLOAD

//#define mma_bf16_m8n32k16;
// m8n32k16





/*template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      unsigned short, matrix::matrix_type::bf16, matrix::matrix_use::a, 8, 16, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::bf16,
                                   matrix::matrix_use::a, 8, 16, Layout> &res,
              multi_ptr<unsigned short, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
int32_t* tileptr = reinterpret_cast<int32_t*>(src.get()); 
      __mma_bf16_m8n32k16_ld_a(res.data, tileptr, stride, get_layout_id<Layout>());
      __mma_bf16_m8n32k16_ld_a
#endif
#endif
    }
  };*/

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      unsigned short, matrix::matrix_type::bf16, matrix::matrix_use::b, 16, 32, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::bf16,
                                   matrix::matrix_use::b, 16, 32, Layout> &res,
              multi_ptr<unsigned short, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
int32_t* tileptr = reinterpret_cast<int32_t*>(src.get()); 
      __mma_bf16_m8n32k16_ld_b(res.data, tileptr, stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

    template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      float, matrix::matrix_type::fp32, matrix::matrix_use::accumulator, 8, 32,
      Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::fp32,
                                   matrix::matrix_use::accumulator, 8, 32,
                                   Layout> &res,
              multi_ptr<float, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __hmma_m8n32k16_ld_c_f32(res.data, src.get(), stride,
                                get_layout_id<Layout>());
#endif
#endif
    }
  };

template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      unsigned short, matrix::matrix_type::bf16, matrix::matrix_use::a, 16, 16, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::bf16,
                                   matrix::matrix_use::a, 16, 16, Layout> &res,
              multi_ptr<unsigned short, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
int32_t* tileptr = reinterpret_cast<int32_t*>(src.get()); 
      __mma_bf16_m16n16k16_ld_a(res.data, tileptr, stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      unsigned short, matrix::matrix_type::bf16, matrix::matrix_use::b, 16, 16, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::bf16,
                                   matrix::matrix_use::b, 16, 16, Layout> &res,
              multi_ptr<unsigned short, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
int32_t* tileptr = reinterpret_cast<int32_t*>(src.get()); 
      __mma_bf16_m16n16k16_ld_b(res.data, tileptr, stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      int8_t, matrix::matrix_type::sint8, matrix::matrix_use::a, 16, 16, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::sint8,
                                   matrix::matrix_use::a, 16, 16, Layout> &res,
              multi_ptr<int8_t, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
int32_t* tileptr = reinterpret_cast<int32_t*>(src.get());
      __imma_m16n16k16_ld_a_s8(res.data, tileptr, stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      int8_t, matrix::matrix_type::sint8, matrix::matrix_use::b, 16, 16, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::sint8,
                                   matrix::matrix_use::b, 16, 16, Layout> &res,
              multi_ptr<int8_t, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
int32_t* tileptr = reinterpret_cast<int32_t*>(src.get());
      __imma_m16n16k16_ld_b_s8(res.data, tileptr, stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      uint8_t, matrix::matrix_type::uint8, matrix::matrix_use::a, 16, 16, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::uint8,
                                   matrix::matrix_use::a, 16, 16, Layout> &res,
              multi_ptr<uint8_t, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
int32_t* tileptr = reinterpret_cast<int32_t*>(src.get());
      __imma_m16n16k16_ld_a_u8(res.data, tileptr, stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      uint8_t, matrix::matrix_type::uint8, matrix::matrix_use::b, 16, 16, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::uint8,
                                   matrix::matrix_use::b, 16, 16, Layout> &res,
              multi_ptr<uint8_t, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
int32_t* tileptr = reinterpret_cast<int32_t*>(src.get());
      __imma_m16n16k16_ld_b_u8(res.data, tileptr, stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      int, matrix::matrix_type::sint32, matrix::matrix_use::accumulator, 16, 16,
      Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::sint32,
                                   matrix::matrix_use::accumulator, 16, 16,
                                   Layout> &res,
              multi_ptr<int32_t, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
//uint32_t* tileptr = reinterpret_cast<uint32_t*>(src.get());
      __imma_m16n16k16_ld_c(res.data, src.get(), stride,
                                get_layout_id<Layout>());
#endif
#endif
    }
  };


  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      double, matrix::matrix_type::fp64, matrix::matrix_use::b, 4, 8, Layout,
      Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::fp64,
                                   matrix::matrix_use::b, 4, 8, Layout> &res,
              multi_ptr<double, Space> src, size_t stride) {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __dmma_m8n8k4_ld_b(res.data, src.get(), stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      double, matrix::matrix_type::fp64, matrix::matrix_use::accumulator, 8, 8,
      Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<matrix::matrix_type::fp64,
                                   matrix::matrix_use::accumulator, 8, 8,
                                   Layout> &res,
              multi_ptr<double, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __dmma_m8n8k4_ld_c(res.data, src.get(), stride, get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <typename U, matrix::matrix_type T, size_t NumRows, size_t NumCols,
            matrix::matrix_layout Layout, access::address_space Space,
            typename Cond = void>
  struct joint_matrix_store_impl {
    void store(matrix::joint_matrix<T, matrix::matrix_use::accumulator, NumRows,
                                    NumCols, Layout> &src,
               multi_ptr<U, Space> dst, size_t stride);
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_store_impl<
      double, matrix::matrix_type::fp64, 8, 8, Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void store(matrix::joint_matrix<matrix::matrix_type::fp64,
                                    matrix::matrix_use::accumulator, 8, 8,
                                    Layout> &src,
               multi_ptr<double, Space> dst, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __dmma_m8n8k4_st_c_f64(dst.get(), src.data, stride,
                             get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_store_impl<
      float, matrix::matrix_type::fp32, 16, 16, Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void store(matrix::joint_matrix<matrix::matrix_type::fp32,
                                    matrix::matrix_use::accumulator, 16, 16,
                                    Layout> &src,
               multi_ptr<float, Space> dst, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __hmma_m16n16k16_st_c_f32(dst.get(), src.data, stride,
                                get_layout_id<Layout>());
#endif
#endif
    }
  };

    template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_store_impl<
      float, matrix::matrix_type::fp32, 8, 32, Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void store(matrix::joint_matrix<matrix::matrix_type::fp32,
                                    matrix::matrix_use::accumulator, 8, 32,
                                    Layout> &src,
               multi_ptr<float, Space> dst, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __hmma_m8n32k16_st_c_f32(dst.get(), src.data, stride,
                                get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_store_impl<
      int32_t, matrix::matrix_type::sint32, 16, 16, Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void store(matrix::joint_matrix<matrix::matrix_type::sint32,
                                    matrix::matrix_use::accumulator, 16, 16,
                                    Layout> &src,
               multi_ptr<int32_t, Space> dst, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __imma_m16n16k16_st_c_i32(dst.get(), src.data, stride,
                                get_layout_id<Layout>());
#endif
#endif
    }
  };

  template <matrix::matrix_type T1, matrix::matrix_type T2, std::size_t M,
            std::size_t K, std::size_t N, matrix::matrix_layout LayoutA,
            matrix::matrix_layout LayoutB, matrix::matrix_layout LayoutC,
            typename Cond = void>
  struct joint_matrix_mad_impl {
    matrix::joint_matrix<T2, matrix::matrix_use::accumulator, M, N, LayoutC>
    mad(matrix::joint_matrix<T1, matrix::matrix_use::a, M, K, LayoutA> A,
        matrix::joint_matrix<T1, matrix::matrix_use::b, K, N, LayoutB> B,
        matrix::joint_matrix<T2, matrix::matrix_use::accumulator, M, N, LayoutC>
            C);
  };

  template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB>
  constexpr int get_layout_pair_id();

  template <>
  constexpr int get_layout_pair_id<matrix::matrix_layout::row_major,
                                   matrix::matrix_layout::row_major>() {
    return 0;
  }

  template <>
  constexpr int get_layout_pair_id<matrix::matrix_layout::row_major,
                                   matrix::matrix_layout::col_major>() {
    return 1;
  }

  template <>
  constexpr int get_layout_pair_id<matrix::matrix_layout::col_major,
                                   matrix::matrix_layout::row_major>() {
    return 2;
  }

  template <>
  constexpr int get_layout_pair_id<matrix::matrix_layout::col_major,
                                   matrix::matrix_layout::col_major>() {
    return 3;
  }

  template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
            matrix::matrix_layout LayoutC>
  struct joint_matrix_mad_impl<
      matrix::matrix_type::fp64, matrix::matrix_type::fp64, 8, 4, 8, LayoutA,
      LayoutB, LayoutC,
      typename std::enable_if_t<(LayoutA == matrix::matrix_layout::row_major ||
                                 LayoutA == matrix::matrix_layout::col_major) &&
                                (LayoutB == matrix::matrix_layout::row_major ||
                                 LayoutB == matrix::matrix_layout::col_major) &&
                                (LayoutC == matrix::matrix_layout::row_major ||
                                 LayoutC ==
                                     matrix::matrix_layout::col_major)>> {
    matrix::joint_matrix<matrix::matrix_type::fp64,
                         matrix::matrix_use::accumulator, 8, 8, LayoutC>
    mad(matrix::joint_matrix<matrix::matrix_type::fp64, matrix::matrix_use::a,
                             8, 4, LayoutA>
            A,
        matrix::joint_matrix<matrix::matrix_type::fp64, matrix::matrix_use::b,
                             4, 8, LayoutB>
            B,
        matrix::joint_matrix<matrix::matrix_type::fp64,
                             matrix::matrix_use::accumulator, 8, 8, LayoutC>
            C) {
      matrix::joint_matrix<matrix::matrix_type::fp64,
                           matrix::matrix_use::accumulator, 8, 8, LayoutC>
          D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data,
                            get_layout_pair_id<LayoutA, LayoutB>(), 0);
#endif
#endif

      return D;
    }
  };

  template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
            matrix::matrix_layout LayoutC>
  struct joint_matrix_mad_impl<
      matrix::matrix_type::bf16, matrix::matrix_type::fp32, 16, 16, 16, LayoutA,
      LayoutB, LayoutC,
      typename std::enable_if_t<(LayoutA == matrix::matrix_layout::row_major ||
                                 LayoutA == matrix::matrix_layout::col_major) &&
                                (LayoutB == matrix::matrix_layout::row_major ||
                                 LayoutB == matrix::matrix_layout::col_major) &&
                                (LayoutC == matrix::matrix_layout::row_major ||
                                 LayoutC ==
                                     matrix::matrix_layout::col_major)>> {
    matrix::joint_matrix<matrix::matrix_type::fp32,
                         matrix::matrix_use::accumulator, 16, 16, LayoutC>
    mad(matrix::joint_matrix<matrix::matrix_type::bf16, matrix::matrix_use::a,
                             16, 16, LayoutA>
            A,
        matrix::joint_matrix<matrix::matrix_type::bf16, matrix::matrix_use::b,
                             16, 16, LayoutB>
            B,
        matrix::joint_matrix<matrix::matrix_type::fp32,
                             matrix::matrix_use::accumulator, 16, 16, LayoutC>
            C) {
      matrix::joint_matrix<matrix::matrix_type::fp32,
                           matrix::matrix_use::accumulator, 16, 16, LayoutC>
          D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __mma_bf16_m16n16k16_mma_f32(D.data, A.data, B.data, C.data,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
#endif
#endif

      return D;
    }
  };

   template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
            matrix::matrix_layout LayoutC>
  struct joint_matrix_mad_impl<
      matrix::matrix_type::bf16, matrix::matrix_type::fp32, 8, 16, 32, LayoutA,
      LayoutB, LayoutC,
      typename std::enable_if_t<(LayoutA == matrix::matrix_layout::row_major ||
                                 LayoutA == matrix::matrix_layout::col_major) &&
                                (LayoutB == matrix::matrix_layout::row_major ||
                                 LayoutB == matrix::matrix_layout::col_major) &&
                                (LayoutC == matrix::matrix_layout::row_major ||
                                 LayoutC ==
                                     matrix::matrix_layout::col_major)>> {
    matrix::joint_matrix<matrix::matrix_type::fp32,
                         matrix::matrix_use::accumulator, 8, 32, LayoutC>
    mad(matrix::joint_matrix<matrix::matrix_type::bf16, matrix::matrix_use::a,
                             8, 16, LayoutA>
            A,
        matrix::joint_matrix<matrix::matrix_type::bf16, matrix::matrix_use::b,
                             16, 32, LayoutB>
            B,
        matrix::joint_matrix<matrix::matrix_type::fp32,
                             matrix::matrix_use::accumulator, 8, 32, LayoutC>
            C) {
      matrix::joint_matrix<matrix::matrix_type::fp32,
                           matrix::matrix_use::accumulator, 8, 32, LayoutC>
          D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __mma_bf16_m8n32k16_mma_f32(D.data, A.data, B.data, C.data,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
#endif
#endif

      return D;
    }
  };

    template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
            matrix::matrix_layout LayoutC>
  struct joint_matrix_mad_impl<
      matrix::matrix_type::sint8, matrix::matrix_type::sint32, 16, 16, 16, LayoutA,
      LayoutB, LayoutC,
      typename std::enable_if_t<(LayoutA == matrix::matrix_layout::row_major ||
                                 LayoutA == matrix::matrix_layout::col_major) &&
                                (LayoutB == matrix::matrix_layout::row_major ||
                                 LayoutB == matrix::matrix_layout::col_major) &&
                                (LayoutC == matrix::matrix_layout::row_major ||
                                 LayoutC ==
                                     matrix::matrix_layout::col_major)>> {
    matrix::joint_matrix<matrix::matrix_type::sint32,
                         matrix::matrix_use::accumulator, 16, 16, LayoutC>
    mad(matrix::joint_matrix<matrix::matrix_type::sint8, matrix::matrix_use::a,
                             16, 16, LayoutA>
            A,
        matrix::joint_matrix<matrix::matrix_type::sint8, matrix::matrix_use::b,
                             16, 16, LayoutB>
            B,
        matrix::joint_matrix<matrix::matrix_type::sint32,
                             matrix::matrix_use::accumulator, 16, 16, LayoutC>
            C) {
      matrix::joint_matrix<matrix::matrix_type::sint32,
                           matrix::matrix_use::accumulator, 16, 16, LayoutC>
          D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __imma_m16n16k16_mma_s8(D.data, A.data, B.data, C.data,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
#endif
#endif

      return D;
    }
  };

      template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
            matrix::matrix_layout LayoutC>
  struct joint_matrix_mad_impl<
      matrix::matrix_type::uint8, matrix::matrix_type::sint32, 16, 16, 16, LayoutA,
      LayoutB, LayoutC,
      typename std::enable_if_t<(LayoutA == matrix::matrix_layout::row_major ||
                                 LayoutA == matrix::matrix_layout::col_major) &&
                                (LayoutB == matrix::matrix_layout::row_major ||
                                 LayoutB == matrix::matrix_layout::col_major) &&
                                (LayoutC == matrix::matrix_layout::row_major ||
                                 LayoutC ==
                                     matrix::matrix_layout::col_major)>> {
    matrix::joint_matrix<matrix::matrix_type::sint32,
                         matrix::matrix_use::accumulator, 16, 16, LayoutC>
    mad(matrix::joint_matrix<matrix::matrix_type::uint8, matrix::matrix_use::a,
                             16, 16, LayoutA>
            A,
        matrix::joint_matrix<matrix::matrix_type::uint8, matrix::matrix_use::b,
                             16, 16, LayoutB>
            B,
        matrix::joint_matrix<matrix::matrix_type::sint32,
                             matrix::matrix_use::accumulator, 16, 16, LayoutC>
            C) {
      matrix::joint_matrix<matrix::matrix_type::sint32,
                           matrix::matrix_use::accumulator, 16, 16, LayoutC>
          D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      __imma_m16n16k16_mma_u8(D.data, A.data, B.data, C.data,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
#endif
#endif

      return D;
    }
  };

  } // namespace detail

  namespace experimental::matrix {

  template <typename Group, typename U, matrix_type T, matrix_use MT,
            size_t NumRows, size_t NumCols, matrix_layout Layout,
            access::address_space Space>
  void
  joint_matrix_load(Group sg,
                    joint_matrix<T, MT, NumRows, NumCols, Layout, Group> &res,
                    multi_ptr<U, Space> src, size_t stride) {
    detail::joint_matrix_load_impl<U, T, MT, NumRows, NumCols, Layout, Space>{}
        .load(res, src, stride);
  }

  template <typename Group, typename U, matrix_type T, size_t NumRows,
            size_t NumCols, matrix_layout Layout, access::address_space Space>
  void joint_matrix_store(Group sg,
                          joint_matrix<T, matrix_use::accumulator, NumRows,
                                       NumCols, Layout, Group> &src,
                          multi_ptr<U, Space> dst, size_t stride) {
    detail::joint_matrix_store_impl<U, T, NumRows, NumCols, Layout, Space>{}
        .store(src, dst, stride);
  }

  template <typename Group, matrix_type T1, matrix_type T2, std::size_t M,
            std::size_t K, std::size_t N, matrix_layout LayoutA,
            matrix_layout LayoutB, matrix_layout LayoutC>
  joint_matrix<T2, matrix_use::accumulator, M, N, LayoutC, Group>
  joint_matrix_mad(
      Group sg, joint_matrix<T1, matrix_use::a, M, K, LayoutA, Group> A,
      joint_matrix<T1, matrix_use::b, K, N, LayoutB, Group> B,
      joint_matrix<T2, matrix_use::accumulator, M, N, LayoutC, Group> C) {
    return detail::joint_matrix_mad_impl<T1, T2, M, K, N, LayoutA, LayoutB,
                                         LayoutC>{}
        .mad(A, B, C);
  }

  } // namespace experimental::matrix
  } // namespace oneapi
  } // namespace ext
  } // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
