// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <type_traits>

using namespace sycl;

static device dev;

// Uncomment to print all test mismatches
// #define VERBOSE_PRINT

// Helpers and utilities
struct util {
  template <typename DType, int NChannels,
            std::enable_if_t<std::is_integral<DType>::value, bool> = true>
  static void fill_rand(std::vector<sycl::vec<DType, NChannels>> &v) {
    std::default_random_engine generator;
    std::uniform_int_distribution<DType> distribution(0, 100);
    for (int i = 0; i < v.size(); ++i) {
      v[i] = sycl::vec<DType, NChannels>(distribution(generator));
    }
  }

  template <typename DType, int NChannels,
            std::enable_if_t<std::is_floating_point<DType>::value, bool> = true>
  static void fill_rand(std::vector<sycl::vec<DType, NChannels>> &v) {
    std::default_random_engine generator;
    std::uniform_real_distribution<DType> distribution(0.0, 100.0);
    for (int i = 0; i < v.size(); ++i) {
      v[i] = sycl::vec<DType, NChannels>(distribution(generator));
    }
  }

  template <typename DType, int NChannels,
            std::enable_if_t<std::is_same<DType, half>::value, bool> = true>
  static void fill_rand(std::vector<sycl::vec<DType, NChannels>> &v) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 100.0);
    for (int i = 0; i < v.size(); ++i) {
      v[i] = sycl::vec<DType, NChannels>(distribution(generator));
    }
  }

  template <typename DType, int NChannels>
  static void add_host(const std::vector<sycl::vec<DType, NChannels>> &in_0,
                       const std::vector<sycl::vec<DType, NChannels>> &in_1,
                       std::vector<sycl::vec<DType, NChannels>> &out) {
    for (int i = 0; i < out.size(); ++i) {
      for (int j = 0; j < NChannels; ++j) {
        out[i][j] = in_0[i][j] + in_1[i][j];
      }
    }
  }

  template <typename DType, int NChannels,
            typename = std::enable_if_t<NChannels == 1>>
  static DType add_kernel(const DType in_0, const DType in_1) {
    return in_0 + in_1;
  }

  template <typename DType, int NChannels,
            typename = std::enable_if_t<(NChannels > 1)>>
  static sycl::vec<DType, NChannels>
  add_kernel(const sycl::vec<DType, NChannels> &in_0,
             const sycl::vec<DType, NChannels> &in_1) {
    sycl::vec<DType, NChannels> out;
    for (int i = 0; i < NChannels; ++i) {
      out[i] = in_0[i] + in_1[i];
    }
    return out;
  }

  // parallel_for 2D
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 2>>
  static void
  run_ndim_test(sycl::queue q, sycl::range<2> globalSize,
                sycl::range<2> bufSize, sycl::range<2> localSize,
                sycl::ext::oneapi::unsampled_image_handle input_0,
                sycl::ext::oneapi::unsampled_image_handle input_1,
                sycl::buffer<sycl::vec<DType, NChannels>, 2> &output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](handler &cgh) {
        auto outAcc =
            output.template get_access<access_mode::write>(cgh, bufSize);

        cgh.parallel_for<KernelName>(
            nd_range<NDims>{globalSize, localSize}, [=](nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);

              VecType px1 = sycl::ext::oneapi::read_image<VecType>(
                  input_0, int2(dim0, dim1));
              VecType px2 = sycl::ext::oneapi::read_image<VecType>(
                  input_1, int2(dim0, dim1));

              auto sum = util::add_kernel<DType, NChannels>(px1, px2);

              outAcc[id<2>{dim1, dim0}] = VecType(sum);
            });
      });
    } catch (sycl::exception e) {
      std::cout << "\tKernel submission failed! " << e.what() << std::endl;
    } catch (...) {
      std::cout << "\tKernel submission failed!" << std::endl;
    }
  }

  // parallel_for 1D
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 1>>
  static void
  run_ndim_test(sycl::queue q, sycl::range<1> globalSize,
                sycl::range<1> bufSize, sycl::range<1> localSize,
                sycl::ext::oneapi::unsampled_image_handle input_0,
                sycl::ext::oneapi::unsampled_image_handle input_1,
                sycl::buffer<sycl::vec<DType, NChannels>, 1> &output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](handler &cgh) {
        auto outAcc =
            output.template get_access<access_mode::write>(cgh, bufSize);

        cgh.parallel_for<KernelName>(
            nd_range<NDims>{globalSize, localSize}, [=](nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);

              VecType px1 =
                  sycl::ext::oneapi::read_image<VecType>(input_0, int(dim0));
              VecType px2 =
                  sycl::ext::oneapi::read_image<VecType>(input_1, int(dim0));

              auto sum = util::add_kernel<DType, NChannels>(px1, px2);

              outAcc[id<1>{dim0}] = VecType(sum);
            });
      });
    } catch (sycl::exception e) {
      std::cout << "\tKernel submission failed! " << e.what() << std::endl;
    } catch (...) {
      std::cout << "\tKernel submission failed!" << std::endl;
    }
  }
};

template <int NDims, typename DType, int NChannels, image_channel_type CType,
          image_channel_order COrder, typename KernelName>
bool run_test(sycl::range<NDims> dims, sycl::range<NDims> localSize,
              unsigned int seed = 0) {
  using VecType = sycl::vec<DType, NChannels>;

  queue q(dev);
  auto ctxt = q.get_context();

  size_t num_elems = dims[0];
  if (NDims > 1)
    num_elems *= dims[1];
  if (NDims > 2)
    num_elems *= dims[2];

  std::vector<VecType> input_0(num_elems);
  std::vector<VecType> input_1(num_elems);
  std::vector<VecType> expected(num_elems);
  std::vector<VecType> actual(num_elems);

  std::srand(seed);
  util::fill_rand(input_0);
  util::fill_rand(input_1);
  util::add_host(input_0, input_1, expected);

  unsigned int element_size_bytes = sizeof(DType) * NChannels;
  size_t width_in_bytes = dims[0] * element_size_bytes;
  size_t pitch_in_bytes1 = 0;
  size_t pitch_in_bytes2 = 0;

  void *device_ptr_0, *device_ptr_1;

  // 1D and 2D images use USM
  if constexpr (NDims == 1) {
    device_ptr_0 = sycl::malloc_device(width_in_bytes, q);
    device_ptr_1 = sycl::malloc_device(width_in_bytes, q);
  } else if constexpr (NDims == 2) {
    device_ptr_0 = sycl::ext::oneapi::pitched_alloc_device(
        &pitch_in_bytes1, width_in_bytes, dims[1], element_size_bytes, q);
    device_ptr_1 = sycl::ext::oneapi::pitched_alloc_device(
        &pitch_in_bytes2, width_in_bytes, dims[1], element_size_bytes, q);
  }

  sycl::ext::oneapi::image_descriptor desc(dims, COrder, CType,
                                           pitch_in_bytes1);

  if (device_ptr_0 == nullptr || device_ptr_1 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return false;
  }

  if constexpr (NDims == 1) {
    q.memcpy(device_ptr_0, input_0.data(), width_in_bytes);
    q.memcpy(device_ptr_1, input_1.data(), width_in_bytes);
  } else if constexpr (NDims == 2) {
    q.ext_image_memcpy(device_ptr_0, input_0.data(), desc);
    q.ext_image_memcpy(device_ptr_1, input_1.data(), desc);
  }

  q.wait();

  auto img_input_0 = sycl::ext::oneapi::create_image(ctxt, device_ptr_0, desc);
  auto img_input_1 = sycl::ext::oneapi::create_image(ctxt, device_ptr_1, desc);

  {
    sycl::range<NDims> bufSize = dims;
    if constexpr (NDims == 2) {
      bufSize[0] = dims[1];
      bufSize[1] = dims[0];
    }
    sycl::range<NDims> globalSize = dims;
    sycl::buffer<VecType, NDims> outBuf((VecType *)actual.data(), bufSize);
    q.wait();
    util::run_ndim_test<NDims, DType, NChannels, KernelName>(
        q, globalSize, bufSize, localSize, img_input_0, img_input_1, outBuf);
    q.wait();
  }

  // Cleanup
  try {
    sycl::free(device_ptr_0, ctxt);
    sycl::free(device_ptr_1, ctxt);
    sycl::ext::oneapi::destroy_image_handle(ctxt, img_input_0);
    sycl::ext::oneapi::destroy_image_handle(ctxt, img_input_1);
  } catch (...) {
    std::cout << "Failed to destroy image handle." << std::endl;
    return false;
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < num_elems; i++) {
    for (int j = 0; j < NChannels; ++j) {
      bool mismatch = false;
      if (actual[i][j] != expected[i][j]) {
        mismatch = true;
        validated = false;
      }
      if (mismatch) {
        std::cout << "\tResult mismatch at [" << i << "][" << j
                  << "] Expected: " << expected[i][j]
                  << ", Actual: " << actual[i][j] << std::endl;
#ifndef VERBOSE_PRINT
        return false;
#endif //VERBOSE_PRINT
      }
    }
  }
  if (validated) {
    std::cout << "\tCorrect output!" << std::endl;
  }

  return validated;
}

int main() {

  unsigned int seed = 0;

  std::cout << "Running 1D int\n";
  run_test<1, int, 1, image_channel_type::signed_int32, image_channel_order::r,
           class int_1d>({32}, {2}, seed);
  std::cout << "Running 2D int\n";
  run_test<2, int32_t, 1, image_channel_type::signed_int32,
           image_channel_order::r, class int_2d>({2816, 32}, {32, 32}, seed);
  std::cout << "Running 1D int2\n";
  run_test<1, int, 2, image_channel_type::signed_int32, image_channel_order::rg,
           class int2_1d>({32}, {2}, seed);
  std::cout << "Running 2D int2\n";
  run_test<2, int32_t, 2, image_channel_type::signed_int32,
           image_channel_order::rg, class int2_2d>({2816, 32}, {32, 32}, seed);
  std::cout << "Running 1D int4\n";
  run_test<1, int, 4, image_channel_type::signed_int32,
           image_channel_order::rgba, class int4_1d>({32}, {2}, seed);
  std::cout << "Running 2D int4\n";
  run_test<2, int32_t, 4, image_channel_type::signed_int32,
           image_channel_order::rgba, class int4_2d>({2816, 32}, {32, 32},
                                                     seed);

  std::cout << "Running 1D unsigned int\n";
  run_test<1, unsigned int, 1, image_channel_type::unsigned_int32,
           image_channel_order::r, class uint_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned int\n";
  run_test<2, uint32_t, 1, image_channel_type::unsigned_int32,
           image_channel_order::r, class uint_2d>({2816, 32}, {32, 32}, seed);
  std::cout << "Running 1D unsigned int2\n";
  run_test<1, unsigned int, 2, image_channel_type::unsigned_int32,
           image_channel_order::rg, class uint2_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned int2\n";
  run_test<2, uint32_t, 2, image_channel_type::unsigned_int32,
           image_channel_order::rg, class uint2_2d>({2816, 32}, {32, 32}, seed);
  std::cout << "Running 1D unsigned int4\n";
  run_test<1, unsigned int, 4, image_channel_type::unsigned_int32,
           image_channel_order::rgba, class uint4_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned int4\n";
  run_test<2, uint32_t, 4, image_channel_type::unsigned_int32,
           image_channel_order::rgba, class uint4_2d>({2816, 32}, {32, 32},
                                                      seed);

  std::cout << "Running 1D short\n";
  run_test<1, short, 1, image_channel_type::signed_int16,
           image_channel_order::r, class short_1d>({32}, {2}, seed);
  std::cout << "Running 2D short\n";
  run_test<2, short, 1, image_channel_type::signed_int16,
           image_channel_order::r, class short_2d>({2816, 32}, {32, 32}, seed);
  std::cout << "Running 1D short2\n";
  run_test<1, short, 2, image_channel_type::signed_int16,
           image_channel_order::rg, class short2_1d>({32}, {2}, seed);
  std::cout << "Running 2D short2\n";
  run_test<2, short, 2, image_channel_type::signed_int16,
           image_channel_order::rg, class short2_2d>({2816, 32}, {32, 32},
                                                     seed);
  std::cout << "Running 1D short4\n";
  run_test<1, short, 4, image_channel_type::signed_int16,
           image_channel_order::rgba, class short4_1d>({32}, {2}, seed);
  std::cout << "Running 2D short4\n";
  run_test<2, short, 4, image_channel_type::signed_int16,
           image_channel_order::rgba, class short4_2d>({2816, 32}, {32, 32},
                                                       seed);

  std::cout << "Running 1D unsigned short\n";
  run_test<1, unsigned short, 1, image_channel_type::unsigned_int16,
           image_channel_order::r, class ushort_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned short\n";
  run_test<2, unsigned short, 1, image_channel_type::unsigned_int16,
           image_channel_order::r, class ushort_2d>({2816, 32}, {32, 32}, seed);

  std::cout << "Running 1D unsigned short2\n";
  run_test<1, unsigned short, 2, image_channel_type::unsigned_int16,
           image_channel_order::rg, class ushort2_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned short2\n";
  run_test<2, unsigned short, 2, image_channel_type::unsigned_int16,
           image_channel_order::rg, class ushort2_2d>({2816, 32}, {32, 32},
                                                      seed);
  std::cout << "Running 1D unsigned short4\n";
  run_test<1, unsigned short, 4, image_channel_type::unsigned_int16,
           image_channel_order::rgba, class ushort4_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned short4\n";
  run_test<2, unsigned short, 4, image_channel_type::unsigned_int16,
           image_channel_order::rgba, class ushort4_2d>({2816, 32}, {32, 32},
                                                        seed);

  std::cout << "Running 1D char\n";
  run_test<1, signed char, 1, image_channel_type::signed_int8,
           image_channel_order::r, class char_1d>({32}, {2}, seed);
  std::cout << "Running 2D char\n";
  run_test<2, signed char, 1, image_channel_type::signed_int8,
           image_channel_order::r, class char_2d>({2816, 32}, {32, 32}, seed);

  std::cout << "Running 1D char2\n";
  run_test<1, signed char, 2, image_channel_type::signed_int8,
           image_channel_order::rg, class char2_1d>({32}, {2}, seed);
  std::cout << "Running 2D char2\n";
  run_test<2, signed char, 2, image_channel_type::signed_int8,
           image_channel_order::rg, class char2_2d>({2816, 32}, {32, 32}, seed);

  std::cout << "Running 1D char4\n";
  run_test<1, signed char, 4, image_channel_type::signed_int8,
           image_channel_order::rgba, class char4_1d>({32}, {2}, seed);
  std::cout << "Running 2D char4\n";
  run_test<2, signed char, 4, image_channel_type::signed_int8,
           image_channel_order::rgba, class char4_2d>({2816, 32}, {32, 32},
                                                      seed);

  std::cout << "Running 1D unsigned char\n";
  run_test<1, unsigned char, 1, image_channel_type::unsigned_int8,
           image_channel_order::r, class uchar_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned char\n";
  run_test<2, unsigned char, 1, image_channel_type::unsigned_int8,
           image_channel_order::r, class uchar_2d>({2816, 32}, {32, 32}, seed);

  std::cout << "Running 1D unsigned char2\n";
  run_test<1, unsigned char, 2, image_channel_type::unsigned_int8,
           image_channel_order::rg, class uchar2_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned char2\n";
  run_test<2, unsigned char, 2, image_channel_type::unsigned_int8,
           image_channel_order::rg, class uchar2_2d>({2816, 32}, {32, 32},
                                                     seed);
  std::cout << "Running 1D unsigned char4\n";
  run_test<1, unsigned char, 4, image_channel_type::unsigned_int8,
           image_channel_order::rgba, class uchar4_1d>({32}, {2}, seed);
  std::cout << "Running 2D unsigned char4\n";
  run_test<2, unsigned char, 4, image_channel_type::unsigned_int8,
           image_channel_order::rgba, class uchar4_2d>({2816, 32}, {32, 32},
                                                       seed);

  std::cout << "Running 1D float\n";
  run_test<1, float, 1, image_channel_type::fp32, image_channel_order::r,
           class float_1d>({1024}, {512}, seed);
  std::cout << "Running 2D float\n";
  run_test<2, float, 1, image_channel_type::fp32, image_channel_order::r,
           class float_2d>({4096, 3808}, {32, 32}, seed);
  std::cout << "Running 1D float2\n";
  run_test<1, float, 2, image_channel_type::fp32, image_channel_order::rg,
           class float2_1d>({608}, {32}, seed);
  std::cout << "Running 2D float2\n";
  run_test<2, float, 2, image_channel_type::fp32, image_channel_order::rg,
           class float2_2d>({3808, 4096}, {32, 32}, seed);
  std::cout << "Running 1D float4\n";
  run_test<1, float, 4, image_channel_type::fp32, image_channel_order::rgba,
           class float4_1d>({1024}, {512}, seed);
  std::cout << "Running 2D float4\n";
  run_test<2, float, 4, image_channel_type::fp32, image_channel_order::rgba,
           class float4_2d>({4096, 4096}, {32, 32}, seed);

  std::cout << "Running 1D half\n";
  run_test<1, half, 1, image_channel_type::fp16, image_channel_order::r,
           class half_1d>({32}, {2}, seed);
  std::cout << "Running 2D half\n";
  run_test<2, half, 1, image_channel_type::fp16, image_channel_order::r,
           class half_2d>({2816, 32}, {32, 32}, seed);
  std::cout << "Running 1D half2\n";
  run_test<1, half, 2, image_channel_type::fp16, image_channel_order::rg,
           class half2_1d>({32}, {2}, seed);
  std::cout << "Running 2D half2\n";
  run_test<2, half, 2, image_channel_type::fp16, image_channel_order::rg,
           class half2_2d>({2816, 32}, {32, 32}, seed);
  std::cout << "Running 1D half4\n";
  run_test<1, half, 4, image_channel_type::fp16, image_channel_order::rgba,
           class half4_1d>({32}, {2}, seed);
  std::cout << "Running 2D half4\n";
  run_test<2, half, 4, image_channel_type::fp16, image_channel_order::rgba,
           class half4_2d>({2816, 32}, {32, 32}, seed);

  return 0;
}
