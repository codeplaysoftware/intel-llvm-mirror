#include <CL/sycl.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

constexpr bool print_values = false;

enum class Result { Success, Failure };

Result under_copy_test(sycl::queue &q, size_t max_sub_group_size) {
  constexpr int num_of_work_groups = 3;
  constexpr size_t num_of_copies = 5;
  const size_t work_group_size = max_sub_group_size * 2;
  const size_t numValues = num_of_work_groups * work_group_size;
  std::vector<int> values(numValues, 1);
  std::vector<int> src(numValues, 1);
  std::vector<int> dst(numValues, 1);
  size_t sub_group_size = 0;
  std::iota(std::begin(values), std::end(values), 0);
  {
    sycl::buffer<int> bufA(values);
    sycl::buffer<int> bufSrc(src);
    sycl::buffer<int> bufDst(dst);
    sycl::buffer<size_t> buf_sub_group_size(&sub_group_size, sycl::range<1>{1});

    q.submit([&](sycl::handler &h) {
      auto acc_a = bufA.template get_access<sycl::access::mode::read_write>(h);
      auto acc_src = bufSrc.template get_access<sycl::access::mode::write>(h);
      auto acc_dst = bufDst.template get_access<sycl::access::mode::write>(h);
      auto acc_sub_group_size =
          buf_sub_group_size.template get_access<sycl::access::mode::write>(h);
      auto local_data =
          sycl::accessor<int, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>(work_group_size, h);
      h.parallel_for<class UnderCopyTest>(
          sycl::nd_range<1>{numValues, work_group_size},
          [=](sycl::nd_item<1> it) {
            auto i = it.get_global_id();
            auto group = it.get_group();
            auto sub_group = it.get_sub_group();
            auto sub_i = it.get_local_linear_id();

            if (i == 0) {
              acc_sub_group_size[0] = sub_group.get_local_range()[0];
            }
            local_data[sub_i] = 0;
            sycl::group_barrier(group);

            auto sub_group_offset =
                sub_group.get_local_range() * sub_group.get_group_id();
            auto work_group_offset =
                group.get_local_range(0) * group.get_id() + sub_group_offset;

            acc_src[i] = work_group_offset;
            acc_dst[i] = sub_group_offset;
            auto global_src = acc_a.get_pointer() + work_group_offset;
            auto local_dst = local_data.get_pointer() + sub_group_offset;

            auto e = sycl::ext::oneapi::async_group_copy(
                sub_group, local_dst, global_src, num_of_copies);
            sycl::ext::oneapi::wait_for(sub_group, e);

            acc_a[i] = local_data[sub_i];
          });
    });
  }

  // print values for manual inspection
  if constexpr (print_values) {
    std::cout << "chosen subgroup size: " << sub_group_size << '\n';
    for (size_t i = 0; i < num_of_work_groups; ++i) {
      std::cout << "src: ";
      for (size_t j = 0; j < work_group_size; ++j) {
        std::cout << std::setfill('0') << std::setw(3)
                  << src[i * work_group_size + j] << ' ';
      }
      std::cout << "\ndst: ";
      for (size_t j = 0; j < work_group_size; ++j) {
        std::cout << std::setfill('0') << std::setw(3)
                  << dst[i * work_group_size + j] << ' ';
      }
      std::cout << "\nval: ";
      for (size_t j = 0; j < work_group_size; ++j) {
        std::cout << std::setfill('0') << std::setw(3)
                  << values[i * work_group_size + j] << ' ';
      }
      std::cout << '\n';
      std::cout << '\n';
    }
  }

  const size_t num_of_sub_groups = work_group_size / sub_group_size;
  int i = 0;
  for (size_t work_group = 0; work_group != num_of_work_groups; ++work_group) {
    for (size_t sub_group = 0; sub_group != num_of_sub_groups; ++sub_group) {
      for (size_t work_item = 0; work_item != sub_group_size; ++work_item) {
        const auto v = values[work_group * work_group_size +
                              sub_group * sub_group_size + work_item];
        const auto expected_value = work_item < num_of_copies ? i : 0;
        if (v != expected_value) {
          std::cout << "index: " << i << " (work_group: " << work_group
                    << ", sub_group: " << sub_group
                    << ", work_item: " << work_item << ")\nexpected the value "
                    << expected_value << " but found " << v << '\n';
          return Result::Failure;
        }
        i += 1;
      }
    }
  }

  return Result::Success;
}

Result over_copy_test(sycl::queue &q, size_t max_sub_group_size) {
  constexpr size_t num_of_work_groups = 3;
  constexpr size_t chosen_sub_group = 1;
  const size_t num_of_copies = max_sub_group_size + 5;
  const size_t work_group_size = max_sub_group_size * 4;
  const size_t numValues = num_of_work_groups * work_group_size;
  std::vector<int> values(numValues, 1);
  std::vector<int> src(numValues, 1);
  std::vector<int> dst(numValues, 1);
  size_t sub_group_size = 0;
  std::iota(std::begin(values), std::end(values), 0);
  {
    sycl::buffer<int> bufA(values);
    sycl::buffer<int> bufSrc(src);
    sycl::buffer<int> bufDst(dst);
    sycl::buffer<size_t> buf_sub_group_size(&sub_group_size, sycl::range<1>{1});

    q.submit([&](sycl::handler &h) {
      auto acc_a = bufA.template get_access<sycl::access::mode::read_write>(h);
      auto acc_src = bufSrc.template get_access<sycl::access::mode::write>(h);
      auto acc_dst = bufDst.template get_access<sycl::access::mode::write>(h);
      auto acc_sub_group_size =
          buf_sub_group_size.template get_access<sycl::access::mode::write>(h);
      auto local_data =
          sycl::accessor<int, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>(work_group_size, h);
      h.parallel_for<class OverCopyTest>(
          sycl::nd_range<1>{numValues, work_group_size},
          [=](sycl::nd_item<1> it) {
            auto i = it.get_global_id();
            auto group = it.get_group();
            auto sub_group = it.get_sub_group();
            auto sub_i = it.get_local_linear_id();

            if (i == 0) {
              acc_sub_group_size[0] = sub_group.get_local_range()[0];
            }
            local_data[sub_i] = 0;
            sycl::group_barrier(group);

            auto sub_group_offset =
                sub_group.get_local_range() * sub_group.get_group_id();
            auto work_group_offset =
                group.get_local_range(0) * group.get_id() + sub_group_offset;

            acc_src[i] = work_group_offset;
            acc_dst[i] = sub_group_offset;
            auto global_src = acc_a.get_pointer() + work_group_offset;
            auto local_dst = local_data.get_pointer() + sub_group_offset;

            if (sub_group.get_group_id() == chosen_sub_group) {
              auto e = sycl::ext::oneapi::async_group_copy(
                  sub_group, local_dst, global_src, num_of_copies);
              sycl::ext::oneapi::wait_for(sub_group, e);
            }
            sycl::group_barrier(group);


            acc_a[i] = local_data[sub_i];
          });
    });
  }

  // print values for manual inspection
  if constexpr (print_values) {
    std::cout << "number of copies: " << num_of_copies
              << "\nchosen subgroup size: " << sub_group_size << '\n';
    for (size_t i = 0; i < num_of_work_groups; ++i) {
      std::cout << "src: ";
      for (size_t j = 0; j < work_group_size; ++j) {
        std::cout << std::setfill('0') << std::setw(3)
                  << src[i * work_group_size + j] << ' ';
      }
      std::cout << "\ndst: ";
      for (size_t j = 0; j < work_group_size; ++j) {
        std::cout << std::setfill('0') << std::setw(3)
                  << dst[i * work_group_size + j] << ' ';
      }
      std::cout << "\nval: ";
      for (size_t j = 0; j < work_group_size; ++j) {
        std::cout << std::setfill('0') << std::setw(3)
                  << values[i * work_group_size + j] << ' ';
      }
      std::cout << '\n';
      std::cout << '\n';
    }
  }

  const size_t num_of_sub_groups = work_group_size / sub_group_size;
  int i = 0;
  for (size_t work_group = 0; work_group != num_of_work_groups; ++work_group) {
    for (size_t sub_group = 0; sub_group != num_of_sub_groups; ++sub_group) {
      for (size_t work_item = 0; work_item != sub_group_size; ++work_item) {
        const auto v = values[work_group * work_group_size +
                              sub_group * sub_group_size + work_item];
        const auto expected_value =
            (sub_group < chosen_sub_group ||
             sub_group * sub_group_size + work_item >
                 sub_group_size * chosen_sub_group + (num_of_copies - 1))
                ? 0
                : i;
        if (v != expected_value) {
          std::cout << "index: " << i << " (work_group: " << work_group
                    << ", sub_group: " << sub_group
                    << ", work_item: " << work_item << ")\nexpected the value "
                    << expected_value << " but found " << v << '\n';
          return Result::Failure;
        }
        i += 1;
      }
    }
  }

  return Result::Success;
}

int main() {
  sycl::queue q{sycl::default_selector{}};
  auto sub_group_sizes =
      q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto max_sub_group_size =
      *std::max_element(std::begin(sub_group_sizes), std::end(sub_group_sizes));

  if constexpr (print_values) {
    std::cout << "Test copying less than the size of the subgroup\n";
  }
  if (under_copy_test(q, max_sub_group_size) != Result::Success) {
    return 1;
  } else if (print_values) {
    std::cout << "Success!\n";
  }

  if constexpr (print_values) {
    std::cout << "Test copying more than the size of the subgroup\n";
  }
  if (over_copy_test(q, max_sub_group_size) != Result::Success) {
    return 1;
  } else if (print_values) {
    std::cout << "Success!\n";
  }
}
