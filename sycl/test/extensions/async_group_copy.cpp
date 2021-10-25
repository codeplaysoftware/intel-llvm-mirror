#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>

enum class SYCLGroup {
    group,
    subgroup
};

void copy_test(sycl::queue& q, size_t sub_group_size){
    constexpr int num_of_work_groups = 3;
    const int work_group_size = sub_group_size*2;
    const int numValues = num_of_work_groups*work_group_size;
    std::vector<int> values(numValues,1);
    std::vector<int> src(numValues,1);
    std::vector<int> dst(numValues,1);
    std::iota(std::begin(values), std::end(values), 0);
    {
        sycl::buffer<int> bufA(values);
        sycl::buffer<int> bufSrc(src);
        sycl::buffer<int> bufDst(dst);

        q.submit([&](sycl::handler& h){
            auto acc_a = bufA.template get_access<sycl::access::mode::read_write>(h);
            auto acc_src = bufSrc.template get_access<sycl::access::mode::write>(h);
            auto acc_dst = bufDst.template get_access<sycl::access::mode::write>(h);
            auto local_data = sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local>(work_group_size,h);
            h.parallel_for<class CopyTest>(sycl::nd_range<1>{numValues, work_group_size}, [=](sycl::nd_item<1> it){
                auto i = it.get_global_id();
                auto group = it.get_group();
                auto sub_group = it.get_sub_group();

                local_data[i] = 0;
                sycl::group_barrier(group);

                auto sub_group_offset = sub_group.get_local_range()*sub_group.get_group_id();
                auto work_group_offset = group.get_local_range(0)*group.get_id() + sub_group_offset;

                acc_src[i] = work_group_offset;
                acc_dst[i] = sub_group_offset;
                auto global_src = acc_a.get_pointer()+work_group_offset;
                auto local_dst = local_data.get_pointer() + sub_group_offset;

                auto e = sycl::ext::oneapi::async_group_copy(sub_group, local_dst, global_src, 5,1);
                sycl::ext::oneapi::wait_for(e);

                acc_a[i] = local_data[it.get_local_linear_id()];
            });
        });

    }

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


int main(){
    sycl::queue q{sycl::default_selector{}};
    std::cout << q.get_device().get_info<sycl::info::device::name>() <<'\n';
    auto sub_group_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    std::cout << "sub_group sizes:\n";
    for (auto s : sub_group_sizes){
        std::cout << '\t' << s<< '\n';
    }
    auto sub_group_size = *std::max_element(std::begin(sub_group_sizes), std::end(sub_group_sizes));
    std::cout << "maximum sub_group size: " << sub_group_size << '\n';

    copy_test(q, sub_group_size);
}

