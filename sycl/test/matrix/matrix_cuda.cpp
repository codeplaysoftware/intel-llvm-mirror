#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::experimental::matrix;

#define x 16
#define y 16

int main() {

  int C[256];

  for (auto &c : C) {
    c = 0;
  }

  buffer<int, 2> bufC(C, range<2>(16, 16));

  queue q;
  q.submit([&](handler &cgh) {
    auto accC = bufC.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class imatrix>(
        nd_range<2>({1, 32}, {1, 32}), [=](nd_item<2> item)

        {
          ext::oneapi::sub_group sg = item.get_sub_group();
          joint_matrix<sub_group, int, matrix_type::c, x, y,
                       matrix_layout::row_major>
              sub_c;
          //__imma_m16n16k16_ld_c(sub_c.data, accC.get_pointer(), 23, 1);
          // joint_matrix_load(sg, sub_c, accC.get_pointer(), 23);
        });
  });

  return 0;
};
