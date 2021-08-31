#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::experimental::matrix;

// This matrix size only requires a single subgroup operation: The "big matrix"
// matches the size of a tile "sub matrix" for A,B,C: i.e. N=M=K=n=m=k=16.
#define x 16
#define y 16

// constexpr int M = 16;
// constexpr int N = 16;
// constexpr int K = 16;

constexpr int N_THREADS_PER_MATRIX_OP = 32;
constexpr int N_TILES = 1;

// The stride should equal the leading dimension of the "big matrix" matrix I
// think.
constexpr int STRIDE = 16;

int main() {

  int32_t C[256];
  int32_t A[256];
  int32_t B[256];

  for (int i = 0; i < 256; i++) {
    A[i] = 0;
    B[i] = 0;
    C[i] = 0;
  }

  // input data appears to the same for each matrix in this case...
  buffer<int32_t, 2> bufC(C, range<2>(x, y));
  buffer<int32_t, 2> bufA(A, range<2>(x, y));
  buffer<int32_t, 2> bufB(B, range<2>(x, y));

  queue q;
  q.submit([&](handler &cgh) {
    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class imatrix>(
        nd_range<2>({N_TILES, N_THREADS_PER_MATRIX_OP},
                    {N_TILES, N_THREADS_PER_MATRIX_OP}),
        [=](nd_item<2> item)

        {
          ext::oneapi::sub_group sg = item.get_sub_group();
          joint_matrix<sub_group, matrix_type::c, x, y,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<sub_group, matrix_type::a, x, y,
                       matrix_layout::row_major>
              sub_a;

          joint_matrix<sub_group, matrix_type::b, x, y,
                       matrix_layout::row_major>
              sub_b;

          // Note that matrix "C" here is acting as matrices C,D in ptx
          // terminology. It is added to the matrix product of A and B. calls:
          //__imma_m16n16k16_ld_c(sub_c.data, accC.get_pointer(), 16, 0);
          joint_matrix_load(sg, sub_c, accC.get_pointer(), 16);

          // TODO There is a seemingly notintuitive loop consisting of K / k
          // iterations in the AMX tests. At first glance this looks like the
          // A/B matrices are loaded one row/column at a time and a row of
          // matrix A is multiplied by a column of matrix B at a time.  But this
          // idea doesn't match the implementation.  Hopefully we can ignore
          // this detail until there is at least a working implementation for a
          // single matrix tile.

          // e.g. for (int i = 0; i < K/k; i++)
          {
            joint_matrix_load(sg, sub_a, accA.get_pointer(), 16);
            //__imma_m16n16k16_ld_a_s8(sub_a.data, accA.get_pointer(), 16, 0);
            joint_matrix_load(sg, sub_b, accB.get_pointer(), 16);
            //__imma_m16n16k16_ld_b_s8(sub_b.data, accB.get_pointer(), 16, 0);

            joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          }

          joint_matrix_store(sg, sub_c, accC.get_pointer(), 16);
        });
  });

  return 0;
};
