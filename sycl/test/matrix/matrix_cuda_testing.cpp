#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::experimental::matrix;

// Optimizations such as memory paddings are not included in order to aid clarity.
// Optimizations for avoiding bank conflicts can be added following e.g. https://github.com/NVIDIA/cuda-samples/blob/master/Samples/immaTensorCoreGemm/immaTensorCoreGemm.cu
// This example forms a matrix from a single "TILE" using cuda example terminology.  Multiple TILES can be used to construct yet larger matrices.

// These guys define the unit size of the matrix per subgroup operation
constexpr int M = 16; //number of rows of C/D sub-matrices, number of cols of B sub-matrix
constexpr int N = 16; //number of cols of C/D sub-matrices, number of rows of A sub-matrix
constexpr int K = 16; //number of cols of A/number of rows of B sub-matrices

constexpr int N_THREADS_PER_MATRIX_OP = 1; // the number of threads per MMA subgroup is always 32 for cuda

// This matrix size currently set only requires a single subgroup operation: The "big matrix"
// matches the size of a subtile "sub matrix" for A,B,C: i.e. N=M=K=n=m=k=16.
constexpr int SUB_TILES_M = 1; //number of submatrices per row of C/D matrices
constexpr int SUB_TILES_N = 1; //number of submatrices per col of C/D matrices
constexpr int SUB_TILES_K = 1; //number of submatrices per col of A/per row of B, matrices

constexpr int BIG_M = SUB_TILES_M*M; //total number of M dimension matrix elements
constexpr int BIG_N = SUB_TILES_N*N; //total number of N dimension matrix elements
constexpr int BIG_K = SUB_TILES_K*K; //total number of N dimension matrix elements

// The stride should equal the number of elements between consecutive rows (columns for Matrix B) of the "big matrix". Assuming all matrices are indexed row major.
// The stride tells the implementation how many elements to skip in memory matrix row/column multiplications.
constexpr int STRIDE_A = BIG_N; // row major
constexpr int STRIDE_B = BIG_K; // row major: e.g. if column major should equal BIG_N?
constexpr int STRIDE_C = BIG_N; // row major


int main() {

  int32_t C[BIG_M * BIG_N];
  int32_t D[BIG_M * BIG_N];

  int32_t A[BIG_N * BIG_K];
  int32_t B[BIG_K * BIG_M];


  for (int i = 0; i < BIG_N*BIG_M; i++) {
    C[i] = 5;
    D[i] = 0;
  }

  for (int i = 0; i < BIG_N*BIG_K; i++) {
    A[i] = 3;
  }

  for (int i = 0; i < BIG_K*BIG_M; i++) {
    B[i] = 3;
  }

  buffer<int32_t, 1> bufC(C, range<1>(BIG_N*BIG_M));
  buffer<int32_t, 1> bufD(D, range<1>(BIG_N*BIG_M));

  buffer<int32_t, 1> bufA(A, range<1>(BIG_N*BIG_K));
  buffer<int32_t, 1> bufB(B, range<1>(BIG_K*BIG_M));


  queue q;
  q.submit([&](handler &cgh) {

    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);

    auto accD = bufD.get_access<access::mode::read_write>(cgh);

range<2> sGroup = {1, N_THREADS_PER_MATRIX_OP};

    cgh.parallel_for<class imatrix>(
        nd_range<2>({SUB_TILES_M, SUB_TILES_N*N_THREADS_PER_MATRIX_OP}, //TODO observe effect of changing this on ptx..
                    sGroup), //group range should match the sub-group range for given architecture
        [=](nd_item<2> item)

        {
          sub_group sg = item.get_sub_group();

          const auto m = item.get_group().get_id()[0]; // row id of current submatrix of BIG C matrix
          const auto n = item.get_group().get_id()[1]; // column id of current submatrix of BIG C matrix

          joint_matrix<sub_group, matrix_type::c, M, N,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<sub_group, matrix_type::c, M, N,
                       matrix_layout::row_major>
              sub_d;

          joint_matrix<sub_group, matrix_type::a, N, K,
                       matrix_layout::row_major>
              sub_a;

          joint_matrix<sub_group, matrix_type::b, K, M,
                       matrix_layout::row_major>
              sub_b;

          // Note that matrix "C" here is acting as matrices C,D in ptx
          // terminology. It is added to the matrix product of A and B. calls:

         //calls __imma_m16n16k16_ld_c(sub_c.data, accC.get_pointer() + ..., 0, 0);
         // Third argument should be the number of elements to skip from beginning of BIG matrix to start of sub matrix represented by group.
         //joint_matrix_load(sg, sub_c, accC.get_pointer() + (m * M) * BIG_N  + n * N, STRIDE_C);
		 joint_matrix_load(sg, sub_c, accC.get_pointer(), STRIDE_C);

          for (int k = 0; k < SUB_TILES_K * K; k += K) // row/col id of current submatrix of BIG A/B matrices
          {
            //joint_matrix_load(sg, sub_a, accA.get_pointer() + (k * K) * BIG_N + n * N, STRIDE_A);
			joint_matrix_load(sg, sub_a, accA.get_pointer(), STRIDE_A);
            //__imma_m16n16k16_ld_a_s8(sub_a.data, accA.get_pointer() + ..., 16, 0);
			joint_matrix_load(sg, sub_b, accB.get_pointer(), STRIDE_B);
            //__imma_m16n16k16_ld_b_s8(sub_b.data, accB.get_pointer() + ..., 16, 0);

            sub_d = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
}
          // calls //__imma_m16n16k16_st_c_i32(accD.get_pointer() + ..., sub_c.data, 0, 0);
          //joint_matrix_store(sg, sub_d, accD.get_pointer() + (m * M) * BIG_N  + n * N, STRIDE_C);
		  joint_matrix_store(sg, sub_d, accD.get_pointer(), STRIDE_C); //subgroup arg....

        });
  });

const auto host_accessor =
        bufD.get_access<cl::sycl::access::mode::read>();
for (int i = 0; i < 256; i++)
  std::cout << "\nThe value of D[0]= " << host_accessor[i];

  return 0;
};
