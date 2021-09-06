#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::experimental::matrix;

// This matrix size only requires a single subgroup operation: The "big matrix"
// matches the size of a tile "sub matrix" for A,B,C: i.e. N=M=K=n=m=k=16.
// Optimizations such as memory paddings are removed in order to aid clarity.
// Optimizations for avoiding bank conflicts can be added following e.g. https://github.com/NVIDIA/cuda-samples/blob/master/Samples/immaTensorCoreGemm/immaTensorCoreGemm.cu

// These guys define the unit size of the matrix per subgroup operation
constexpr int M = 16; //number of rows of C/D sub-matrices, number of cols of B sub-matrix
constexpr int N = 16; //number of cols of C/D sub-matrices, number of rows of A sub-matrix
constexpr int K = 16; //number of cols of A/number of rows of B sub-matrices

constexpr int N_THREADS_PER_MATRIX_OP = 32; // the number of threads per MMA subgroup is always 32 for cuda

constexpr int TILES_M = 1; //number of submatrices per row of C/D matrices
constexpr int TILES_N = 1; //number of submatrices per col of C/D matrices
constexpr int TILES_K = 1; //number of submatrices per col of A/per row of B, matrices

constexpr int BIG_M = TILES_M*M; //total number of M dimension matrix elements
constexpr int BIG_N = TILES_N*N; //total number of N dimension matrix elements
constexpr int BIG_K = TILES_K*K; //total number of N dimension matrix elements 

// The stride should equal the number of cols the "big matrix" (i.e. the length of each row that is skipped). The stride tells the implementation how many elements to skip in memory between consecutive subgroup operations.
constexpr int STRIDE = TILES_N * N;

int main() {

  int32_t C[BIG_M * BIG_N];
  int32_t D[BIG_M * BIG_N];

  int32_t A[BIG_N * BIG_K];
  int32_t B[BIG_K * BIG_M];


  for (int i = 0; i < BIG_N*BIG_M; i++) {
    C[i] = 3;
    D[i] = 0;
  }

  for (int i = 0; i < BIG_N*BIG_K; i++) {
    A[i] = 2;
  }

  for (int i = 0; i < BIG_K*BIG_M; i++) {
    B[i] = 1;
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


    cgh.parallel_for<class imatrix>(
        nd_range<2>({TILES_M, TILES_N*N_THREADS_PER_MATRIX_OP},
                    {1, N_THREADS_PER_MATRIX_OP}),
        [=](nd_item<2> item)

        {
          sub_group sg = item.get_sub_group();

          const auto m = item.get_global_id(0) * M; // row coordinate of start point of BIG C matrix
          const auto n = item.get_global_id(1) * N / N_THREADS_PER_MATRIX_OP; // column coordinate of start point of BIG C matrix

          joint_matrix<sub_group, matrix_type::c, M, N,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<sub_group, matrix_type::a, N, K,
                       matrix_layout::row_major>
              sub_a;

          joint_matrix<sub_group, matrix_type::b, K, M,
                       matrix_layout::row_major>
              sub_b;

          // Note that matrix "C" here is acting as matrices C,D in ptx
          // terminology. It is added to the matrix product of A and B. calls:
          
         //calls __imma_m16n16k16_ld_c(sub_c.data, accC.get_pointer() + ..., 0, 0);
         // TODO: one of these must be wrong: c or a
         joint_matrix_load(sg, sub_c, accC.get_pointer() + m * BIG_N  + n, STRIDE);
          // if TILES_K > 1 then we would call this look and 
          for (int k = 0; k < TILES_K * K; k += K) // row/col coordinate of start point of BIG A/B matrices
          {
            joint_matrix_load(sg, sub_a, accA.get_pointer() + k * BIG_N + n, STRIDE);
            //__imma_m16n16k16_ld_a_s8(sub_a.data, accA.get_pointer() + ..., 16, 0);
            joint_matrix_load(sg, sub_b, accB.get_pointer() + m * BIG_K + k, STRIDE);
            //__imma_m16n16k16_ld_b_s8(sub_b.data, accB.get_pointer() + ..., 16, 0);

            sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          }
          // calls //__imma_m16n16k16_st_c_i32(accD.get_pointer() + ..., sub_c.data, 0, 0);
          joint_matrix_store(sg, sub_c, accD.get_pointer() + m * BIG_N  + n, STRIDE);

    

          
        });
  });
//for (auto& c : C)
//assert(c == 0);
for (auto& d : D)
  std::cout << "\nThe value of D[0]= " << d;

  return 0;
};
