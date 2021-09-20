#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::experimental::matrix;

// Optimizations such as memory paddings are not included in order to aid clarity.
// Optimizations for avoiding bank conflicts can be added following e.g. https://github.com/NVIDIA/cuda-samples/blob/master/Samples/immaTensorCoreGemm/immaTensorCoreGemm.cu
// This example forms a matrix from a single "TILE" using cuda example terminology.  Multiple TILES can be used to construct yet larger matrices.

// These guys define the unit size of the matrix per subgroup operation
constexpr int M = 8; //number of rows of C/D sub-matrices, number of cols of B sub-matrix
constexpr int N = 8; //number of cols of C/D sub-matrices, number of rows of A sub-matrix
constexpr int K = 4; //number of cols of A/number of rows of B sub-matrices

constexpr int N_THREADS_PER_MATRIX_OP = 32; // the number of threads per MMA subgroup is always 32 for cuda

// This matrix size currently set only requires a single subgroup operation: The "big matrix"
// matches the size of a subtile "sub matrix" for A,B,C: i.e. N=M=K=n=m=k=16.

// if SUB_TILES_M or SUB_TILES_N > 1 then getting incorrect result.
constexpr int SUB_TILES_M = 1; //number of submatrices per row of C/D matrices
constexpr int SUB_TILES_N = 1; //number of submatrices per col of C/D matrices
constexpr int SUB_TILES_K = 10; //number of submatrices per col of A/per row of B, matrices

constexpr int BIG_M = SUB_TILES_M*M; //total number of M dimension matrix elements
constexpr int BIG_N = SUB_TILES_N*N; //total number of N dimension matrix elements
constexpr int BIG_K = SUB_TILES_K*K; //total number of N dimension matrix elements

// The stride should equal the number of elements between consecutive rows (columns for Matrix B) of the "big matrix". Assuming all matrices are indexed row major.
// The stride tells the implementation how many elements to skip in memory matrix row/column multiplications.
constexpr int STRIDE_A = BIG_K; // row major
constexpr int STRIDE_B = BIG_K; // col major: e.g. if row major should equal BIG_M.
constexpr int STRIDE_C = BIG_N; // row major

// function that returns correct mn element of matrix D.
double matrix_mn(double* _A, double * _B, double* _C, int m, int n)
{

double res = _C[m * BIG_N + n];

for(int k = 0; k < BIG_K; k++)
  //res += _A[n * BIG_K + k] * _B[k * BIG_M + m]; // b row leading
  res += _A[n * BIG_K + k] * _B[m * BIG_K + k]; // b column leading

return res;

}

int main() {

  double C[BIG_M * BIG_N];
  double D[BIG_M * BIG_N];

  double A[BIG_N * BIG_K];
  double B[BIG_K * BIG_M];

  for (int i = 0; i < BIG_N*BIG_M; i++) {
    C[i] = i;
    D[i] = 0;
  }

  for (int i = 0; i < BIG_N*BIG_K; i++) {
    A[i] = i;
  }

  for (int i = 0; i < BIG_K*BIG_M; i++) {
    B[i] = i;
  }


  buffer<double, 1> bufC(C, range<1>(BIG_N*BIG_M));
  buffer<double, 1> bufD(D, range<1>(BIG_N*BIG_M));

  buffer<double, 1> bufA(A, range<1>(BIG_N*BIG_K));
  buffer<double, 1> bufB(B, range<1>(BIG_K*BIG_M));


  queue q;
  q.submit([&](handler &cgh) {

    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);

    auto accD = bufD.get_access<access::mode::read_write>(cgh);

range<2> sGroup = {1, N_THREADS_PER_MATRIX_OP};

    cgh.parallel_for<class imatrix>(
        nd_range<2>({SUB_TILES_M , SUB_TILES_N*N_THREADS_PER_MATRIX_OP}, 
                    sGroup), //group range should match the sub-group range for given architecture
        [=](nd_item<2> item)

        {
          // this guy does nothing but has to be past to matrix interfaces (following AMX implementation).
          sub_group sg = item.get_sub_group();

          const auto m = item.get_group().get_id()[0]; // row id of current submatrix of BIG C matrix
          const auto n = item.get_group().get_id()[1]; // column id of current submatrix of BIG C matrix

          joint_matrix<sub_group, matrix_type::c, matrix_layout::row_major, M, N>
          sub_c;

          joint_matrix<sub_group, matrix_type::a, matrix_layout::row_major, N, K>
          sub_a;

          joint_matrix<sub_group, matrix_type::b, matrix_layout::col_major, K, M>
          sub_b;

         //calls e.g. __imma_m16n16k16_ld_c(sub_c.data, accC.get_pointer() + ..., 0, 0);
         // Third argument should be the number of elements to skip from beginning of BIG matrix to start of sub matrix represented by group.
         joint_matrix_load(sg, sub_c, accC.get_pointer() + (m * M) * BIG_N  + n * N, STRIDE_C);

          for (int k = 0; k < SUB_TILES_K; k += 1) // row/col id of current submatrix of BIG A/B matrices
          {
            joint_matrix_load(sg, sub_a, accA.get_pointer() + (k * K) + (n * N * BIG_K), STRIDE_A);

            // stride in memory also will depend on the row/comumn major
            joint_matrix_load(sg, sub_b, accB.get_pointer() + (k * K) + (m * M * BIG_K), STRIDE_B, matrix_layout::col_major); // works as normal matrix multiplication in this case.
	          //joint_matrix_load(sg, sub_b, accB.get_pointer() + (k * K * BIG_M) + (m * M), STRIDE_B); //row major: using this returns C + the transpose of A*B which is a bit weird.
            //__imma_m16n16k16_ld_b_s8(sub_b.data, accB.get_pointer() + ..., 16, 0);

            sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
}
          // calls //__imma_m16n16k16_st_c_i32(accD.get_pointer() + ..., sub_c.data, 0, 0);
          joint_matrix_store(sg, sub_c, accD.get_pointer() + (m * M) * BIG_N  + n * N, STRIDE_C);

        });
  });

const auto host_accessor =
        bufD.get_access<cl::sycl::access::mode::read>();
for (int m = 0; m < BIG_M; m++)
    for (int n = 0; n < BIG_N; n++)
{
assert(host_accessor[m * BIG_N + n] == matrix_mn(A, B , C , m , n));
  std::cout << "\nThe value of D[i,j] = " << host_accessor[m * BIG_N + n];
  std::cout << "\nThe value of D_check[i,j] = " << matrix_mn(A, B , C , m , n);
}
  return 0;
};
