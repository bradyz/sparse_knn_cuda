#include "sparse_matrix.h"
#include "dense_matrix.h"

#include <cassert>
#include <cmath>
#include <vector>
#include <tuple>

using namespace std;

float EPSILON = 1e-6;

// Reads from stdin
// m n
// A_11, A_12, ..., A_1m
// A_21, A_22, ..., A_2m
// ...
// A_n1, A_n2, ..., A_nm
DenseMatrix<float> parse() {
  int m, n;

  cin >> m >> n;

  DenseMatrix<float> mat(m, n);

  float tmp;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      cin >> tmp;

      mat.assign(i, j, tmp);
    }
  }

  return mat;
}

void assert_close(const DenseMatrix<float> &A, const DenseMatrix<float> &B) {
  assert(A.m == B.m);
  assert(A.n == B.n);

  for (unsigned int i = 0; i < A.m; i++) {
    for (unsigned int j = 0; j < A.n; j++) {
      float relative_error = fabs(A(i, j) - B(i, j)) / (B(i, j) + EPSILON);

      assert(relative_error < EPSILON);
    }
  }
}

int main() {
  DenseMatrix<float>A = parse();
  DenseMatrix<float>B = parse();
  DenseMatrix<float>C = parse();

  DenseMatrix<float>C_prime = SparseMatrix<float>(A).mat_mul(B);

  assert_close(C, C_prime);

  cout << "pass." << endl;

  return 0;
}
