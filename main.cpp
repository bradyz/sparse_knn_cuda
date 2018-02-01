#include "sparse_matrix.h"
#include "dense_matrix.h"

#include <vector>
#include <tuple>

using namespace std;

template <class T>
using row_col_val = tuple<unsigned int, unsigned int, T>;

DenseMatrix<float> get_A() {
  DenseMatrix<float> result(4, 4);

  result.assign(0, 0, 2.0);
  result.assign(1, 1, 2.0);
  result.assign(2, 2, 2.0);
  result.assign(3, 3, 2.0);

  return result;
}

DenseMatrix<float> get_B() {
  DenseMatrix<float> result(4, 4);

  result.assign(0, 0, 1.0);
  result.assign(1, 1, 2.0);
  result.assign(2, 2, 3.0);
  result.assign(3, 3, 5.0);

  return result;
}

void knn(const DenseMatrix<float> &Q, const DenseMatrix<float> &R) {
  // Q is (d, m), R is (d, n).

  // C is (m, n).
  DenseMatrix<float> C = (-SparseMatrix<float>(Q.transpose())).mat_mul(R);

  cout << C << endl;
}

int main() {
  DenseMatrix<float> A = get_A();
  DenseMatrix<float> B = get_B();

  knn(A, B);

  return 0;
}
