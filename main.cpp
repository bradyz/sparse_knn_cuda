#include "sparse_matrix.h"
#include "dense_matrix.h"
#include "knn_kernel.h"

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

int main() {
  DenseMatrix<float> A = get_A();
  DenseMatrix<float> B = get_B();

  knn(A, B);

  return 0;
}
