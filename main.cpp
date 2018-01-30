#include "sparse_matrix.h"
#include "dense_matrix.h"

#include <vector>
#include <tuple>

using namespace std;

template <class T>
using row_col_val = tuple<unsigned int, unsigned int, T>;

SparseMatrix<float> sample_A() {
  vector<row_col_val<float>> values;

  values.push_back(row_col_val<float>(0, 0, 2.0));
  values.push_back(row_col_val<float>(1, 1, 2.0));
  values.push_back(row_col_val<float>(2, 2, 1.0));
  values.push_back(row_col_val<float>(3, 3, 1.0));

  return SparseMatrix<float>(4, 4, values);
}

SparseMatrix<float> sample_B() {
  vector<row_col_val<float>> values;

  values.push_back(row_col_val<float>(0, 0, 2.0));
  values.push_back(row_col_val<float>(1, 1, 3.0));
  values.push_back(row_col_val<float>(2, 2, 4.0));
  values.push_back(row_col_val<float>(3, 3, 5.0));

  return SparseMatrix<float>(4, 4, values);
}

int main() {
  SparseMatrix<float> A = sample_A();
  SparseMatrix<float> B = sample_B();

  DenseMatrix<float> C = A.mat_mul(B);

  cout << C << endl;

  return 0;
}
