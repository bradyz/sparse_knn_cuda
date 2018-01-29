#include "sparse_matrix.h"

#include <vector>
#include <tuple>

using namespace std;

template <class T>
using row_col_val = tuple<unsigned int, unsigned int, T>;

int main() {
  int m = 4;
  int n = 4;

  vector<row_col_val<float>> values;

  values.push_back(row_col_val<float>(0, 0, 1.0));
  values.push_back(row_col_val<float>(1, 1, 2.0));
  values.push_back(row_col_val<float>(3, 1, 3.0));
  values.push_back(row_col_val<float>(0, 2, 4.0));
  values.push_back(row_col_val<float>(2, 2, 5.0));

  SparseMatrix<float> tmp(m, n, values);

  cout << tmp << endl;

  return 0;
}
