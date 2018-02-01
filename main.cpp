#include "sparse_matrix.h"
#include "dense_matrix.h"
#include "knn_kernel.h"

#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>

using namespace std;

int num_reference = 1000000;
int num_query = 512;
int dim_sizes[] = {512, 1024, 2048};
float sparsity_levels[] = {0.001, 0.01, 0.1, 0.5};

DenseMatrix<float> get_mat(int m, int n, float sparsity) {
  DenseMatrix<float> result(m, n);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float prob = rand() / RAND_MAX;

      if (prob < sparsity)
        result.assign(i, j, rand() / RAND_MAX);
    }
  }

  return result;
}

int main() {
  for (int d : dim_sizes) {
    for (float sparsity : sparsity_levels) {
      DenseMatrix<float> Q = get_mat(d, num_query, sparsity);
      DenseMatrix<float> R = get_mat(d, num_reference, sparsity);

      cout << "d: " << d << endl;
      cout << "sparsity: " << sparsity << endl;

      knn(Q, R);
    }
  }

  return 0;
}
