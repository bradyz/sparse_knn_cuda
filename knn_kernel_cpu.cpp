#include "knn_kernel_cpu.h"

#include "dense_matrix.h"
#include "sparse_matrix.h"

#include <ctime>
#include <algorithm>
#include <vector>

using namespace std;

vector<float> get_sq_norms(const DenseMatrix<float> &mat) {
  vector<float> result(mat.n, 0.0);

  for (unsigned int j = 0; j < mat.n; j++)
    for (unsigned int i = 0; i < mat.m; i++)
      result[j] += mat(i, j) * mat(i, j);

  return result;
}

void knn(const DenseMatrix<float> &Q, const DenseMatrix<float> &R) {
  vector<float> Q_norms = get_sq_norms(Q);
  vector<float> R_norms = get_sq_norms(R);

  time_t start = time(0);

  // Q is size d x m.
  // R is size d x n.
  // C is size m x n.
  DenseMatrix<float> C = (-SparseMatrix<float>(Q.transpose())).mat_mul(R);

  for (unsigned int i = 0; i < C.m; i++)
    for (unsigned int j = 0; j < C.n; j++)
      C.assign(i, j, C(i, j) + Q_norms[i] + R_norms[j]);

  time_t finish_matmul = time(0);

  // Neighbor search.
  vector<vector<pair<float, int>>> result(C.m);

  for (unsigned int i = 0; i < C.m; i++) {
    for (unsigned int j = 0; j < C.n; j++)
      result[i].push_back(pair<float, int>(C(i, j), j));

    sort(result[i].begin(), result[i].end());
  }

  time_t finish_search = time(0);

  double time_matmul = difftime(finish_matmul, start);
  double time_search = difftime(finish_search, finish_matmul);

  cout << "Matmul (sec): " << time_matmul << endl;
  cout << "Search (sec): " << time_search << endl;
}
