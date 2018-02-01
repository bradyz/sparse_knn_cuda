#include "knn_kernel.h"

#include "dense_matrix.h"
#include "sparse_matrix.h"

#include <vector>

using namespace std;

vector<float> get_sq_norms(const DenseMatrix<float> &mat) {
  vector<float> result(mat.n, 0.0);

  for (int j = 0; j < mat.n; j++)
    for (int i = 0; i < mat.m; i++)
      result[j] += mat(i, j) * mat(i, j);

  return result;
}

void knn(const DenseMatrix<float> &Q, const DenseMatrix<float> &R) {
  // Q is size d x m.
  // R is size d x n.
  // C is size m x n.
  DenseMatrix<float> C = (-SparseMatrix<float>(Q.transpose())).mat_mul(R);

  vector<float> Q_norms = get_sq_norms(Q);
  vector<float> R_norms = get_sq_norms(R);

  for (int i = 0; i < C.m; i++)
    for (int j = 0; j < C.n; j++)
      C.assign(i, j, C(i, j) + Q_norms[i] + R_norms[j]);

  // Neighbor search.
  vector<vector<pair<float, int>>> result;

  for (int i = 0; i < C.m; i++) {
    result.push_back(vector<pair<float, int>>());

    for (int j = 0; j < C.n; j++)
      result[i].push_back(pair<float, int>(C(i, j), j));

    sort(result[i].begin(), result[i].end());
  }
}
