#include "knn_kernel_gpu.h"

#include <iostream>
#include <numeric>
#include <vector>
#include <tuple>

using namespace std;

template <class T>
using row_col_val = tuple<unsigned int, unsigned int, T>;

void get_mat(int m, int n, float sparsity,
             vector<int> &row, vector<int> &col, vector<float> &val) {
  vector<row_col_val<float>> result;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float prob = (float) rand() / RAND_MAX;

      if (prob < sparsity) {
        row.push_back(i);
        col.push_back(j);
        val.push_back((float) rand() / RAND_MAX);
      }
    }
  }
}

void print_mat(vector<int> &row, vector<int> &col, vector<float> &val,
               int m, int n) {
  cout << m << " " << n << endl;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      bool found = false;
      float x = 0.0;

      for (unsigned int idx = 0; idx < row.size(); idx++) {
        if (row[idx] == i && col[idx] == j) {
          found = true;
          x = val[idx];
        }
      }

      if (found)
        cout << x << " ";
      else
        cout << 0.0 << " ";
    }

    cout << endl;
  }
}

int main() {
  int d = 128;
  int m = 128;
  int n = 64;

  vector<int> Q_row, R_row;
  vector<int> Q_col, R_col;
  vector<float> Q_val, R_val;

  get_mat(d, m, 0.01, Q_row, Q_col, Q_val);
  get_mat(d, n, 0.01, R_row, R_col, R_val);

  knn(Q_row, Q_col, Q_val, R_row, R_col, R_val, d, m, n);

  return 0;
}
