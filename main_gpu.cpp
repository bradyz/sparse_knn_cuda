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
      float prob = (float) rand() / (float) RAND_MAX;

      if (prob < sparsity) {
        row.push_back(i);
        col.push_back(j);
        val.push_back((float) rand() / (float) RAND_MAX);
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

void doit() {
  int sizes[] = {512, 1024, 2048, 4096};
  int dimensions[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  float sparsities[] = {0.001, 0.01, 0.1, 0.5, 1.0};

  // sizes[0] = 512;
  // dimensions[0] = 512;
  // sparsities[0] = 1.0;

  int k_unused = 0;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < 5; k++) {
        int n = sizes[i];
        int m = n;

        int d = dimensions[j];
        float sparsity = sparsities[k];

        vector<int> Q_row, R_row;
        vector<int> Q_col, R_col;
        vector<float> Q_val, R_val;

        cout << n << " " << d << " " << sparsity << endl;

        get_mat(d, m, sparsity, Q_row, Q_col, Q_val);
        get_mat(d, n, sparsity, R_row, R_col, R_val);

        // print_mat(Q_row, Q_col, Q_val, d, m);
        // print_mat(R_row, R_col, R_val, d, n);

        knn(Q_row, Q_col, Q_val, R_row, R_col, R_val, d, m, n, k_unused);
      }
    }
  }
}

int main() {
  doit();
  doit();

  return 0;
}
