#include "../include/spgsknn.hpp"

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

void benchmark(int m, int n, int k, int d, float s) {
  vector<int> Q_row, R_row;
  vector<int> Q_col, R_col;
  vector<float> Q_val, R_val;

  cout << n << " " << d << " " << k << " " << s << endl;

  get_mat(d, m, s, Q_row, Q_col, Q_val);
  get_mat(d, n, s, R_row, R_col, R_val);

  // print_mat(Q_row, Q_col, Q_val, d, m);
  // print_mat(R_row, R_col, R_val, d, n);

  vector<float> distances;
  vector<int> indices;

  spgsknn(d, m, n, k,
      Q_row, Q_col, Q_val, R_row, R_col, R_val,
      distances, indices);

  for (int i = 0; i < m; i++) {
    for (int j = 1; j < k; j++) {
      if (distances[i + j * m] < distances[i + (j-1) * m]) {
        cout << "break" << endl;
        exit(1);
      }
    }
  }
}

void doit() {
  int sizes[] = {512, 1024, 2048, 4096};
  int dimensions[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  float sparsities[] = {0.001, 0.01, 0.1, 0.5, 1.0};
  int neighbors[] = {8, 64, 256, 512, 1024, 2048, 4096};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < 5; k++) {
        for (int l = 0; l < 7; l++) {
          int n = sizes[i];
          int m = n;
          int k_neighbors = neighbors[l];

          if (k_neighbors > n)
            continue;

          int d = dimensions[j];
          float sparsity = sparsities[k];

          benchmark(m, n, k_neighbors, d, sparsity);
        }
      }
    }
  }
}

int main() {
  doit();
  doit();

  return 0;
}
