#include "knn_kernel_gpu.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>
#include <map>
#include <chrono>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include "cusparse.h"

#include <omp.h>

using namespace std;

template <class T>
using row_col_val = tuple<unsigned int, unsigned int, T>;

inline void check(cudaError_t status, string error) {
  if (status != cudaSuccess) {
    cout << error << endl;
    exit(1);
  }
}

inline void check(cusparseStatus_t status, string error) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cout << error << endl;
    exit(1);
  }
}

template <class T>
void sort_by_col_row(vector<row_col_val<T>> &triplets) {
  sort(triplets.begin(), triplets.end(),
      [](const row_col_val<T> &u, const row_col_val<T> &v) {
        if (get<1>(u) == get<1>(v))
          return get<0>(u) < get<0>(v);

        return get<1>(u) < get<1>(v);
      });
}

template <class T>
void populate_sq_norms(const vector<int> &A_col, const vector<T> &A_val,
                       vector<T> &A_sq_norms) {
  for (unsigned int i = 0; i < A_val.size(); i++)
    A_sq_norms[A_col[i]] += A_val[i] * A_val[i];
}

template <class T>
void cpu_sort(const vector<int> &host_row, const vector<int> &host_col,
              const vector<T> &host_val,
              const vector<T> &Q_sq_norms, const vector<T> &R_sq_norms,
              int k) {
  int m = Q_sq_norms.size();
  int n = R_sq_norms.size();

  // Save the inner products.
  map<pair<int, int>, float> values;

  int val_idx = 0;

  for (unsigned int i = 0; i < m; i++) {
    for (int j_idx = host_row[i]; j_idx < host_row[i+1]; j_idx++) {
      int j = host_col[j_idx];

      values[make_pair(i, j)] = host_val[val_idx++];
    }
  }

  // Sort.
  vector<vector<int>> neighbors(m, vector<int>(n));

  #pragma omp parallel for
  for (int i = 0; i < m; i++) {
    iota(neighbors[i].begin(), neighbors[i].end(), 0);

    nth_element(neighbors[i].begin(), neighbors[i].begin() + k, neighbors[i].end(),
        [&](int j_1, int j_2) {
        T dist_1 = Q_sq_norms[i] + -2.0 * values[make_pair(i, j_1)] + R_sq_norms[j_1];
        T dist_2 = Q_sq_norms[i] + -2.0 * values[make_pair(i, j_2)] + R_sq_norms[j_2];

        return dist_1 < dist_2;
    });
  }
}

struct ZipComparator {
  __host__ __device__
  inline bool operator() (const thrust::tuple<unsigned int, float> &a,
                          const thrust::tuple<unsigned int, float> &b) {
    if(a.head < b.head)
      return true;
    else if(a.head == b.head)
      return a.tail < b.tail;
    return false;
  }
};

template <class T>
void gpu_sort(const vector<int> &host_row, const vector<int> &host_col,
              const vector<T> &host_val,
              const vector<T> &Q_sq_norms, const vector<T> &R_sq_norms,
              int k) {
  int m = Q_sq_norms.size();
  int n = Q_sq_norms.size();

  int val_idx = 0;

#pragma omp parallel for
  for (unsigned int i = 0; i < m; i++) {
    vector<int> idx(n);
    vector<T> dist(n);

    /* cout << omp_get_thread_num() << " " << omp_get_num_threads() << endl; */
    /* cout << i << " " << m << endl; */

    for (int j_idx = host_row[i]; j_idx < host_row[i+1]; j_idx++) {
      int j = host_col[j_idx];

      dist[j] = -2.0 * host_val[val_idx++];
    }

    for (unsigned int j = 0; j < n; j++)
      dist[j] += Q_sq_norms[i] + R_sq_norms[j];

    thrust::device_vector<int> gpu_idx(n);
    thrust::sequence(gpu_idx.begin(), gpu_idx.end());

    thrust::device_vector<T> gpu_dist(dist.begin(), dist.end());

    thrust::sort_by_key(gpu_dist.begin(), gpu_dist.end(), gpu_idx.begin());

    thrust::copy(gpu_idx.begin(), gpu_idx.end(), &idx[0]);
  }
}

template <class T>
void coo_to_csr(vector<int> &A_row, vector<int> &A_col, vector<T> &A_val,
                unsigned int m, cusparseHandle_t handle,
                int *&row_csr, int *&col_csr, T *&val_csr) {
  int *row_coo = 0;

  check(cudaMalloc((void**) &row_coo, A_row.size() * sizeof(T)),
        "coo malloc failed");

  check(cudaMalloc((void**) &row_csr, (m+1) * sizeof(T)),
        "csr row malloc failed");

  check(cudaMalloc((void**) &col_csr, A_row.size() * sizeof(T)),
        "csr col malloc failed");

  check(cudaMalloc((void**) &val_csr, A_row.size() * sizeof(T)),
        "csr val malloc failed");

  check(cudaMemcpy(row_coo, &A_row[0], (size_t) (A_row.size() * sizeof(int)),
                   cudaMemcpyHostToDevice),
        "copy to row failed");

  check(cudaMemcpy(col_csr, &A_col[0], (size_t) (A_col.size() * sizeof(int)),
                   cudaMemcpyHostToDevice),
        "copy to col failed");

  check(cudaMemcpy(val_csr, &A_val[0],
                   (size_t) (A_col.size() * sizeof(T)), cudaMemcpyHostToDevice),
        "copy to val failed");

  check(cusparseXcoo2csr(handle, row_coo, A_row.size(),
                         m, row_csr, CUSPARSE_INDEX_BASE_ZERO),
        "convert failed");

  check(cudaFree(row_coo), "free coo");
}

template <class T>
void inner_product(vector<int> &Q_row, vector<int> &Q_col, vector<T> &Q_val,
                   vector<int> &R_row, vector<int> &R_col, vector<T> &R_val,
                   unsigned int d, unsigned int m, unsigned int n,
                   vector<int> &host_row, vector<int> &host_col, vector<T> &host_val) {
  cusparseHandle_t handle = 0;

  check(cusparseCreate(&handle), "initialization failed");

  int *Q_row_csr = 0;
  int *Q_col_csr = 0;
  T *Q_val_csr = 0;

  int *R_row_csr = 0;
  int *R_col_csr = 0;
  T *R_val_csr = 0;

  coo_to_csr(Q_row, Q_col, Q_val, d, handle, Q_row_csr, Q_col_csr, Q_val_csr);
  coo_to_csr(R_row, R_col, R_val, d, handle, R_row_csr, R_col_csr, R_val_csr);

  int *C_row_csr = 0;
  int *C_col_csr = 0;
  T *C_val_csr = 0;

  int C_nnz = -1;

  cusparseMatDescr_t real_sparse_desc = 0;

  check(cusparseCreateMatDescr(&real_sparse_desc),
        "create failed");
  check(cusparseSetMatType(real_sparse_desc, CUSPARSE_MATRIX_TYPE_GENERAL),
        "set 1 failed");
  check(cusparseSetMatIndexBase(real_sparse_desc, CUSPARSE_INDEX_BASE_ZERO),
        "set 2 failed");

  check(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST),
        "set pointer");
  check(cudaMalloc((void**) &C_row_csr, (m+1) * sizeof(int)),
        "malloc row fail");

  check(cusparseXcsrgemmNnz(handle,
                            CUSPARSE_OPERATION_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            m, n, d,
                            real_sparse_desc, Q_val.size(), Q_row_csr, Q_col_csr,
                            real_sparse_desc, R_val.size(), R_row_csr, R_col_csr,
                            real_sparse_desc, C_row_csr, &C_nnz),
        "gemm nnz");

  if (C_nnz == -1)
    exit(1);

  check(cudaMalloc((void**) &C_col_csr, C_nnz * sizeof(int)),
        "malloc device col");
  check(cudaMalloc((void**) &C_val_csr, C_nnz * sizeof(T)),
        "malloc device val");

  check(cusparseScsrgemm(handle,
                         CUSPARSE_OPERATION_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         m, n, d,
                         real_sparse_desc, Q_val.size(), Q_val_csr, Q_row_csr, Q_col_csr,
                         real_sparse_desc, R_val.size(), R_val_csr, R_row_csr, R_col_csr,
                         real_sparse_desc, C_val_csr, C_row_csr, C_col_csr),
        "gemm");

  check(cudaFree(Q_row_csr), "free row csr");
  check(cudaFree(Q_col_csr), "free col csr");
  check(cudaFree(Q_val_csr), "free val csr");

  check(cudaFree(R_row_csr), "free row csr");
  check(cudaFree(R_col_csr), "free col csr");
  check(cudaFree(R_val_csr), "free val csr");

  // Copy result back to CPU.
  host_row.resize(m+1, 0);
  host_col.resize(C_nnz, 0);
  host_val.resize(C_nnz, 0.0);

  check(cudaMemcpy(&host_row[0], C_row_csr, (size_t) ((m+1) * sizeof(int)),
                   cudaMemcpyDeviceToHost),
        "copy from failed row");
  check(cudaMemcpy(&host_col[0], C_col_csr, (size_t) (C_nnz * sizeof(int)),
                   cudaMemcpyDeviceToHost),
        "copy from failed col");
  check(cudaMemcpy(&host_val[0], C_val_csr, (size_t) (C_nnz * sizeof(T)),
                   cudaMemcpyDeviceToHost),
        "copy from failed val");

  check(cudaFree(C_row_csr), "free row csr");
  check(cudaFree(C_col_csr), "free col csr");
  check(cudaFree(C_val_csr), "free val csr");
}

template <class T>
void knn(vector<int> &Q_row, vector<int> &Q_col, vector<T> &Q_val,
         vector<int> &R_row, vector<int> &R_col, vector<T> &R_val,
         unsigned int d, unsigned int m, unsigned int n, unsigned int k) {
  cusparseHandle_t handle = 0;

  check(cusparseCreate(&handle), "initialization failed");

  auto start = chrono::high_resolution_clock::now();

  // Copy result back to CPU.
  vector<int> host_row;
  vector<int> host_col;
  vector<T> host_val;

  inner_product(Q_row, Q_col, Q_val, R_row, R_col, R_val, d, m, n,
                host_row, host_col, host_val);

  auto mult_done = chrono::high_resolution_clock::now();

  // Norms.
  vector<T> Q_sq_norms(m, 0.0);
  vector<T> R_sq_norms(n, 0.0);

  populate_sq_norms(Q_col, Q_val, Q_sq_norms);
  populate_sq_norms(R_col, R_val, R_sq_norms);

  auto norm_done = chrono::high_resolution_clock::now();

  gpu_sort(host_row, host_col, host_val, Q_sq_norms, R_sq_norms, k);

  auto sort_done = chrono::high_resolution_clock::now();

  float total = chrono::duration_cast<chrono::milliseconds>(sort_done-start).count();
  float mult = chrono::duration_cast<chrono::milliseconds>(mult_done-start).count();
  float norm = chrono::duration_cast<chrono::milliseconds>(norm_done-mult_done).count();
  float sort = chrono::duration_cast<chrono::milliseconds>(sort_done-norm_done).count();

  cout << "total: " << total / 1000.0 << endl;
  cout << "mult: " << mult / 1000.0 << endl;
  cout << "norm: " << norm / 1000.0 << endl;
  cout << "sort: " << sort / 1000.0 << endl;
}

// Possible instantiations.
template void knn(vector<int>&, vector<int>&, vector<float>&,
                  vector<int>&, vector<int>&, vector<float>&,
                  unsigned int, unsigned int, unsigned int, unsigned int);
