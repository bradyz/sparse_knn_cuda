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

template <class T>
void gpu_sort(int k) {
/*   int m = Q_sq_norms.size(); */
/*   int n = Q_sq_norms.size(); */
/*  */
/*   int val_idx = 0; */
/*  */
/* #pragma omp parallel for */
/*   for (unsigned int i = 0; i < m; i++) { */
/*     vector<int> idx(n); */
/*     vector<T> dist(n); */
/*  */
    /* cout << omp_get_thread_num() << " " << omp_get_num_threads() << endl; */
    /* cout << i << " " << m << endl; */
/*  */
/*     thrust::device_vector<int> gpu_idx(n); */
/*     thrust::sequence(gpu_idx.begin(), gpu_idx.end()); */
/*  */
/*     thrust::device_vector<T> gpu_dist(dist.begin(), dist.end()); */
/*  */
/*     thrust::sort_by_key(gpu_dist.begin(), gpu_dist.end(), gpu_idx.begin()); */
/*  */
/*     thrust::copy(gpu_idx.begin(), gpu_idx.end(), &idx[0]); */
/*   } */
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
T* sparse_to_dense(int *row, int *col, T *val,
                   int m, int n,
                   cusparseHandle_t handle, cusparseMatDescr_t desc) {
  T *A;

  check(cudaMalloc((void**) &A, m * n * sizeof(T)),
        "dense malloc failed");

  check(cusparseScsr2dense(handle, m, n, desc,
                           val, row, col, A, m),
        "csr 2 dense.");

  return A;
}

template <class T>
T* inner_product(
    int *Q_row_csr, int *Q_col_csr, T *Q_val_csr, unsigned int Q_nnz,
    int *R_row_csr, int *R_col_csr, T *R_val_csr, unsigned int R_nnz,
    unsigned int d, unsigned int m, unsigned int n, cusparseHandle_t handle) {
  int *C_row_csr = 0;
  int *C_col_csr = 0;
  T *C_val_csr = 0;

  int C_nnz = -1;

  cusparseMatDescr_t real_sparse_desc = 0;

  check(cusparseCreateMatDescr(&real_sparse_desc), "create failed");
  check(cusparseSetMatType(real_sparse_desc, CUSPARSE_MATRIX_TYPE_GENERAL), "set 1 failed");
  check(cusparseSetMatIndexBase(real_sparse_desc, CUSPARSE_INDEX_BASE_ZERO), "set 2 failed");

  check(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST), "set pointer");

  check(cudaMalloc((void**) &C_row_csr, (m+1) * sizeof(int)),
        "malloc row fail");

  check(cusparseXcsrgemmNnz(handle,
                            CUSPARSE_OPERATION_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            m, n, d,
                            real_sparse_desc, Q_nnz, Q_row_csr, Q_col_csr,
                            real_sparse_desc, R_nnz, R_row_csr, R_col_csr,
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
                         real_sparse_desc, Q_nnz, Q_val_csr, Q_row_csr, Q_col_csr,
                         real_sparse_desc, R_nnz, R_val_csr, R_row_csr, R_col_csr,
                         real_sparse_desc, C_val_csr, C_row_csr, C_col_csr),
        "gemm");

  T* C = sparse_to_dense(C_row_csr, C_col_csr, C_val_csr,
                         m, n, handle, real_sparse_desc);

  check(cudaFree(C_row_csr), "free row csr");
  check(cudaFree(C_col_csr), "free col csr");
  check(cudaFree(C_val_csr), "free val csr");

  return C;
}

__global__
void get_col_norms(int *col_csr, float *val_csr, float *sq_norms, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nnz)
      atomicAdd(&sq_norms[col_csr[i]], val_csr[i] * val_csr[i]);
}

__global__
void add_norms(float *C, float *Q_sq_norms, float *R_sq_norms, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = index / n;
    int j = index % n;

    // Cuda store column major.
    if (i < m && j < n)
      C[i + j * m] = Q_sq_norms[i] - 2.0 * C[i + j * m] + R_sq_norms[j];
}

template <class T>
void add_sq_norms(int *Q_col_csr, T *Q_val_csr, int Q_nnz,
                  int *R_col_csr, T *R_val_csr, int R_nnz,
                  T *C, int m, int n, cusparseHandle_t handle) {
  T *Q_sq_norms = 0;
  T *R_sq_norms = 0;

  check(cudaMalloc((void**) &Q_sq_norms, m * sizeof(T)), "coo malloc failed");
  check(cudaMalloc((void**) &R_sq_norms, n * sizeof(T)), "coo malloc failed");

  get_col_norms<<<(Q_nnz + 255) / 256, 256>>>(Q_col_csr, Q_val_csr, Q_sq_norms, Q_nnz);
  get_col_norms<<<(R_nnz + 255) / 256, 256>>>(R_col_csr, R_val_csr, R_sq_norms, R_nnz);

  add_norms<<<(m * n + 255) / 256, 256>>>(C, Q_sq_norms, R_sq_norms, m, n);
}

template <class T>
void knn(vector<int> &Q_row, vector<int> &Q_col, vector<T> &Q_val,
         vector<int> &R_row, vector<int> &R_col, vector<T> &R_val,
         unsigned int d, unsigned int m, unsigned int n, unsigned int k) {
  cusparseHandle_t handle = 0;

  check(cusparseCreate(&handle), "initialization failed");

  auto start = chrono::high_resolution_clock::now();

  int *Q_row_csr = 0;
  int *Q_col_csr = 0;
  T *Q_val_csr = 0;

  int *R_row_csr = 0;
  int *R_col_csr = 0;
  T *R_val_csr = 0;

  coo_to_csr(Q_row, Q_col, Q_val, d, handle, Q_row_csr, Q_col_csr, Q_val_csr);
  coo_to_csr(R_row, R_col, R_val, d, handle, R_row_csr, R_col_csr, R_val_csr);

  auto conversion_done = chrono::high_resolution_clock::now();

  T *C = inner_product(Q_row_csr, Q_col_csr, Q_val_csr, Q_val.size(),
                       R_row_csr, R_col_csr, R_val_csr, R_val.size(),
                       d, m, n, handle);

  auto mult_done = chrono::high_resolution_clock::now();

  add_sq_norms(
      Q_col_csr, Q_val_csr, Q_val.size(),
      R_col_csr, R_val_csr, R_val.size(),
      C, m, n, handle);

  auto norm_done = chrono::high_resolution_clock::now();

  /* gpu_sort(host_row, host_col, host_val, Q_sq_norms, R_sq_norms, k); */

  auto sort_done = chrono::high_resolution_clock::now();

  vector<float> C_host(m * n);

  check(cudaMemcpy(&C_host[0], C, (size_t) ((m * n) * sizeof(T)),
                   cudaMemcpyDeviceToHost),
        "copy from failed val asdfasdf");

  /* cout << m << " " << n << endl; */
  /* for (int i = 0; i < m; i++) { */
  /*   for (int j = 0; j < n; j++) { */
  /*     cout << C_host[i + j * m] << " "; */
  /*   } */
  /*   cout << endl; */
  /* } */

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
