#include "sparse_matrix.h"
#include "dense_matrix.h"

#include <algorithm>
#include <iostream>
#include <cassert>

using namespace std;

template <class T>
using row_col_val = tuple<unsigned int, unsigned int, T>;

template <class T>
void SparseMatrix<T>::init(unsigned int m,
                           unsigned int n,
                           vector<row_col_val<T>> &trips) {
  this->m = m;
  this->n = n;

  // Sort by column, then row.
  sort(trips.begin(), trips.end(),
      [](const row_col_val<T> &u, const row_col_val<T> &v) {
        if (get<1>(u) == get<1>(v))
          return get<0>(u) < get<0>(v);

        return get<1>(u) < get<1>(v);
      });

  // Populate pointers.
  int i = 0;

  for (int current_col = 0; current_col < n; current_col++) {
    this->colptr.push_back(this->rowid.size());

    // Get all the values in this column.
    while (i < trips.size() && get<1>(trips[i]) == current_col) {
      this->rowid.push_back(get<0>(trips[i]));
      this->value.push_back(get<2>(trips[i]));

      // Move along the values.
      ++i;
    }
  }

  // End spot.
  this->colptr.push_back(this->rowid.size());
}

template <class T>
SparseMatrix<T>::SparseMatrix(unsigned int m,
                              unsigned int n,
                              vector<row_col_val<T>> &trips) {
  this->init(m, n, trips);
}

template <class T>
SparseMatrix<T>::SparseMatrix(const DenseMatrix<T> &dense) {
  vector<row_col_val<T>> trips;

  for (int i = 0; i < dense.m; i++) {
    for (int j = 0; j < dense.n; j++) {
      if (dense(i, j) != 0.0)
        trips.push_back(row_col_val<T>(i, j, dense(i, j)));
    }
  }

  this->init(dense.m, dense.n, trips);
}

template <class T>
DenseMatrix<T> SparseMatrix<T>::mat_mul(const SparseMatrix<T> &that) const {
  assert(this->n == that.m);

  DenseMatrix<T> result(this->m, that.n);

  // (m x k) (k x n)
  for (int j = 0; j < that.n; j++) {
    for (int k_idx = that.colptr[j]; k_idx < that.colptr[j+1]; k_idx++) {
      int k = that.rowid[k_idx];
      T b_kj = that.value[k_idx];

      for (int i_idx = this->colptr[k]; i_idx < this->colptr[k+1]; i_idx++) {
        int i = this->rowid[i_idx];
        T a_ik = this->value[i_idx];

        result.assign(i, j, result(i, j) + a_ik * b_kj);
      }
    }
  }

  return result;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::operator-() {
  SparseMatrix<T> result(*this);

  for (int i = 0; i < result.value.size(); i++)
    result.value[i] = -result.value[i];

  return result;
}

template <class T>
ostream& operator<<(ostream &os, const SparseMatrix<T> &mat) {
  os << "Colptr:" << endl;
  for (unsigned int it : mat.colptr) os << it << " ";
  os << endl;

  os << "Rowid:" << endl;
  for (unsigned int it : mat.rowid) os << it << " ";
  os << endl;

  os << "Values:" << endl;
  for (T it : mat.value) os << it << " ";

  return os;
}

// Possible instantiations.
template class SparseMatrix<float>;
template class SparseMatrix<double>;

template ostream &operator<<(ostream&, const SparseMatrix<float>&);
template ostream &operator<<(ostream&, const SparseMatrix<double>&);
