#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "dense_matrix.h"

#include <iostream>
#include <tuple>
#include <vector>

// Uses compressed sparse column (CSC) to represent a matrix of size (m, n).
template <class T>
class SparseMatrix {

private:
  unsigned int m;
  unsigned int n;

  std::vector<unsigned int> colptr;
  std::vector<unsigned int> rowid;
  std::vector<T> value;

public:
  SparseMatrix<T>(unsigned int,
                  unsigned int,
                  std::vector<std::tuple<unsigned int, unsigned int, T>>&);

  DenseMatrix<T> mat_mul(const SparseMatrix<T>&) const;

  // Still not sure why a new template has to be used.
  template <class S>
  friend std::ostream& operator<<(std::ostream&, const SparseMatrix<S>&);

};

#endif
