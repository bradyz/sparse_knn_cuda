#include "dense_matrix.h"

#include <iostream>

using namespace std;

template <class T>
DenseMatrix<T>::DenseMatrix(unsigned int m, unsigned int n) : m(m), n(n) {
  this->values = new T*[m];

  for (int i = 0; i < m; i++)
    this->values[i] = new T[n];

  // TODO(bradyz): find a better way to clear, maybe memset.
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      this->values[i][j] = 0.0;
}

template <class T>
DenseMatrix<T>::~DenseMatrix() {
  for (int i = 0; i < m; i++)
    delete[] this->values[i];

  delete[] this->values;
}

template <class T>
T DenseMatrix<T>::operator()(unsigned int m, unsigned int n) const {
  return this->values[m][n];
}

template <class T>
void DenseMatrix<T>::assign(unsigned int m, unsigned int n, T val) {
  this->values[m][n] = val;
}

template <class T>
ostream& operator<<(ostream &os, const DenseMatrix<T> &mat) {
  os << "m: " << mat.m << " n: " << mat.n << endl;

  for (int i = 0; i < mat.m; i++) {
    for (int j = 0; j < mat.n; j++)
      os << mat(i, j) << " ";
    os << endl;
  }

  return os;
}

// Possible instantiations.
template class DenseMatrix<float>;
template class DenseMatrix<double>;

template ostream &operator<<(ostream&, const DenseMatrix<float>&);
template ostream &operator<<(ostream&, const DenseMatrix<double>&);
