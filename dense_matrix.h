#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <iostream>

template <class T>
class DenseMatrix {

private:
  T** values;

public:
  unsigned int m;
  unsigned int n;

  DenseMatrix<T>(unsigned int, unsigned int);
  ~DenseMatrix<T>();

  // Get by row col.
  T operator()(unsigned int, unsigned int) const;

  void assign(unsigned int, unsigned int, T);
  DenseMatrix<T> transpose() const;

  template <class S>
  friend std::ostream& operator<<(std::ostream&, const DenseMatrix<S>&);

};

#endif
