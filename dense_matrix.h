#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <iostream>

template <class T>
class DenseMatrix {

private:
  unsigned int m;
  unsigned int n;

  T** values;

public:
  DenseMatrix<T>(unsigned int, unsigned int);
  ~DenseMatrix<T>();

  // Get by row col.
  T operator()(unsigned int, unsigned int) const;

  void assign(unsigned int, unsigned int, T);

  template <class S>
  friend std::ostream& operator<<(std::ostream&, const DenseMatrix<S>&);

};

#endif
