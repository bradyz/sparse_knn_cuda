#include "sparse_matrix.h"

#include <iostream>

using namespace std;

template <class T>
using row_col_val = tuple<unsigned int, unsigned int, T>;

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

template <class T>
SparseMatrix<T>::SparseMatrix(unsigned int m,
                              unsigned int n,
                              vector<row_col_val<T>> &trips) : m(m), n(n) {
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

// Possible instantiations.
template class SparseMatrix<float>;
template ostream &operator<<(ostream&, const SparseMatrix<float>&);
