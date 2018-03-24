#ifndef SPGSKNN
#define SPGSKNN

#include <vector>

/**
 *  STL compatible interface
 */
template<typename T>
void spgsknn
(
  unsigned int d, unsigned int m, unsigned int n, unsigned int k,
  std::vector<int> &Q_row,
  std::vector<int> &Q_col,
  std::vector<T>   &Q_val,
  std::vector<int> &R_row,
  std::vector<int> &R_col,
  std::vector<T>   &R_val,
  std::vector<T>   &distances,
  std::vector<int> &indices
);

/** Q' = Q( :, Q_map ), R' = R( :, R_map ) */
// template<typename T>
// void spgsknn
// (
//   unsigned int d, unsigned int m, unsigned int n, unsigned int k,
//   std::vector<int> &Q_row,
//   std::vector<int> &Q_col,
//   std::vector<T>   &Q_val,
//   std::vector<int>  Q_map,
//   std::vector<int> &R_row,
//   std::vector<int> &R_col,
//   std::vector<T>   &R_val,
//   std::vector<int>  R_map,
//   std::vector<T>   &distances,
//   std::vector<int> &indices
// );

#endif /** define SPGSKNN */
