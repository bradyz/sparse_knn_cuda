#ifndef SPGSKNN
#define SPGSKNN

/** 
 *  STL competable interface
 */
template<typename T>
void spgsknn
(
  unsigned int d, unsigned int m, unsigned int n, unsigned int k
  std::vector<int> &Q_row, 
  std::vector<int> &Q_col, 
  std::vector<T>   &Q_val,
  std::vector<int>  Q_map,
  std::vector<int> &R_row, 
  std::vector<int> &R_col, 
  std::vector<T>   &R_val, 
  std::vector<int>  R_map,
  std::vector<T>   &dis, 
  std::vector<int> &idx
)
{
  /** Q' = Q( :, Q_map ), R' = R( :, R_map ) */
};


template<typename T>
void spgsknn
(
  unsigned int d, unsigned int m, unsigned int n, unsigned int k
  thrust::host_vector<int> &Q_row, 
  thrust::host_vector<int> &Q_col, 
  thrust::host_vector<T>   &Q_val, 
  thrust::host_vector<int>  Q_map,
  thrust::host_vector<int> &R_row, 
  thrust::host_vector<int> &R_col, 
  thrust::host_vector<T>   &R_val, 
  thrust::host_vector<int>  R_map,
  thrust::host_vector<T>   &dis, 
  thrust::host_vector<int> &idx
)
{
  /** Q' = Q( :, Q_map ), R' = R( :, R_map ) */
};


template<typename T>
void spgsknn
(
  unsigned int d, unsigned int m, unsigned int n, unsigned int k
  thrust::device_vector<int> &Q_row, 
  thrust::device_vector<int> &Q_col, 
  thrust::device_vector<T>   &Q_val, 
  thrust::device_vector<int>  Q_map,
  thrust::device_vector<int> &R_row, 
  thrust::device_vector<int> &R_col, 
  thrust::device_vector<T>   &R_val, 
  thrust::device_vector<int>  R_map,
  thrust::device_vector<T>   &dis, 
  thrust::device_vector<int> &idx
)
{
  /** Q' = Q( :, Q_map ), R' = R( :, R_map ) */
};



#endif /** define SPGSKNN */
