#ifndef KNN_KERNEL_H
#define KNN_KERNEL_H

#include <vector>

template <class T>
void knn(std::vector<int>&, std::vector<int>&, std::vector<T>&,
         std::vector<int>&, std::vector<int>&, std::vector<T>&,
         unsigned int d, unsigned int m, unsigned int n,
         unsigned int k);

#endif
