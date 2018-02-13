NVCC=nvcc
NVCCFLAGS=-std=c++11 -O3 -Xcompiler -fopenmp

CXX=g++
CXXFLAGS=-std=c++11 -Wall -O3

dense_matrix.o: dense_matrix.cpp
	$(CXX) $(CXXFLAGS) -c dense_matrix.cpp

sparse_matrix.o: sparse_matrix.cpp
	$(CXX) $(CXXFLAGS) -c sparse_matrix.cpp

knn_kernel_cpu.o: knn_kernel_cpu.cpp
	$(CXX) $(CXXFLAGS) -c knn_kernel_cpu.cpp

knn_kernel_gpu.o: knn_kernel_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c knn_kernel_gpu.cu

main_cpu.o: main_cpu.cpp
	$(CXX) $(CXXFLAGS) -c main_cpu.cpp

main_gpu.o: main_gpu.cpp
	$(CXX) $(CXXFLAGS) -c main_gpu.cpp

test.o: test.cpp
	$(CXX) $(CXXFLAGS) -c test.cpp

run_cpu: dense_matrix.o sparse_matrix.o knn_kernel_cpu.o main_cpu.o
	$(CXX) $(CXXFLAGS) main_cpu.o dense_matrix.o sparse_matrix.o knn_kernel_cpu.o -o main_cpu.out

run_gpu: knn_kernel_gpu.o main_gpu.o
	$(NVCC) $(NVCCFLAGS) -lcusparse -lgomp main_gpu.o knn_kernel_gpu.o -o main_gpu.out

test: dense_matrix.o sparse_matrix.o test.o
	$(CXX) $(CXXFLAGS) test.o dense_matrix.o sparse_matrix.o -o test.out

clean:
	rm -f *.gch
	rm -f *.o
	rm -f *.out
