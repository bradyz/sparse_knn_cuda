NVCC=nvcc
NVCCFLAGS=-std=c++11 -O3 -Xcompiler -fopenmp

CXX=g++
CXXFLAGS=-std=c++11 -Wall -O3

knn_kernel_gpu.o: src/knn_kernel_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c src/knn_kernel_gpu.cu

main_gpu.o: src/main_gpu.cpp
	$(CXX) $(CXXFLAGS) -c src/main_gpu.cpp

test.o: test.cpp
	$(CXX) $(CXXFLAGS) -c test.cpp

run_gpu: knn_kernel_gpu.o main_gpu.o
	$(NVCC) $(NVCCFLAGS) -lcusparse -lgomp main_gpu.o knn_kernel_gpu.o -o main_gpu.out

clean:
	rm -f *.gch
	rm -f *.o
	rm -f *.out
