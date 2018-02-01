dense_matrix.o: dense_matrix.cpp
	g++ -std=c++11 -c dense_matrix.cpp

sparse_matrix.o: sparse_matrix.cpp
	g++ -std=c++11 -c sparse_matrix.cpp

main.o: main.cpp
	g++ -std=c++11 -Wall -c main.cpp

test.o: test.cpp
	g++ -std=c++11 -Wall -c test.cpp

run: dense_matrix.o sparse_matrix.o main.o
	g++ -std=c++11 -Wall main.o dense_matrix.o sparse_matrix.o -o main.out

test: dense_matrix.o sparse_matrix.o test.o
	g++ -std=c++11 -Wall test.o dense_matrix.o sparse_matrix.o -o test.out

clean:
	rm -f *.o
	rm -f *.out
	rm -f *.gch
