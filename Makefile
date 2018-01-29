sparse_matrix.o: sparse_matrix.cpp
	g++ -std=c++11 -c sparse_matrix.cpp

main.o: main.cpp
	g++ -std=c++11 -Wall -c main.cpp

run: sparse_matrix.o main.o
	g++ -std=c++11 -Wall main.o sparse_matrix.o -o main.out

clean:
	rm -f *.o
	rm -f *.out
	rm -f *.gch
