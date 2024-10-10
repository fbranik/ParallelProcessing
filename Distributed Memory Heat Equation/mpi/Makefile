CC=gcc
CFLAGS=-O3 -lm -Wall
RES=-DPRINT_RESULTS
CONV=-DTEST_CONV

all: jacobi gaussSeidelSor redBlackSor

jacobi: Jacobi_mpi.c utils.c
	$(CC) $(CFLAGS) $(RES) $(CONV) Jacobi_mpi.c utils.c -o jacobi

gaussSeidelSor: GaussSeidelSOR_mpi.c utils.c
	$(CC) $(CFLAGS) $(RES) $(CONV) GaussSeidelSOR_mpi.c utils.c -o gaussSeidelSor

redBlackSor: RedBlackSOR_mpi.c utils.c 
	$(CC) $(CFLAGS) $(RES) $(CONV) RedBlackSOR_mpi.c utils.c -o redBlackSor

clean:
	rm jacobi gaussSeidelSor redBlackSor
