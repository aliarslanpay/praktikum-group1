#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

int main(int argc, char const *argv[])
{
    int myid, commSize, ierr, lnbr, rnbr, sum;
    int *values;
    MPI_Status status;

    // Initialization of MPI communications
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        printf("MPI is not initialized, return code %d\n", ierr);
        return -1;
    }
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Distributing data over ranks
    int arraySize = atoi(argv[1]);
    int arraySizePerRank = ceil(arraySize / commSize);

    sum = 0;
    
    // Data initialization on local memory
    if (myid != (commSize - 1)) {
        values = (int *) malloc(arraySizePerRank * sizeof(int));
        for (int i = 0; i < arraySizePerRank; i++) {
            values[i] = (arraySizePerRank * myid) + i;
        }
    }
    else {
        int arraySizeForLastRank = arraySize - (arraySizePerRank * myid);
        values = (int *) malloc(arraySizeForLastRank * sizeof(int));
        for (int i = 0; i < arraySizeForLastRank; i++) {
            values[i] = (arraySizePerRank * myid) + i;
        }
    }

    // Summing up the data of the local array
    if (myid != (commSize - 1)) {
        for (int i = 0; i < arraySizePerRank; i++) {
            sum += values[i];
        }
    }
    else {
        int arraySizeForLastRank = arraySize - (arraySizePerRank * myid);
        for (int i = 0; i < arraySizeForLastRank; i++) {
            sum += values[i];
        }
    }


    // Building tree structure according to the rank and the MPI world size
    int parent = ((myid + 1) / 2) - 1;
    int isRoot = (parent == -1);
    int isLeaf = ((myid + 1) * 2) > commSize;
    int numOfChildNodes = !isLeaf ? (((myid + 1) * 2) > (commSize - 1) ? 1 : 2) : 0;
    int child1 = ((myid + 1) * 2) - 1;
    int child2 = ((myid + 1) * 2);
    int resultBufferFromChild[2];


    if (isLeaf) {
        // If it is a leaf node, just pass the sum to its parent node
        MPI_Send(&sum, 1, MPI_INTEGER, parent, 0, MPI_COMM_WORLD);
    }
    else if (!isRoot) {
        // Get the results from the children and add them to our own sum
        MPI_Recv(&(resultBufferFromChild[0]), 1, MPI_INTEGER, child1, 0, MPI_COMM_WORLD, &status);
        if (numOfChildNodes == 2) MPI_Recv(&(resultBufferFromChild[1]), 1, MPI_INTEGER, child2, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < numOfChildNodes; i++) {
            sum += resultBufferFromChild[i];
        }
        // Send the resulting sum to the parent node
        MPI_Send(&sum, 1, MPI_INTEGER, parent, 0, MPI_COMM_WORLD);
    }
    else {
        // Root node (0)
        // Gets the result from its children
        MPI_Recv(&(resultBufferFromChild[0]), 1, MPI_INTEGER, child1, 0, MPI_COMM_WORLD, &status);
        if (numOfChildNodes == 2) MPI_Recv(&(resultBufferFromChild[1]), 1, MPI_INTEGER, child2, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < numOfChildNodes; i++) {
            sum += resultBufferFromChild[i];
        }
    }

    // Print the result on the rank 0
    if (myid == 0) printf("%d\n", sum);

    return 0;
}
