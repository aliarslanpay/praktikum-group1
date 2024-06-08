#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_TRIALS 5000


void ping_pong(int rank, int size, int message_size) {
    char *message = (char*)malloc(message_size * sizeof(char));
    MPI_Status status;
    double start_time, end_time, total_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting

    for (int i = 0; i < NUM_TRIALS; ++i) {
        if (rank == 0) {
            start_time = MPI_Wtime();
            MPI_Send(message, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(message, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &status);
            end_time = MPI_Wtime();
            total_time += (end_time - start_time);
        } else if (rank == 1) {
            MPI_Recv(message, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Send(message, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        double avg_time = total_time / NUM_TRIALS;
        double bandwidth = (message_size * 2.0 * NUM_TRIALS) / (total_time * 1024 * 1024); // MB/s
        printf("Message size: %d bytes, Avg time: %f s, Bandwidth: %f MB/s\n", message_size, avg_time, bandwidth);
    }

    free(message);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly 2 MPI processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize processes at the beginning

    for (int i = 0; i <= 24; ++i) {
        int message_size = pow(2, i);
        ping_pong(rank, size, message_size);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
