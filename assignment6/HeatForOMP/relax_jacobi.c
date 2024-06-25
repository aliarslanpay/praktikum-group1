
/*
 * relax_jacobi.c
 *
 * Jacobi Relaxation
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void exchange_boundaries(double **u, int local_nx, int local_ny, MPI_Comm cart_comm, int rank, int north, int south, int east, int west);
double compute_global_residual(double local_sum, MPI_Comm cart_comm);

double** allocate_2d_array(int rows, int cols) {
    double** array = (double**)malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        array[i] = (double*)malloc(cols * sizeof(double));
    }
    return array;
}

void free_2d_array(double** array, int rows) {
    for(int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

int mpi_test(int argc, char *argv[]) {
    int rank, size, dims[2], periods[2] = {0, 0}, reorder = 1;
    MPI_Comm cart_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <dim_x> <dim_y>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    dims[0] = atoi(argv[1]);
    dims[1] = atoi(argv[2]);

    if (dims[0] * dims[1] != size) {
        if (rank == 0) {
            fprintf(stderr, "Error: Product of dimensions must equal number of processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    // Domain decomposition
    int global_nx = 2000, global_ny = 2000; // Example problem size
    int local_nx = global_nx / dims[0];
    int local_ny = global_ny / dims[1];

    // Allocate local arrays for each process
    double **u = allocate_2d_array(local_nx + 2, local_ny + 2);
    double **utmp = allocate_2d_array(local_nx + 2, local_ny + 2);

    // Determine neighboring processes
    int coords[2], north, south, east, west;
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    // Main loop
    double local_residual, global_residual;
    int max_iter = 1000;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Exchange boundaries
        exchange_boundaries(u, local_nx, local_ny, cart_comm, rank, north, south, east, west);

        // Perform local Jacobi relaxation
        local_residual = relax_jacobi(u, utmp, local_nx + 2, local_ny + 2);

        // Compute global residual
        global_residual = compute_global_residual(local_residual, cart_comm);

        // Check for convergence (example threshold)
        if (global_residual < 1e-5) {
            break;
        }
    }

    // Clean up
    free_2d_array(u, local_nx + 2);
    free_2d_array(utmp, local_nx + 2);

    MPI_Finalize();
    return 0;
}

void exchange_boundaries(double **u, int local_nx, int local_ny, MPI_Comm cart_comm, int rank, int north, int south, int east, int west) {
    MPI_Request requests[8];

    // Send to north, receive from south
    MPI_Isend(u[1], local_nx, MPI_DOUBLE, north, 0, cart_comm, &requests[0]);
    MPI_Irecv(u[local_ny+1], local_nx, MPI_DOUBLE, south, 0, cart_comm, &requests[1]);

    // Send to south, receive from north
    MPI_Isend(u[local_ny], local_nx, MPI_DOUBLE, south, 0, cart_comm, &requests[2]);
    MPI_Irecv(u[0], local_nx, MPI_DOUBLE, north, 0, cart_comm, &requests[3]);

    // Send to east, receive from west
    for (int i = 1; i <= local_ny; i++) {
        MPI_Isend(&u[i][local_nx], 1, MPI_DOUBLE, east, 0, cart_comm, &requests[4 + 2 * (i - 1)]);
        MPI_Irecv(&u[i][0], 1, MPI_DOUBLE, west, 0, cart_comm, &requests[5 + 2 * (i - 1)]);
    }

    // Send to west, receive from east
    for (int i = 1; i <= local_ny; i++) {
        MPI_Isend(&u[i][1], 1, MPI_DOUBLE, west, 0, cart_comm, &requests[4 + 2 * local_ny + 2 * (i - 1)]);
        MPI_Irecv(&u[i][local_nx+1], 1, MPI_DOUBLE, east, 0, cart_comm, &requests[5 + 2 * local_ny + 2 * (i - 1)]);
    }

    MPI_Waitall(8 + 4 * local_ny, requests, MPI_STATUSES_IGNORE);
}

double compute_global_residual(double local_sum, MPI_Comm cart_comm) {
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    return global_sum;
}


double relax_jacobi( double **u1, double **utmp1,
         unsigned sizex, unsigned sizey )
{
  int i, j;
  double *help,*u, *utmp,factor=0.5;

  utmp=*utmp1;
  u=*u1;
  double unew, diff, sum=0.0;


  for( i=1; i<sizey-1; i++ ) {
  	int ii=i*sizex;
  	int iim1=(i-1)*sizex;
  	int iip1=(i+1)*sizex;
#pragma ivdep
    for( j=1; j<sizex-1; j++ ){
       unew = 0.25 * (u[ ii+(j-1) ]+
        		            u[ ii+(j+1) ]+
        		            u[ iim1+j ]+
        		            u[ iip1+j ]);
		    diff = unew - u[ii + j];
		    utmp[ii+j] = unew;
		    sum += diff * diff;

       }
    }

  *u1=utmp;
  *utmp1=u;
  return(sum);
}



