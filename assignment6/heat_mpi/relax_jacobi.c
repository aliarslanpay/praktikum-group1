/*
 * relax_jacobi.c
 *
 * Jacobi Relaxation
 *
 */

#include "heat.h"

double relax_jacobi_blocking(double **u1, double **utmp1, unsigned sizex, unsigned sizey,
                             MPI_Comm* comm_cart, int* neighbors_ranks,
                             double** send_west_buf, double** recv_west_buf, double** send_east_buf, double** recv_east_buf)
{
    int i, j;
    double *u = *u1, *utmp = *utmp1;
    double unew, diff, sum = 0.0;

    double *send_west = *send_west_buf;
    double *recv_west = *recv_west_buf;
    double *send_east = *send_east_buf;
    double *recv_east = *recv_east_buf;

    // Communication with south neighbor
    if (neighbors_ranks[south] != MPI_PROC_NULL) {
        MPI_Sendrecv(u + (sizey-2) * sizex + 1, sizex-2, MPI_DOUBLE, neighbors_ranks[south], 0,
                     u + (sizey-1) * sizex + 1, sizex-2, MPI_DOUBLE, neighbors_ranks[south], 1, *comm_cart, MPI_STATUS_IGNORE);
    }

    // Communication with north neighbor
    if (neighbors_ranks[north] != MPI_PROC_NULL) {
        MPI_Sendrecv(u + sizex + 1, sizex-2, MPI_DOUBLE, neighbors_ranks[north], 1,
                     u + 1, sizex-2, MPI_DOUBLE, neighbors_ranks[north], 0, *comm_cart, MPI_STATUS_IGNORE);
    }

    // Communication with east neighbor
    if (neighbors_ranks[east] != MPI_PROC_NULL) {
        for (i = 0; i < sizey - 2; ++i) {
            send_east[i] = u[(i + 2) * sizex - 2];
        }
        MPI_Sendrecv(send_east, sizey-2, MPI_DOUBLE, neighbors_ranks[east], 2,
                     recv_east, sizey-2, MPI_DOUBLE, neighbors_ranks[east], 3, *comm_cart, MPI_STATUS_IGNORE);
        for (i = 0; i < sizey - 2; ++i) {
            u[(i + 2) * sizex - 1] = recv_east[i];
        }
    }

    // Communication with west neighbor
    if (neighbors_ranks[west] != MPI_PROC_NULL) {
        for (i = 0; i < sizey - 2; ++i) {
            send_west[i] = u[(i + 1) * sizex + 1];
        }
        MPI_Sendrecv(send_west, sizey-2, MPI_DOUBLE, neighbors_ranks[west], 3,
                     recv_west, sizey-2, MPI_DOUBLE, neighbors_ranks[west], 2, *comm_cart, MPI_STATUS_IGNORE);
        for (i = 0; i < sizey - 2; ++i) {
            u[(i + 1) * sizex] = recv_west[i];
        }
    }

    // Jacobi iteration over the grid interior
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

    *u1 = utmp;
    *utmp1 = u;
    return sum;
}

double relax_jacobi_nonblocking(double **u1, double **utmp1, unsigned sizex, unsigned sizey,
                                MPI_Comm * comm_cart, int * neighbors_ranks,
                                double** send_west_buf, double** recv_west_buf, double** send_east_buf, double** recv_east_buf)
{

    int i, j;
    double *u = *u1, *utmp = *utmp1;
    double unew, diff, sum = 0.0;

    double *send_west = *send_west_buf;
    double *recv_west = *recv_west_buf;
    double *send_east = *send_east_buf;
    double *recv_east = *recv_east_buf;

    MPI_Request send_request[4]; // Array to store send requests
    MPI_Request recv_request[4]; // Array to store receive requests
    MPI_Status status[4];       // Array to store status of communication

    // Initiate non-blocking send/receive with south neighbor
    if (neighbors_ranks[south] != MPI_PROC_NULL){
        MPI_Isend(u + (sizey-2) * sizex + 1, (sizex - 2), MPI_DOUBLE, neighbors_ranks[south], 0, *comm_cart, &send_request[south]);
        MPI_Irecv(u + (sizey-1) * sizex + 1, (sizex - 2), MPI_DOUBLE, neighbors_ranks[south], 1, *comm_cart, &recv_request[south]);
    }

    // Initiate non-blocking send/receive with north neighbor
    if (neighbors_ranks[north] != MPI_PROC_NULL){
        MPI_Isend(u + sizex + 1, (sizex - 2), MPI_DOUBLE, neighbors_ranks[north], 1, *comm_cart, &send_request[north]);
        MPI_Irecv(u + 1, (sizex - 2), MPI_DOUBLE, neighbors_ranks[north], 0, *comm_cart, &recv_request[north]);
    }

    // Initiate non-blocking send/receive with east neighbor
    if (neighbors_ranks[east] != MPI_PROC_NULL){
        for (i = 0; i < sizey - 2; ++i){
            send_east[i] = u[(i+2)*sizex - 2];
        }
        MPI_Isend(send_east, (sizey-2), MPI_DOUBLE, neighbors_ranks[east], 2, *comm_cart, &send_request[east]);
        MPI_Irecv(recv_east, (sizey-2), MPI_DOUBLE, neighbors_ranks[east], 3, *comm_cart, &recv_request[east]);
    }

    // Initiate non-blocking send/receive with west neighbor
    if (neighbors_ranks[west] != MPI_PROC_NULL){
        for (i = 0; i < sizey - 2; ++i){
            send_west[i] = u[(i+1)*sizex +1];
        }
        MPI_Isend(send_west, (sizey-2), MPI_DOUBLE, neighbors_ranks[west], 3, *comm_cart, &send_request[west]);
        MPI_Irecv(recv_west, (sizey-2), MPI_DOUBLE, neighbors_ranks[west], 2, *comm_cart, &recv_request[west]);
    }

    // Perform Jacobi iteration over the interior points while communication is ongoing
    for( i=2; i<sizey-2; i++ ) {
        int ii=i*sizex;
        int iim1=(i-1)*sizex;
        int iip1=(i+1)*sizex;
#pragma ivdep
        for( j=2; j<sizex-2; j++ ){
            unew = 0.25 * (u[ ii+(j-1) ]+
                           u[ ii+(j+1) ]+
                           u[ iim1+j ]+
                           u[ iip1+j ]);
            diff = unew - u[ii + j];
            utmp[ii+j] = unew;
            sum += diff * diff;

        }
    }

    // Wait for non-blocking communication to complete
    if (neighbors_ranks[east] != MPI_PROC_NULL){
        MPI_Wait(&recv_request[east], &status[east]);
    }
    if (neighbors_ranks[west] != MPI_PROC_NULL){
        MPI_Wait(&recv_request[west], &status[west]);
    }
    if (neighbors_ranks[south] != MPI_PROC_NULL){
        MPI_Wait(&recv_request[south], &status[south]);
    }
    if (neighbors_ranks[north] != MPI_PROC_NULL){
        MPI_Wait(&recv_request[north], &status[north]);
    }

    // Update boundaries after communication
    if (neighbors_ranks[east] != MPI_PROC_NULL){
        for (i = 0; i < sizey - 2; ++i){
            u[(i+2)*sizex - 1] = recv_east[i];
        }
    }

    if (neighbors_ranks[west] != MPI_PROC_NULL){
        for (i = 0; i < sizey - 2; ++i){
            u[(i+1)*sizex] = recv_west[i];
        }
    }

    // Continue Jacobi iteration for boundary points after communication is complete
    // for east
#pragma ivdep
    for (i = 2; i < sizey - 2; ++i){
        int ii = i * sizex;
        int iim1 = (i - 1) * sizex;
        int iip1 = (i + 1) * sizex;
        j = sizex - 2;
        unew = 0.25 * (u[ii + (j - 1)] +
                       u[ii + (j + 1)] +
                       u[iim1 + j] +
                       u[iip1 + j]);
        diff = unew - u[ii + j];
        utmp[ii + j] = unew;
        sum += diff * diff;
    }

    // for west
#pragma ivdep
    for (i = 2; i < sizey - 2; ++i){
        int ii = i * sizex;
        int iim1 = (i - 1) * sizex;
        int iip1 = (i + 1) * sizex;

        unew = 0.25 * (u[ii] +
                       u[ii + 2] +
                       u[iim1+1] +
                       u[iip1+1]);
        diff = unew - u[ii+1];
        utmp[ii + 1] = unew;
        sum += diff * diff;
    }

    // for north
    i = 1;
    int ii=i*sizex;
    int iim1=(i-1)*sizex;
    int iip1=(i+1)*sizex;
#pragma ivdep
    for (j = 1; j < sizex - 1; ++j){
        unew = 0.25 * (u[ii + (j - 1)] +
                       u[ii + (j + 1)] +
                       u[iim1 + j] +
                       u[iip1 + j]);
        diff = unew - u[ii + j];
        utmp[ii + j] = unew;
        sum += diff * diff;
    }

    // for south
    i = sizey - 2;
    ii=i*sizex;
    iim1=(i-1)*sizex;
    iip1=(i+1)*sizex;
#pragma ivdep
    for (j = 1; j < sizex - 1; ++j){
        unew = 0.25 * (u[ii + (j - 1)] +
                       u[ii + (j + 1)] +
                       u[iim1 + j] +
                       u[iip1 + j]);
        diff = unew - u[ii + j];
        utmp[ii + j] = unew;
        sum += diff * diff;
    }

    *u1=utmp;
    *utmp1=u;
    return(sum);
}