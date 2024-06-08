#include <stdio.h>
#include <stdlib.h>

#include "input.h"
#include "heat.h"
#include "timing.h"
#include <mpi.h>

void usage(char *s) {
    fprintf(stderr, "Usage: %s <input file> <mpi topology grid rows> <mpi topology grid columns> [result file]\n\n", s);
}

int main(int argc, char *argv[]){
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }

    int i, j;
    FILE *infile, *resfile;
    char *resfilename;
    int iter;

    double* time;
    double localResidual = 0.0;
    double globalResidual = 0.0;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Error: The MPI library does not have the required thread support level\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get grid dimensions from command-line arguments
    int grid_rows = atoi(argv[2]);
    int grid_cols = atoi(argv[3]);
    int grid_dim[2] = {grid_rows, grid_cols};
    int period[2] = {0, 0};
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid_dim, period, 0, &comm_cart);

    //get cartesian cartesian_coords
    int cartesian_coords[2] = {0, 0};
    MPI_Cart_coords(comm_cart, rank, 2, cartesian_coords);

    // determining neighbors
    int neighbors_ranks[4];
    MPI_Cart_shift(comm_cart, 0, 1, &neighbors_ranks[north], &neighbors_ranks[south]);
    MPI_Cart_shift(comm_cart, 1, 1, &neighbors_ranks[west], &neighbors_ranks[east]);

    algoparam_t param;
    param.visres = 100;

    // check input file
    if (!(infile = fopen(argv[1], "r"))) {
        fprintf(stderr, "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
        usage(argv[0]);
        return 1;
    }

    // check result file
    resfilename = (argc >= 5) ? argv[4] : "heat.ppm";

    if (!(resfile = fopen(resfilename, "w"))) {
        fprintf(stderr, "\nError: Cannot open \"%s\" for writing.\n\n", resfilename);
        usage(argv[0]);
        return 1;
    }

    // check input
    if (!read_input(infile, &param)) {
        fprintf(stderr, "\nError: Error parsing input file.\n\n");
        usage(argv[0]);
        return 1;
    }

    // check communication (communication)
    if (param.communication < 0 || param.communication > 1) {
        fprintf(stderr, "\nError: Incorrect value for the communication.\n\n");
        usage(argv[0]);
        return 1;
    }

    if (rank == 0) {
        printf("Two-dimensional distribution: %d x %d \n", grid_rows, grid_cols);
        printf("Communication: %s \n\n", string_from_communication(param.communication));
        print_params(&param);
        time = (double *) calloc(sizeof(double), (int) (param.max_res - param.initial_res + param.res_step_size) / param.res_step_size);
    }

    int exp_number = 0;
    int remaining_rows;

    for (param.act_res = param.initial_res; param.act_res <= param.max_res; param.act_res = param.act_res + param.res_step_size) {
        // Horizontally divide the grid
        param.number_of_rows = param.act_res / grid_rows;
        remaining_rows = param.act_res % grid_rows;

        if(cartesian_coords[0] < remaining_rows) {
            param.number_of_rows++;
            param.row_start_index = cartesian_coords[0] * param.number_of_rows + 1;
        }
        else {
            param.row_start_index = cartesian_coords[0] * param.number_of_rows + 1 + remaining_rows;
        }
        param.row_end_index = param.row_start_index + param.number_of_rows - 1;

        // Vertically divide the grid
        param.number_of_columns = param.act_res / grid_cols;
        remaining_rows = param.act_res % grid_cols;
        if(cartesian_coords[1] < remaining_rows) {
            param.number_of_columns++;
            param.column_start_index = cartesian_coords[1] * param.number_of_columns + 1;
        }
        else {
            param.column_start_index = cartesian_coords[1] * param.number_of_columns + 1 + remaining_rows;
        }
        param.column_end_index = param.column_start_index + param.number_of_columns - 1;

        param.extended_rows = param.number_of_rows + 2;
        param.extended_columns = param.number_of_columns + 2;

        if (!initialize(&param)) {
            fprintf(stderr, "Error in Jacobi initialization.\n\n");
            usage(argv[0]);
        }

        // Allocate memory for communication buffers
        double* send_west_buffer = (double*)malloc( sizeof(double)* param.number_of_rows);
        double* recv_west_buffer = (double*)malloc( sizeof(double)* param.number_of_rows);
        double* send_east_buffer = (double*)malloc( sizeof(double)* param.number_of_rows);
        double* recv_east_buffer = (double*)malloc( sizeof(double)* param.number_of_rows);

        MPI_Barrier(comm_cart);

        if (rank==0)
            time[exp_number] = wtime();

        switch(param.communication){
            case blocking:
                for (iter = 0; iter < param.maxiter; iter++) {
                    localResidual = relax_jacobi_blocking(&(param.u), &(param.uhelp), param.extended_columns, param.extended_rows,
                                                          &comm_cart, neighbors_ranks, &send_west_buffer, &recv_west_buffer, &send_east_buffer, &recv_east_buffer);
                    MPI_Reduce(&localResidual, &globalResidual, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
                }
                break;
            case non_blocking:
                for (iter = 0; iter < param.maxiter; iter++) {
                    localResidual = relax_jacobi_nonblocking(&(param.u), &(param.uhelp), param.extended_columns, param.extended_rows,
                                                             &comm_cart, neighbors_ranks, &send_west_buffer, &recv_west_buffer, &send_east_buffer, &recv_east_buffer);
                    MPI_Reduce(&localResidual, &globalResidual, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
                }
                break;
        }

        if(rank == 0) {
            time[exp_number] = wtime() - time[exp_number];

            printf("\n\nResolution: %u\n", param.act_res);
            printf("===================\n");
            printf("Execution time: %f\n", time[exp_number]);
            printf("Residual: %f\n\n", globalResidual);

            printf("megaflops:  %.1lf\n", (double) param.maxiter * param.act_res * param.act_res * 7 / time[exp_number] / 1000000);
            printf("  flop instructions (M):  %.3lf\n", (double) param.maxiter * param.act_res * param.act_res * 7 / 1000000);
        }

        exp_number++;

        free(send_west_buffer);
        free(recv_west_buffer);
        free(send_east_buffer);
        free(recv_east_buffer);
    }

    param.act_res = param.act_res - param.res_step_size;

    coarsen(param.u, param.act_res + 2, param.act_res + 2, param.column_start_index, param.row_start_index, param.extended_columns, param.extended_rows, param.uvis, param.visres + 2, param.visres + 2);

    double *uvis_gl = NULL;
    if(rank == 0)
        uvis_gl = (double*)calloc( sizeof(double),
                                   (param.visres+2) *
                                   (param.visres+2) );

    MPI_Reduce(param.uvis, uvis_gl, (param.visres + 2)*(param.visres + 2), MPI_DOUBLE, MPI_SUM, 0, comm_cart);

    if(rank==0) {
        write_image(resfile, uvis_gl, param.visres + 2, param.visres + 2);
        free(uvis_gl);
    }

    finalize(&param);
    MPI_Finalize();

    return EXIT_SUCCESS;
}