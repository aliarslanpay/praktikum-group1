/*
 * heat.h
 *
 * Global definitions for the iterative solver
 */

#ifndef JACOBI_H_INCLUDED
#define JACOBI_H_INCLUDED

#include <stdio.h>
#include <mpi.h>
// configuration

typedef struct
{
    float posx;
    float posy;
    float range;
    float temp;
}
heatsrc_t;

typedef struct
{
    unsigned maxiter;       // maximum number of iterations
    unsigned act_res;
    unsigned max_res;       // spatial resolution
    unsigned initial_res;
    unsigned res_step_size;
    unsigned visres;        // visualization resolution

    double *u, *uhelp;
    double *uvis;

    unsigned   numsrcs;     // number of heat sources
    heatsrc_t *heatsrcs;

    int communication;

    int number_of_rows;
    int number_of_columns;
    int row_start_index;
    int row_end_index;
    int column_start_index;
    int column_end_index;
    int extended_rows;
    int extended_columns;
}
algoparam_t;

enum communication {blocking, non_blocking};
enum directions {north, south, west, east};

// function declarations

// misc.c
int initialize( algoparam_t *param );
int finalize( algoparam_t *param );
void write_image( FILE * f, double *u,
		  unsigned sizex, unsigned sizey );
int coarsen(double *uold, unsigned oldx, unsigned oldy ,
        int column_start_index, int row_start_index, int columns, int rows,
	    double *unew, unsigned newx, unsigned newy );

// Gauss-Seidel: relax_gauss.c
double residual_gauss( double *u, double *utmp,
		       unsigned sizex, unsigned sizey );
void relax_gauss( double *u,
		  unsigned sizex, unsigned sizey  );

// Jacobi: relax_jacobi.c
double residual_jacobi( double *u,
			unsigned sizex, unsigned sizey );
double relax_jacobi_blocking( double **u1, double **utmp1,
         unsigned sizex, unsigned sizey,
         MPI_Comm * comm, int * neighborsRanks,
         double** sendWestColumn1, double** recvWestColumn1,
         double** sendEastColumn1, double** recvEastColumn1 );
double relax_jacobi_nonblocking( double **u1, double **utmp1,
         unsigned sizex, unsigned sizey,
         MPI_Comm * comm, int * neighborsRanks,
          double** sendWestColumn1, double** recvWestColumn1,
          double** sendEastColumn1, double** recvEastColumn1);

static inline const char *string_from_communication(enum communication comm){
    static const char *strings[] = { "blocking", "non-blocking" };

    return strings[comm];
}

#endif // JACOBI_H_INCLUDED
