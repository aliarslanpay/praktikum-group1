/*
 * relax_jacobi.c
 *
 * Jacobi Relaxation
 *
 */

#include "heat.h"


double relax_jacobi(double *u, double *utmp, unsigned sizex, unsigned sizey) {
    double unew, diff, sum = 0.0;
    double *u_2, *utmp_2;

    u_2 = u;
    utmp_2 = utmp;

    for (int i = 1; i < sizey - 1; ++i) {
        unsigned row_index = i * sizex;
        unsigned prev_row_index = (i - 1) * sizex;
        unsigned next_row_index = (i + 1) * sizex;

#pragma ivdep
        for (int j = 1; j < sizex - 1; ++j) {
            unew = 0.25 * (u_2[row_index + (j - 1)] +  // left
                           u_2[row_index + (j + 1)] +  // right
                           u_2[prev_row_index + j] +  // top
                           u_2[next_row_index + j]);  // bottom

            diff = unew - u_2[row_index + j];
            utmp_2[row_index + j] = unew;
            sum += diff * diff;

        }
    }

    utmp = u_2;
    u = utmp_2;

    return (sum);
}
