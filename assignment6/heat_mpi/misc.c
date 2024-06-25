/*
 * misc.c
 *
 * Helper functions for
 * - initialization
 * - finalization,
 * - writing out a picture
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#include "heat.h"

/*
 * Initialize the iterative solver
 * - allocate memory for matrices
 * - set boundary conditions according to configuration
 */
int initialize( algoparam_t *param )
{
    int i, j;
    double dist;

    // total number of points (including border)
    const int np = param->act_res + 2;

    //
    // allocate memory
    //
    (param->u)     = (double*)malloc( sizeof(double)* (param->extended_rows)*(param->extended_columns) );
    (param->uhelp) = (double*)malloc( sizeof(double)* (param->extended_rows)*(param->extended_columns) );
    (param->uvis)  = (double*)calloc( sizeof(double),
				      (param->visres+2) *
				      (param->visres+2) );
					  
    for (i=0;i<param->extended_rows;i++){
    	for (j=0;j<param->extended_columns;j++){
    		param->u[i*param->extended_columns+j]=0.0;
			param->uhelp[i*param->extended_columns+j]=0.0;
    	}
    }

    if( !(param->u) || !(param->uhelp) || !(param->uvis) )
    {
	fprintf(stderr, "Error: Cannot allocate memory\n");
	return 0;
    }

    for( i=0; i<param->numsrcs; i++ )
    {
	/* top row */
	if(param->row_start_index==1){
		for( j=0; j<param->number_of_columns; j++ )
		{
			dist = sqrt(pow((double)(param->column_start_index+j)/(double)(np-1) - param->heatsrcs[i].posx, 2)+pow(param->heatsrcs[i].posy, 2));
			if( dist <= param->heatsrcs[i].range )
			{
			(param->u)[j+1] +=
				(param->heatsrcs[i].range-dist) /
				param->heatsrcs[i].range *
				param->heatsrcs[i].temp;
			(param->uhelp)[j+1]=(param->u)[j+1];
			}		
		}
		// west-north corner
		// Although the corner points can be processed together in the previous loop, 
		// we do not want to redundantly store values that we do not need on certain points in the ghost area, 
		// even though it does not affect the computation results.
		if(param->column_start_index==1){
			dist = sqrt(pow(param->heatsrcs[i].posx, 2)+pow(param->heatsrcs[i].posy, 2));
			if( dist <= param->heatsrcs[i].range )
			{
			(param->u)[0] +=
				(param->heatsrcs[i].range-dist) /
				param->heatsrcs[i].range *
				param->heatsrcs[i].temp;
			(param->uhelp)[0]=(param->u)[0];
			}	
		}
		// east-north corner
		if(param->column_end_index==(np-2)){
			dist = sqrt(pow(1.0-param->heatsrcs[i].posx, 2)+pow(param->heatsrcs[i].posy, 2));
			if( dist <= param->heatsrcs[i].range )
			{
			(param->u)[param->number_of_columns+1] +=
				(param->heatsrcs[i].range-dist) /
				param->heatsrcs[i].range *
				param->heatsrcs[i].temp;
			(param->uhelp)[param->number_of_columns+1]=(param->u)[param->number_of_columns+1];
			}			
		}
	}

	/* bottom row */
	if(param->row_end_index==(np-2)){
		for( j=0; j<param->number_of_columns; j++ )
		{
			dist = sqrt(pow((double)(param->column_start_index+j)/(double)(np-1) - param->heatsrcs[i].posx, 2)+pow(1.0-param->heatsrcs[i].posy, 2));
			if( dist <= param->heatsrcs[i].range )
			{
			(param->u)[param->extended_columns*(param->number_of_rows+1)+j+1] +=
				(param->heatsrcs[i].range-dist) /
				param->heatsrcs[i].range *
				param->heatsrcs[i].temp;
			(param->uhelp)[param->extended_columns*(param->number_of_rows+1)+j+1]=(param->u)[param->extended_columns*(param->number_of_rows+1)+j+1];
			}		
		}
		// west-south corner
		if(param->column_start_index==1){
			dist = sqrt(pow(param->heatsrcs[i].posx, 2)+pow(1.0-param->heatsrcs[i].posy, 2));
			if( dist <= param->heatsrcs[i].range )
			{
			(param->u)[param->extended_columns*(param->number_of_rows+1)]+=
				(param->heatsrcs[i].range-dist) /
				param->heatsrcs[i].range *
				param->heatsrcs[i].temp;
			(param->uhelp)[param->extended_columns*(param->number_of_rows+1)]=(param->u)[param->extended_columns*(param->number_of_rows+1)];
			}	
		}
		// east-south corner
		if(param->column_end_index==(np-2)){
			dist = sqrt(pow(1.0-param->heatsrcs[i].posx, 2)+pow(1.0-param->heatsrcs[i].posy, 2));
			if( dist <= param->heatsrcs[i].range )
			{
			(param->u)[param->extended_columns*(param->number_of_rows+2)-1] +=
				(param->heatsrcs[i].range-dist) /
				param->heatsrcs[i].range *
				param->heatsrcs[i].temp;
			(param->uhelp)[param->extended_columns*(param->number_of_rows+2)-1]=(param->u)[param->extended_columns*(param->number_of_rows+2)-1];
			}			
		}
	}	

	/* leftmost column */
	if(param->column_start_index==1){
		for( j=0; j<param->number_of_rows; j++ )
		{
			dist = sqrt( pow(param->heatsrcs[i].posx, 2)+
				pow((double)(param->row_start_index+j)/(double)(np-1) -
					param->heatsrcs[i].posy, 2));
			if( dist <= param->heatsrcs[i].range )
			{
			(param->u)[ (j+1)*param->extended_columns ]+=
				(param->heatsrcs[i].range-dist) /
				param->heatsrcs[i].range *
				param->heatsrcs[i].temp;
			(param->uhelp)[ (j+1)*param->extended_columns ] = (param->u)[ (j+1)*param->extended_columns ];
			}
		}
	}

	/* rightmost column */
	if(param->column_end_index==(np-2)){
		for( j=0; j<param->number_of_rows; j++ )
		{
			dist = sqrt( pow(1.0-param->heatsrcs[i].posx, 2)+
				pow((double)(param->row_start_index+j)/(double)(np-1) -
					param->heatsrcs[i].posy, 2));
			if( dist <= param->heatsrcs[i].range )
			{
			(param->u)[ (j+2)*param->extended_columns-1 ]+=
				(param->heatsrcs[i].range-dist) /
				param->heatsrcs[i].range *
				param->heatsrcs[i].temp;
			(param->uhelp)[ (j+2)*param->extended_columns-1 ] = (param->u)[ (j+2)*param->extended_columns-1 ];
			}
		}
	}

    }

    return 1;
}

/*
 * free used memory
 */
int finalize( algoparam_t *param )
{
    if( param->u ) {
	free(param->u);
	param->u = 0;
    }

    if( param->uhelp ) {
	free(param->uhelp);
	param->uhelp = 0;
    }

    if( param->uvis ) {
	free(param->uvis);
	param->uvis = 0;
    }

    return 1;
}


/*
 * write the given temperature u matrix to rgb values
 * and write the resulting image to file f
 */
void write_image( FILE * f, double *u,
		  unsigned sizex, unsigned sizey )
{
    // RGB table
    unsigned char r[1024], g[1024], b[1024];
    int i, j, k;

    double min, max;

    j=1023;

    // prepare RGB table
    for( i=0; i<256; i++ )
    {
	r[j]=255; g[j]=i; b[j]=0;
	j--;
    }
    for( i=0; i<256; i++ )
    {
	r[j]=255-i; g[j]=255; b[j]=0;
	j--;
    }
    for( i=0; i<256; i++ )
    {
	r[j]=0; g[j]=255; b[j]=i;
	j--;
    }
    for( i=0; i<256; i++ )
    {
	r[j]=0; g[j]=255-i; b[j]=255;
	j--;
    }

    min=DBL_MAX;
    max=-DBL_MAX;

    // find minimum and maximum
    for( i=0; i<sizey; i++ )
    {
	for( j=0; j<sizex; j++ )
	{
	    if( u[i*sizex+j]>max )
		max=u[i*sizex+j];
	    if( u[i*sizex+j]<min )
		min=u[i*sizex+j];
	}
    }


    fprintf(f, "P3\n");
    fprintf(f, "%u %u\n", sizex, sizey);
    fprintf(f, "%u\n", 255);

    for( i=0; i<sizey; i++ )
    {
	for( j=0; j<sizex; j++ )
	{
	    k=(int)(1024.0*(u[i*sizex+j]-min)/(max-min));
	    fprintf(f, "%d %d %d  ", r[k], g[k], b[k]);
	}
	fprintf(f, "\n");
    }
}

int coarsen( double *uold, unsigned oldx, unsigned oldy ,
		int column_start_index, int row_start_index, int extended_columns, int extended_rows,
	    double *unew, unsigned newx, unsigned newy )
{
    int i, j, k, l, ii, jj;

	int column_end_index = column_start_index + extended_columns - 3;
	int row_end_index = row_start_index + extended_rows - 3;
    int stopx = newx;
    int stopy = newy;
    float temp;
    float stepx = (float)oldx/(float)newx;
    float stepy = (float)oldy/(float)newy;

    if (oldx<newx){
	 stopx=oldx;
	 stepx=1.0;
    }
    if (oldy<newy){
     stopy=oldy;
     stepy=1.0;
    }

    //printf("oldx=%d, newx=%d\n",oldx,newx);
    //printf("oldy=%d, newy=%d\n",oldy,newy);
    //printf("rx=%f, ry=%f\n",stepx,stepy);
    // NOTE: this only takes the top-left corner,
    // and doesnt' do any real coarsening
	
    for( i=0; i<stopy; i++ ){
       ii=stepy*i;
       for( j=0; j<stopx; j++ ){
          jj=stepx*j;
          for ( k=0; k<stepy; k++ ){
	       	for ( l=0; l<stepx; l++ ){
	       		if (ii+k>=row_start_index && jj+l>=column_start_index
				&& ii+k<=row_end_index && jj+l<=column_end_index){
		           unew[i*newx+j] += uold[(ii+k+1-row_start_index)*extended_columns+(jj+l+1-column_start_index)] ;
				}
			}
	      }    
       }
    }

	if (row_start_index==1){
		for( j=0; j<stopx; j++ ){
			jj=stepx*j;
			for ( l=0; l<stepx; l++ ){
				if (jj+l>=column_start_index && jj+l<=column_end_index){
					unew[j] += uold[jj+l+1-column_start_index] ;
				}
			} 
		}
	}

	if (row_end_index==(oldy-2)){
		for( j=0; j<stopx; j++ ){
			jj=stepx*j;
			for ( l=0; l<stepx; l++ ){
				if (jj+l>=column_start_index && jj+l<=column_end_index){
					unew[(stopy-1)*newx+j] += uold[(oldy-row_start_index)*extended_columns+(jj+l+1-column_start_index)] ;
				}
			} 
		}
	}

	if(column_start_index==1){
		for( i=0; i<stopy; i++ ){
		ii=stepy*i;
			for ( k=0; k<stepy; k++ ){
				if (ii+k>=row_start_index && ii+k<=row_end_index){
					unew[i*newx] += uold[(ii+k+1-row_start_index)*extended_columns] ;
				}
			}
		}
	}

	if(column_end_index==(oldx-2)){
		for( i=0; i<stopy; i++ ){
		ii=stepy*i;
			for ( k=0; k<stepy; k++ ){
				if (ii+k>=row_start_index && ii+k<=row_end_index){
					unew[i*newx+(stopx-1)] += uold[(ii+k+1-row_start_index)*extended_columns+(oldx-column_start_index)] ;
				}
			}
		}	
	}
	
	if(row_start_index==1 && column_start_index==1){
		unew[0] += uold[0];
	}
   
	if(row_end_index==(oldy-2) && column_start_index==1){
		unew[(stopy-1)*newx] += uold[(extended_rows-1)*extended_columns];
	}

	if(row_start_index==1 && column_end_index==(oldx-2)){
		unew[stopx-1] += uold[extended_columns-1];
	}	    

	if(row_end_index==(oldy-2) && column_end_index==(oldx-2)){
		unew[(stopy-1)*newx+(stopx-1)] += uold[extended_rows*extended_columns-1];
	}

	float tmp = 1/(stepx*stepy);
	for (i = 0; i < newy; ++i){
		for (j = 0; j < newx; ++j){
			unew[i*newx+j] *= tmp;
		}
	}

  return 1;
}
