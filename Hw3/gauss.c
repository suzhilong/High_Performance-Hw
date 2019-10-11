#include <math.h>
#include <time.h>
#include "stdio.h"
#include <stdlib.h>
#include "mpi.h"

const int rows = 4; /*the rows of matrix*/
const int cols = 4; /*the cols of matrix*/

int main(int argc, char **argv)
{
	int i,j,k,myid,numprocs,anstag;
	double A[rows][cols],B[cols],X[rows],AB[rows][cols+1];
	int masterpro;
	double buf[cols+1];
	double starttime,endtime;
	double tmp,totaltime;

	srand((unsigned int)time(NULL));

	MPI_Status status;

	masterpro = 0;

	MPI_Init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

	if (myid==0)
	{
		printf("please input coefficients:\n");
		for(i=0;i<cols;i++)
		{//get A
			for(j=0;j<rows;j++)
			{
				scanf("%lf",&A[i][j]);
			}
		}
		printf("please input constants:\n");
		for(i=0;i<rows;i++)
		{//get B
			scanf("%lf",&B[i]);
		}
		for(i=0;i<rows;i++)
		{//get augmented matrix AB
			AB[i][cols]=B[i];
			for(j=0;j<cols;j++)
				AB[i][j]=A[i][j];
		}
		printf("augmented matrix:\n");
		for(i=0;i<rows;i++)
		{//print augmented matrix AB
			for(j=0;j<cols+1;j++)
				printf("%.2f ",AB[i][j]);
			printf("\n");
		}	
	}

	int x;
	double coe;
	starttime = MPI_Wtime();
	for(x=0;x<rows;x++)
	{
		MPI_Bcast(&AB[x][0],cols+1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		if(myid==0)
		{
			for(i=1;i<numprocs;i++)
			{
				for(k=x+1+i;k<rows;k+=numprocs)
				MPI_Send(&AB[k][0],cols+1,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
			}
			for(k=x+1;k<rows;k+=numprocs)
			{
				coe=AB[k][x]/AB[x][x];
				for(j=x;j<cols+1;j++)
				{
					AB[k][j]-=AB[x][j]*coe;
				}
			}
			for(i=1;i<numprocs;i++)
			{
				for(k=x+1+i;k<rows;k+=numprocs)
				{
					MPI_Recv(&AB[k][0],cols+1,MPI_DOUBLE,i,1,MPI_COMM_WORLD,&status);
				}
			}
		}
		else
		{
			for(k=x+1+myid;k<rows;k+=numprocs)
			{
				MPI_Recv(&AB[k-1][0],cols+1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&status);
				coe=AB[k-1][x]/AB[x][x];
				for(j=x;j<cols+1;j++)
				{
					AB[k-1][j]-=AB[x][j]*coe;
				}
				MPI_Send(&AB[k-1][0],cols+1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
			}
		}
	}

	if(myid==0)
	{
		printf("upper triangle:\n");
		for(i=0;i<rows;i++)
		{//print AB after elimination
			for(j=0;j<cols+1;j++)
				printf("%.2f ",AB[i][j]);
			printf("\n");
		}
	}

	int result;
	if(myid==0)
	{
		if(fabs(AB[rows-1][cols-1])<0.0000001 && fabs(AB[rows-1][cols]<0.0000001))
		{
			printf("ifinite solution\n");
			result=1;
		}
		if(fabs(AB[rows-1][cols-1])<0.0000001 && fabs(AB[rows-1][cols]>0.0000001))  
		{
			printf("no solution\n");
			result=-1;
		}
		else
		{
			printf("unique solution\n");
			result=0;
		}
	}

	double temp=0.0;

	if(result==0)
	{
		if(myid==0)
		{
			X[rows-1]=AB[rows-1][cols]/AB[rows-1][cols-1];
			for(k=rows-2;k>=0;k--)
			{
				temp=0.0;
				for (j=k+1;j<cols;j++)
					temp=temp+AB[k][j]*X[j];
				X[k]=(AB[k][cols]-temp)/AB[k][k];
			}
			printf("X:\n");
			for(i=0;i<rows;i++)
				printf("X[%d]=%.2f\n",i,X[i]);
		}
	}

	if(result==1)
	{
		if(myid==0)
		{
			X[rows-1]=0;
			for(k=rows-2;k>=0;k--)
			{
				temp=0;
				for (j=k+1;j<cols;j++)
					temp=temp+AB[k][j]*X[j];
				X[k]=(AB[k][cols]-temp)/AB[k][k];
			}
			printf("x:\n");
			for(i=0;i<rows;i++)
				printf("X[%d]=%.2f\n",i,X[i]);
		}
	}

	endtime = MPI_Wtime();

	totaltime = endtime - starttime;

	if (myid == masterpro)
		printf("total time :%f s.\n",totaltime);

	MPI_Finalize();

	return 0;
}

