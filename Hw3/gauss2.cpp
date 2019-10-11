#include "mpi.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
using namespace std;
const int N = 15;//number of variable
int myid, numproc;
void copyMemory(const double *M, double *A) {//复制矩阵使得同个进程中需要的内存是连续的
	for (int i = 0; i < N*(N + 1); i++)
		A[((i / (N + 1)) % numproc)*( (N * (N + 1)) / numproc) + (N+1) * ((i / (N + 1)) / numproc) + i % (N + 1)] = M[i];
}
int main(int argc,char* argv[]) {
	MPI_Init(&argc,&argv);
	double t1, t2;
	t1 = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&numproc);
	double *M, *c, *M_trans,*A,*tempRow;
	int *map = new int[N]; int *rmap = new int[N];
	int masterpro=0;
	for (int i = 0; i < N; i++) {
		map[i] = -1; rmap[i] = -1;
	}
	if (myid == 0) {//内存分配
		M = new double[N*(N+1)];
		M_trans = new double[N*(N + 1)];
		printf("Augmented matrix:");
		int num = 1;
		for (int i = 0; i < N*(N + 1); i++) {//make M--augmented matrix
			M[i] = rand() % 10;
			if ((i+1)%(N+1) == 0)
				M[i] += (rand()%100);
			if (i%(N+1) == 0)
				printf("\n");
			printf("%d ",(int)M[i]);
		}
		
		copyMemory(M, M_trans);
		A = new double[N*(N + 1) / numproc + 1];
		c = new double[N];
	}
	else {
		M = new double[1];
		M_trans = new double[1];
		A = new double[N*(N + 1) /numproc+1];
		c = new double[N];
	}
	tempRow = new double[N + 1];

	MPI_Scatter(M_trans, N*(N+1) / numproc,MPI_DOUBLE,A,N*(N+1)/numproc,MPI_DOUBLE,0,MPI_COMM_WORLD);//分发矩阵
	delete M_trans;
	double local_max_value, global_max_value;
	int local_max_id, max_proc_id, max_pro_id_1, global_max_id;
	for (int row = 0; row < N-1; row++) {
		local_max_value = -1e10; local_max_id = -1;  max_pro_id_1 = -1, global_max_id=-1;
		for (int i = 0; i < N / numproc; i++) {
			if (rmap[i * numproc + myid] < 0) {
				if (A[i*(N + 1) + row] > local_max_value) {
					local_max_value = A[i*(N + 1) + row];
					local_max_id = i;
				}
			}
		}//找到对应进程的主元
		MPI_Allreduce(&local_max_value, &global_max_value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);//将每个进程的主元找到一个最大值
		if (global_max_value > local_max_value-0.00001&&global_max_value < local_max_value+0.00001) {//对比最大值找到主元
			max_pro_id_1 = myid;
			for (int i = 0; i < N + 1; i++) {
				tempRow[i] = A[local_max_id*(N + 1) + i];
			}
			global_max_id= local_max_id * numproc + myid;			
		}
		MPI_Allreduce(&max_pro_id_1,&max_proc_id,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);//广播主元所在的进程便于下面的操作
		MPI_Bcast(&max_proc_id,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(tempRow,N+1,MPI_DOUBLE,max_proc_id,MPI_COMM_WORLD);//广播主元行
		MPI_Bcast(&global_max_id,1,MPI_INT,max_proc_id,MPI_COMM_WORLD);//广播主元id
		
		map[row] = global_max_id; rmap[global_max_id] = row;
		for (int i = 0; i < N/numproc; i++) {//求解过程
			if (rmap[i * numproc + myid] < 0) {
				double temp = A[i*(N + 1)+row] / tempRow[row];
				for (int j = row; j < N + 1; j++) {
					A[i*(N + 1) + j] -= tempRow[j] * temp;
				}
			}
		}
	}
	
	MPI_Gather(A,N*(N+1)/numproc,MPI_DOUBLE,M,N*(N+1)/numproc,MPI_DOUBLE,0,MPI_COMM_WORLD);//求解完三角阵后规约起来
	if (myid == 0) {
		for (int i = 0; i < N; i++) {
			if (rmap[i] == -1) {
				map[N - 1] = i;
			}
		}
		printf("\nSolutions X[%d]:\n",N);
	}
	if (myid == 0) {//输出
		for (int i = N-1; i >=0; i--) {
			int index = map[i] % numproc*(N / numproc) + map[i] / numproc;
			for (int j = N-1; j >i; j--) {
				M[index*(N+1)+N] -= c[j] * M[index*(N + 1) + N - j];
			}
			c[i] = M[index*(N + 1) + N] / M[index*(N+1)+i];
		}
		for (int i = 0; i < N; i++) {
			printf("X[%d]=%.2f\n",i, c[i]);
		}
	}
	t2 = MPI_Wtime();
	if (myid==masterpro)
		printf("total time:  %.6lf s\n", t2-t1);
	MPI_Finalize();
	return 0;
}