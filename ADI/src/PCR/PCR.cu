/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>
#include <cstdlib>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "PCR.h"
#include "../constants/alloc.h"
#include "../constants/cutil_math.h"

using namespace std;

#define CHUNK_MAX 256
#define CR_BUFF_MAX 128 // set statically since my current card doesn't support dynamic memory
						  // allocation
#undef TESTING

/* Constructor */
PCR::PCR(int N_tmp, int S_tmp) {
	if (N_tmp > CHUNK_MAX*CR_BUFF_MAX) {
		cout << "Error: system dimension exceeds allowance of " << CHUNK_MAX*CR_BUFF_MAX;
		cout << " equations!" << endl;
		exit(1);
	}

	N = N_tmp;
	S = S_tmp;
	check_return(cudaMalloc((TYPE_VAR**) &A1, S*N*sizeof(TYPE_VAR)));
	check_return(cudaMalloc((TYPE_VAR**) &A2, S*N*sizeof(TYPE_VAR)));
	check_return(cudaMalloc((TYPE_VAR**) &A3, S*N*sizeof(TYPE_VAR)));
	check_return(cudaMalloc((TYPE_VAR**) &b, S*N*sizeof(TYPE_VAR)));
}

/* Destructor */
PCR::~PCR() {
	check_return(cudaFree(A1));
	check_return(cudaFree(A2));
	check_return(cudaFree(A3));
	check_return(cudaFree(b));
}

/* Public Methods */

/* PCR solver method */
__host__ void PCR::PCR_solve(TYPE_VAR* A1_tmp, TYPE_VAR* A2_tmp, TYPE_VAR* A3_tmp,
	TYPE_VAR* b_tmp, TYPE_VAR* x_tmp) {

	PCR_init(A1_tmp, A2_tmp, A3_tmp, b_tmp);

	/* Launch solver here */

	PCR_solver<<<S, CHUNK_MAX>>>(A1, A2, A3, b, N);
	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	check_return(cudaMemcpy(x_tmp, b, S*N*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
}

/* PCR solver for pre-seeded A matrix and B vector */
__host__ void PCR::PCR_solve(TYPE_VAR* x_tmp) {
	PCR_solver<<<S, CHUNK_MAX>>>(A1, A2, A3, b, N);
	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	check_return(cudaMemcpy(x_tmp, b, S*N*sizeof(TYPE_VAR), cudaMemcpyDeviceToDevice));
}

/* Accessors for seeding A matrix and B vector */

__host__ TYPE_VAR* PCR::A1_arr() {
	return A1;
}

__host__ TYPE_VAR* PCR::A2_arr() {
	return A2;
}

__host__ TYPE_VAR* PCR::A3_arr() {
	return A3;
}

__host__ TYPE_VAR* PCR::B_arr() {
	return b;
}

/* Direction reverse for use with ADI solver */
__host__ void PCR::ADI_flip(int N_tmp, int S_tmp) {
	N = N_tmp;
	S = S_tmp;
}

/* Private Methods */

/* Allocates device memory */
__host__ void PCR::PCR_init(TYPE_VAR* A1_tmp, TYPE_VAR* A2_tmp, TYPE_VAR* A3_tmp,
		TYPE_VAR* b_tmp) {
	check_return(cudaMemcpy(A1, A1_tmp, S*N*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));
	check_return(cudaMemcpy(A2, A2_tmp, S*N*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));
	check_return(cudaMemcpy(A3, A3_tmp, S*N*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));
	check_return(cudaMemcpy(b, b_tmp, S*N*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));
}

/* Copies reduced matrix A' to host memory A for testing purposes */
__host__ void PCR::PCR_A_tester(TYPE_VAR* A1_tmp, TYPE_VAR* A2_tmp, TYPE_VAR* A3_tmp,
	TYPE_VAR* b_tmp) {
	check_return(cudaMemcpy(A1_tmp, A1, S*N*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(A2_tmp, A2, S*N*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(A3_tmp, A3, S*N*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(b_tmp, b, S*N*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
}

/* Global functions */

/* Global solver function called from PCR Method PCR_solve(...) */
__global__ void PCR_solver(TYPE_VAR* A1, TYPE_VAR* A2, TYPE_VAR* A3, TYPE_VAR* B,
	int N) {

	int chunks = (N + CHUNK_MAX - 1)/CHUNK_MAX;
	int delta = 1;
	int sys_offset = blockIdx.x*N;

	while (delta < N) {
		PCR_reduce(A1, A2, A3, B, N, chunks, delta, sys_offset);
		delta *= 2;
		__syncthreads();
	}
}

/* Device functions */

/* Carries reduction on the system for a specified distance between equations (delta) */
__device__ void PCR_reduce(TYPE_VAR* A1, TYPE_VAR* A2, TYPE_VAR* A3, TYPE_VAR* B,
	int N, int chunks, int delta, int sys_offset) {

	double4 eqn1[CR_BUFF_MAX];

	/* Fetch top line values that will get overwritten on chunk boundaries */
	for (int i = 0; i < chunks; i++) {
		int eqn_num = i*CHUNK_MAX + threadIdx.x;
		if (eqn_num < N) {
			int id = sys_offset + eqn_num;
			eqn1[i] = (eqn_num-delta >= 0) ?
				make_double4(A1[id-delta], A2[id-delta], A3[id-delta], B[id-delta]) :
				make_double4(0.0);
		}
	}

	/* Reduce */
	for (int i = 0; i < chunks; i++) {
		int eqn_num = i*CHUNK_MAX + threadIdx.x;
		if (eqn_num < N) {
			int id = sys_offset + eqn_num;

			double4 eqn2 = make_double4(A1[id], A2[id], A3[id], B[id]);

			double4 eqn3 = (eqn_num+delta < N) ?
				make_double4(A1[id+delta], A2[id+delta], A3[id+delta], B[id+delta]) :
				make_double4(0.0);

			__syncthreads();

			TYPE_VAR l2 = eqn2.x; TYPE_VAR Lam1 = (eqn_num-delta >= 0) ? eqn1[i].y : 1.0;
			TYPE_VAR m2 = eqn2.z; TYPE_VAR Lam3 = (eqn_num+delta < N) ? eqn3.y : 1.0;

			eqn1[i] *= (l2*Lam3);
			eqn2 *= (-Lam1*Lam3);
			eqn3 *= (Lam1*m2);

			eqn2.y += eqn1[i].z + eqn3.x;
			eqn2.x = eqn1[i].x;
			eqn2.z = eqn3.z;
			eqn2.w += eqn1[i].w + eqn3.w;
			eqn2 /= eqn2.y;
			A1[id] = eqn2.x;
			A2[id] = eqn2.y;
			A3[id] = eqn2.z;
			B[id] = eqn2.w;
		}
	}
}

/* Solves the 1 unknown system (obsolete) */
__device__ void PCR_solve_eqn(TYPE_VAR* A2, TYPE_VAR* B, int N, int chunks,
	int sys_offset) {
	for (int i = 0; i < chunks; i++) {
		int eqn_num = i*CHUNK_MAX + threadIdx.x;
		if (eqn_num < N) {
			int id = sys_offset + eqn_num;
			B[id] /= A2[id];
		}
	}
}

#ifdef TESTING

int main()
{
	cout << "Hello World!" << endl;
	cout << "This is PCR solver!" << endl;

	int N = 64;
	int S = 1;

	TYPE_VAR A1 [N*S];
	TYPE_VAR A2 [N*S];
	TYPE_VAR A3 [N*S];
	TYPE_VAR B [N*S];
	TYPE_VAR X [N*S];

	for (int i = 0; i < N*S; i++) {
		A1[i] = 1.0;
		A2[i] = -2.0;
		A3[i] = 1.0;
		B[i] = 1.0;
		X[i] = 0.0;
	}
	for (int i = 0; i < N*S; i+=N) {
		A1[i] = 0.0;
	}
	for (int i = N-1; i < N*S; i+=N) {
		A3[i] = 0.0;
	}

	for (int k = 0; k < S; k++) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if ((i != 0) && (j == i-1)) cout << A1[i+k*N] << " ";
				else if (j == i) cout << A2[i+k*N] << " ";
				else if ((i != N-1) && (j == i+1)) cout << A3[i+k*N] << " ";
				else cout << "0 ";
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << endl;

	PCR* pcr= new PCR(N, S);

	pcr->PCR_solve(A1, A2, A3, B, X);

	for (int j = 0; j < S; j++) {
		for (int i = 0; i < N; i++) {
			cout << X[i+j*N] << " ";
		}

		cout << endl;
	}
	cout << endl;

	delete pcr;

	return 0;
}

#endif
