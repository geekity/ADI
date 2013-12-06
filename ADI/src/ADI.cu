/*
 * ADI.cu
 *
 *  Created on: 22 Oct 2013
 *      Author: geekity
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>

#include "ADI.h"
#include "constants/constants.h"
#include "constants/alloc.h"

using namespace std;

#define TESTING
#define TOLL 1e-4
#define CHUNK_MAX 256
//#define AN (-Q_E*DT/H*DT/H/2)

/* ADI class public methods */

/* Constructors */
ADI::ADI(int N_tmp, int S_tmp) {
	S = S_tmp;
	N = N_tmp;
	pcr = new PCR(N, S);
	old_err = sqrt(-1);

	h_phi_new = (TYPE_VAR*) safe_malloc(N*S*sizeof(TYPE_VAR));
	h_arr = (TYPE_VAR*) safe_malloc(N*S*sizeof(TYPE_VAR));
	check_return(cudaMalloc((TYPE_VAR**)&d_phi_new, N*S*sizeof(TYPE_VAR)));
	check_return(cudaMalloc((TYPE_VAR**)&d_phi_bar, N*S*sizeof(TYPE_VAR)));
	check_return(cudaMalloc((TYPE_VAR**)&d_u, N*S*sizeof(TYPE_VAR)));
	check_return(cudaMalloc((TYPE_VAR**)&phi_trans, N*S*sizeof(TYPE_VAR)));
	check_return(cudaMalloc((TYPE_VAR**)&rho_trans, N*S*sizeof(TYPE_VAR)));
}

/* Destructor */
ADI::~ADI() {
	delete pcr;

	safe_free(h_phi_new);
	safe_free(h_arr);

	check_return(cudaFree(d_phi_new));
	check_return(cudaFree(d_phi_bar));
	check_return(cudaFree(d_u));
	check_return(cudaFree(phi_trans));
	check_return(cudaFree(rho_trans));
}

/* Actual DADI solver implementation */
__host__ void ADI::adi_solver(TYPE_VAR* d_phi, TYPE_VAR* d_rho) {
	/**/
	TYPE_VAR dt = 1.0;
	TYPE_VAR dh1 = 1.0;
	TYPE_VAR dh2 = 1.0;
	bool accept = false;	/* bool to determine whether iteration was accepted */

	/* finds the transpose of rho for building up of PCR solver RHS */
	transpose<<<BLOCKS, THREADS>>>(d_rho, rho_trans, N, S);
	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	/* loops until convergence is achieved */
	do {
		if (accept) {
			check_return(cudaMemcpy(d_phi, h_phi_new, N*S*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));
		}

		check_return(cudaMemcpy(d_phi_new, d_phi, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToDevice));
		check_return(cudaMemcpy(d_phi_bar, d_phi, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToDevice));

		/* 2 double sweeps of 1*dt */
		double_sweep(d_phi_new, d_rho, dt, dh1, dh2);
		double_sweep(d_phi_new, d_rho, dt, dh1, dh2);

		check_return(cudaMemcpy(h_phi_new, d_phi_new, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));

		/* 1 double sweep of 2*dt */
		double_sweep(d_phi_bar, d_rho, 2*dt, dh1, dh2);

	} while (check_err(d_phi, d_rho, &dt, &accept, dh1, dh2));

	check_return(cudaMemcpy(d_phi, h_phi_new, N*S*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

/*	ADI_rescale<<<S, N>>>(d_phi, EPSILON0, N, S);
	cudaDeviceSynchronize();
	check_return(cudaGetLastError());*/
}

/* ADI class private methods */

/* Check PCR matrix seeding from kerne functions */
void ADI::check_arrays() {
	TYPE_VAR* A1 = (TYPE_VAR*) safe_malloc(N*S*sizeof(TYPE_VAR));
	TYPE_VAR* A2 = (TYPE_VAR*) safe_malloc(N*S*sizeof(TYPE_VAR));
	TYPE_VAR* A3 = (TYPE_VAR*) safe_malloc(N*S*sizeof(TYPE_VAR));
	TYPE_VAR* B = (TYPE_VAR*) safe_malloc(N*S*sizeof(TYPE_VAR));

	check_return(cudaMemcpy(A1, pcr->A1_arr(), N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(A2, pcr->A2_arr(), N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(A3, pcr->A3_arr(), N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(B, pcr->B_arr(), N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));

	cout << "A1: " << endl;
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < N; j++) {
			cout << A1[i*N+j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "A2: " << endl;
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < N; j++) {
			cout << A2[i*N+j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "A3: " << endl;
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < N; j++) {
			cout << A3[i*N+j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "B: " << endl;
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < N; j++) {
			cout << B[i*N+j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	safe_free(A1);
	safe_free(A2);
	safe_free(A3);
	safe_free(B);
}

/* Checks solution convergence and adjusts dt */
bool ADI::check_err(TYPE_VAR* d_phi, TYPE_VAR* rho, TYPE_VAR* dt, bool* accept,
	TYPE_VAR dh1, TYPE_VAR dh2) {
	check_return(cudaMemcpy(d_u, d_phi, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToDevice));
	calc_dif_iter<<<S, N>>>(d_phi_new, d_u, d_phi_bar, N, S);

	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	TYPE_VAR tp_top = my_reduction(d_phi_bar);
	TYPE_VAR tp_bottom = my_reduction(d_phi_new);

//	check_return(cudaMemcpy(d_u, rho, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToDevice));
//	check_return(cudaMemcpy(d_phi_new, h_phi_new, N*S*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));

//	ADI_converge<<<S, N>>>(d_phi_new, d_u, N, S, dh1, dh2);
	TYPE_VAR tp_u = my_reduction(d_u);

	tp_top = sqrt(tp_top);
	tp_bottom = sqrt(tp_bottom);
	tp_u = sqrt(tp_u);

	assert(tp_top==tp_top);
	assert(tp_bottom==tp_bottom);
	assert(tp_u==tp_u);

	if (tp_u == old_err) {
		cout << "Error! You have exceeded the accuracy of the solver and caused an overflow!" << endl;
		exit(1);
	} else {
		old_err = tp_u;
	}

	cout << "tp_u: " << tp_u << endl;

	if (tp_u < TOLL) return false;

	TYPE_VAR tp = tp_top/tp_bottom;
	if (tp <= 0.05) {
		*dt *= 4;
		*accept = true;
	} else if (tp <= 0.1) {
		*dt *= 2;
		*accept = true;
	} else if (tp <= 0.3) {
		*dt *= sqrt(5.0);
		*accept = true;
	} else if (tp <= 0.4) {
		*dt *= 0.5;
		*accept = true;
	} else if (tp <= 0.6) {
		*dt *= 0.25;
		*accept = true;
	} else {
		*dt *= 0.0625;
		*accept = false;
	}
	return true;
}

/* Single double sweep of ADI solver. Input rho & phi are expected to be row sorted */
/* (i.e. x = column, y = row, index_A = y*N + x)                                    */
void ADI::double_sweep(TYPE_VAR* phi_new, TYPE_VAR* rho, TYPE_VAR dt,
	TYPE_VAR dh1, TYPE_VAR dh2) {

	/* system solved along vertical */
	pcr->ADI_flip(S, N);	// sets up PCR # of equations and # of systems

	// Sets up matrix A and vector B for PCR solver
	calcAB<<<N, S>>>(pcr->A1_arr(), pcr->A2_arr(), pcr->A3_arr(), pcr->B_arr(),
			phi_new, rho_trans, dt, dh2, dh1, S, N);
	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	// solve the mesh system
	pcr->PCR_solve(phi_new);

	// transpose phi for setting up vector B for next sweep
	transposes(phi_new);

	/* system solved along horizontal */
	pcr->ADI_flip(N, S);	// sets up PCR # of equations and # of systems

	// Sets up matrix A and vector B for PCR solver
	calcAB<<<S, N>>>(pcr->A1_arr(), pcr->A2_arr(), pcr->A3_arr(), pcr->B_arr(),
			phi_new, rho, dt, dh1, dh2, N, S);
	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	// solve the mesh system
	pcr->PCR_solve(phi_new);

	// transpose phi to return to original orientation
	transposes(phi_new);
}

/* transposes phi "in place" (not really in place, just a wrapper method) */
void ADI::transposes(TYPE_VAR* phi_new) {
	transpose<<<BLOCKS, THREADS>>>(phi_new, phi_trans, N, S);
	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	check_return(cudaMemcpy(phi_new, phi_trans, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToDevice));
}

/* Sum an array */
TYPE_VAR ADI::my_reduction(TYPE_VAR* d_arr)	{
	int B = (N*S+CHUNK_MAX-1)/CHUNK_MAX;
	shared_reduction<<<B, CHUNK_MAX>>>(d_arr, N*S);

	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	check_return(cudaMemcpy(h_arr, d_arr, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));

	for (int i = CHUNK_MAX; i < S*N; i += CHUNK_MAX) {
		h_arr[0] += h_arr[i];
	}
	return h_arr[0];
}

/* assert array does not consist of NaN */
void ADI::assert_notnan(TYPE_VAR* d_arr) {
	check_return(cudaMemcpy(h_arr, d_arr, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));
	for (int i = 0; i < N*S; i++) {
		assert(h_arr[i] == h_arr[i]);
	}
}


/* Device functions */

/* Set up the A matrix and B vector for 2 dimensional Poisson equation */
__global__ void calcAB(TYPE_VAR* A1, TYPE_VAR* A2, TYPE_VAR* A3, TYPE_VAR* B, TYPE_VAR* phi,
	TYPE_VAR* rho, TYPE_VAR dt, TYPE_VAR dh1, TYPE_VAR dh2, int N, int S) {

	int tid1 = blockIdx.x*N + threadIdx.x;

	if (tid1 < N*S) {
		A1[tid1] = (threadIdx.x == 0) ? 0.0 : -dt/(dh1*dh1);
		A2[tid1] = 1 + 2*dt/(dh1*dh1) + 2*dt/(dh2*dh2);
		A3[tid1] = (threadIdx.x == N-1) ? 0.0 : -dt/(dh1*dh1);
		int tid2 = threadIdx.x*S + blockIdx.x;
		B[tid1] = rho[tid1]*dt + phi[tid2];
		B[tid1] += (blockIdx.x == 0) ? 0.0 : phi[tid2-1]*dt/(dh2*dh2);
		B[tid1] += (blockIdx.x == S-1) ? 0.0 : phi[tid2+1]*dt/(dh2*dh2);
	}
}

/* Calculate new B vector only (case if dt is constant) */
__global__ void recalcB(TYPE_VAR* B, TYPE_VAR* phi, TYPE_VAR* rho, TYPE_VAR dt, TYPE_VAR dh1,
	TYPE_VAR dh2, int N, int S) {
	int tid1 = blockIdx.x*N + threadIdx.x;
	if (tid1 < N*S) {
		int tid2 = threadIdx.x*S + blockIdx.x;
		B[tid1] = rho[tid1]*dt + phi[tid2];
		B[tid1] += (blockIdx.x == 0) ? 0.0 : phi[tid2-1]*dt/(dh2*dh2);
		B[tid1] += (blockIdx.x == S-1) ? 0.0 : phi[tid2+1]*dt/(dh2*dh2);
	}
}

/* Check difference between iterations */
__global__ void calc_dif_iter(TYPE_VAR* phi_new, TYPE_VAR* phi_old, TYPE_VAR* phi_bar,
	int N, int S) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < N*S) {
		phi_bar[tid] -= phi_new[tid];
		phi_new[tid] -= phi_old[tid];
		phi_old[tid] = (phi_old[tid] == 0.0) ? phi_new[tid] : phi_new[tid]/phi_old[tid];

		phi_bar[tid] *= phi_bar[tid];
		phi_new[tid] *= phi_new[tid];
		phi_old[tid] *= phi_old[tid];
	}
}

/* Matrix transpose (see G. Reutsch, P. Micikevicius, Optimizing matrix transpose in CUDA) */
__global__ void transpose(TYPE_VAR *iden, TYPE_VAR *oden, int N, int S) {

	// Dimension of tile adjusted to avoid bank conflicts
	__shared__ TYPE_VAR tile[TILE_WIDTH][TILE_WIDTH+1];

	int blockIdx_x, blockIdx_y;

	// Use coordinates in diagonal context to avoid partition camping
	if (N == S) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x+blockIdx.y) % gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y) % gridDim.x;
	}

	int tidx = threadIdx.x + blockIdx_x*TILE_WIDTH;
	int tidy = threadIdx.y + blockIdx_y*TILE_WIDTH;
	int index_in = tidx + tidy*N;

	// copy data to tile
	for (int i = 0; i < (TILE_WIDTH+SHARE_Y-1); i += SHARE_Y) {
		for (int j = 0; j < (TILE_WIDTH+SHARE_X-1); j += SHARE_X) {
			if (((tidx+j < N)
					&& (tidy+i < S))
					&& ((i+threadIdx.y < TILE_WIDTH)
					&& (j+threadIdx.x < TILE_WIDTH))) {
				tile[threadIdx.y+i][threadIdx.x+j] = iden[index_in + j + i*N];
			}
		}
	}

	__syncthreads();

	tidx = threadIdx.x + blockIdx_y*TILE_WIDTH;
	tidy = threadIdx.y + blockIdx_x*TILE_WIDTH;
	int index_out = tidx + tidy*S;

	// insert transpose into output array
	for (int i = 0; i < (TILE_WIDTH+SHARE_Y-1); i += SHARE_Y) {
		for (int j = 0; j < (TILE_WIDTH+SHARE_X-1); j += SHARE_X) {
			if (((tidx+j < S)
					&& (tidy+i < N))
					&& ((i+threadIdx.y < TILE_WIDTH)
					&& (j+threadIdx.x < TILE_WIDTH))) {
				oden[index_out + j + i*S] = tile[threadIdx.x+j][threadIdx.y+i];
			}
		}
	}
}

/* partial reduction in shared memory */
__global__ void shared_reduction(TYPE_VAR* arr, int size) {
	__shared__ TYPE_VAR a[CHUNK_MAX];
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < size) {
		a[threadIdx.x] = arr[tid];

		__syncthreads();

		int step = CHUNK_MAX/2;

		while (step > 0) {
			if ((threadIdx.x < step) && (tid+step < size))
				a[threadIdx.x] += a[threadIdx.x+step];
			__syncthreads();
			step /= 2;
		}

		if (threadIdx.x == 0) arr[tid] = a[threadIdx.x];
		else arr[tid] = 0.0;
	}
}

/* check for convergence */
__global__ void ADI_converge(TYPE_VAR* phi, TYPE_VAR* rho, int N, int S, TYPE_VAR dh1, TYPE_VAR dh2) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < N*S) {
		TYPE_VAR dif1 = (threadIdx.x == 0) ? -2*phi[tid] : -2*phi[tid] + phi[tid-1];
		dif1 += (threadIdx.x == N-1) ? 0 : phi[tid+1];
		dif1 /= (dh1*dh1);
		TYPE_VAR dif2 = (blockIdx.x == 0) ? -2*phi[tid] : -2*phi[tid] + phi[tid-N];
		dif2 += (blockIdx.x == S-1) ? 0 : phi[tid+N];
		dif2 /= (dh2*dh2);

		rho[tid] = rho[tid] + dif1 + dif2;
		rho[tid] *= rho[tid];
	}
}

/* Rescale */
__global__ void ADI_rescale(TYPE_VAR* phi, TYPE_VAR scale, int N, int S) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < N*S) {
		phi[tid] /= scale;
	}
}

/* Test function to see how well solution satisfies equation */
void ADI_test(TYPE_VAR* phi, TYPE_VAR* rho, int N, int S, TYPE_VAR dh1, TYPE_VAR dh2) {

	for (int i = 0; i < S; i++) {
		for (int j = 0; j < N; j++) {
			TYPE_VAR dif1 = (j == 0) ? -2*phi[i*N+j] : -2*phi[i*N+j] + phi[i*N+j-1];
			dif1 += (j == N-1) ? 0 : phi[i*N+j+1];
			dif1 /= (dh1*dh1);
			TYPE_VAR dif2 = (i == 0) ? -2*phi[i*N+j] : -2*phi[i*N+j] + phi[i*N+j-N];
			dif2 += (i == S-1) ? 0 : phi[i*N+j+N];
			dif2 /= (dh2*dh2);

			rho[i*N+j] = rho[i*N+j] + dif1 + dif2;
		}
	}
}

/* test main */

#ifdef TESTING

int main() {
	cout << "Hello World!" << endl;

	int N = 63;
	int S = 63;
	ADI* adi = new ADI(N, S);

	TYPE_VAR phi[N*S];
	TYPE_VAR rho[N*S];
	for (int i = 0; i < N*S; i++) {
		phi[i] = 0.0;
		rho[i] = 1.6e-4*(H*H)/EPSILON0;
//		rho[i] = 0;
	}

	TYPE_VAR* d_phi;
	TYPE_VAR* d_rho;

	check_return(cudaMalloc((TYPE_VAR**)&d_phi, N*S*sizeof(TYPE_VAR)));
	check_return(cudaMalloc((TYPE_VAR**)&d_rho, N*S*sizeof(TYPE_VAR)));

	check_return(cudaMemcpy(d_phi, phi, N*S*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));
	check_return(cudaMemcpy(d_rho, rho, N*S*sizeof(TYPE_VAR), cudaMemcpyHostToDevice));

	adi->adi_solver(d_phi, d_rho);

	check_return(cudaMemcpy(phi, d_phi, N*S*sizeof(TYPE_VAR), cudaMemcpyDeviceToHost));

	for (int i = 0; i < S; i++) {
		for (int j = 0; j < N; j++) {
//			phi[i*N+j] *= Q_E*DENSITY/512*(H*H)/EPSILON0;
			cout << phi[i*N+j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	ADI_test(phi, rho, N, S, 1.0, 1.0);
	TYPE_VAR e_tot = 0.0;
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < N; j++) {
			e_tot += rho[i*N+j]*rho[i*N+j];
		}
	}
	e_tot = sqrt(e_tot);
	cout << "err: " << e_tot << endl;

	cudaFree(d_phi);
	cudaFree(d_rho);

	delete adi;

	return 0;
}

#endif
