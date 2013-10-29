/*
 * ADI.cu
 *
 *  Created on: 22 Oct 2013
 *      Author: geekity
 */

#include <iostream>
#include <cstdlib>

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

#define OUTPUT
#define TOLL 1e-4

/* ADI class public methods */

/* Constructors */
ADI::ADI(int N_tmp, int S_tmp) {
	S = S_tmp;
	N = N_tmp;
	pcr = new PCR(N, S);

	h_phi_new = (float*) safe_malloc(N*S*sizeof(float));
	check_return(cudaMalloc((float**)&d_phi_new, N*S*sizeof(float)));
	h_phi_bar = (float*) safe_malloc(N*S*sizeof(float));
	check_return(cudaMalloc((float**)&d_phi_bar, N*S*sizeof(float)));
}

ADI::~ADI() {
	delete pcr;

	safe_free(h_phi_new);
	safe_free(h_phi_bar);

	check_return(cudaFree(d_phi_new));
	check_return(cudaFree(d_phi_bar));
}

__host__ void ADI::adi_solver(float* h_phi, float* d_phi, float* d_rho) {
	float dt = 1.0;
	float dh1 = 1.0;
	float dh2 = 1.0;
	bool accept = true;

	bool horizontal = true;

/*	do {
		calcAB<<<S, N>>>(pcr->A1_arr(), pcr->A2_arr(), pcr->A3_arr(), pcr->B_arr(),
				d_phi, d_rho, dt, dh1, dh2, N, S);

		cudaDeviceSynchronize();
		check_return(cudaGetLastError());

	} while (check_err(d_phi, &dt, &accept));*/

	check_arrays();

	cudaDeviceSynchronize();
}

/* ADI class private methods */

void ADI::check_arrays() {
	float* A1 = (float*) safe_malloc(N*S*sizeof(float));
	float* A2 = (float*) safe_malloc(N*S*sizeof(float));
	float* A3 = (float*) safe_malloc(N*S*sizeof(float));
	float* B = (float*) safe_malloc(N*S*sizeof(float));

	check_return(cudaMemcpy(A1, pcr->A1_arr(), N*S*sizeof(float), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(A2, pcr->A2_arr(), N*S*sizeof(float), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(A3, pcr->A3_arr(), N*S*sizeof(float), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(B, pcr->B_arr(), N*S*sizeof(float), cudaMemcpyDeviceToHost));

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

bool ADI::check_err(float* d_phi, float* dt, bool* accept) {
	calc_dif_iter<<<S, N>>>(d_phi_new, d_phi, d_phi_bar, N, S);

	cudaDeviceSynchronize();
	check_return(cudaGetLastError());

	thrust::device_ptr<float> t_phi_new_ptr(d_phi_new);
	thrust::device_vector<float> t_phi_new(t_phi_new_ptr, t_phi_new_ptr+N*S);
	thrust::device_ptr<float> t_phi_bar_ptr(d_phi_bar);
	thrust::device_vector<float> t_phi_bar(t_phi_bar_ptr, t_phi_bar_ptr+N*S);

	float tp_top = thrust::reduce(t_phi_bar.begin(), t_phi_bar.end(),
			(float) 0, thrust::plus<float>());
	float tp_bottom = thrust::reduce(t_phi_new.begin(), t_phi_bar.end(),
			(float) 0, thrust::plus<float>());

	if (tp_bottom < TOLL) return false;

	float tp = tp_top/tp_bottom;
	if (tp <= 0.05) {
		*dt *= 4;
		*accept = true;
	} else if (tp <= 0.1) {
		*dt *= 2;
		*accept = true;
	} else if (tp <= 0.3) {
		*dt *= 1;
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

/* Device functions */

__global__ void calcAB(float* A1, float* A2, float* A3, float* B, float* phi,
	float* rho, float dt, float dh1, float dh2, int N, int S) {

	int tid1 = blockIdx.x*N + threadIdx.x;

	if (tid1 < N*S) {
		A1[tid1] = (threadIdx.x == 0) ? 0.0 : -dt/(dh1*dh1);
		A2[tid1] = 1 + 2*dt/(dh1*dh1) + 2*dt/(dh2*dh2);
		A3[tid1] = (threadIdx.x == N-1) ? 0.0 : -dt/(dh1*dh1);
		int tid2 = threadIdx.x*S + blockIdx.x;
		B[tid1] = rho[tid1]*dt/EPSILON0 + phi[tid2];
		B[tid1] += (blockIdx.x == 0) ? 0.0 : phi[tid2-1]*dt/(dh2*dh2);
		B[tid1] += (blockIdx.x == S-1) ? 0.0 : phi[tid2+1]*dt/(dh2*dh2);
	}
}

__global__ void recalcB(float* B, float* phi, float* rho, float dt, float dh1,
	float dh2, int N, int S) {
	int tid1 = blockIdx.x*N + threadIdx.x;
	if (tid1 < N*S) {
		int tid2 = threadIdx.x*S + blockIdx.x;
		B[tid1] = rho[tid1]*dt/EPSILON0 + phi[tid2];
		B[tid1] += (blockIdx.x == 0) ? 0.0 : phi[tid2-1]*dt/(dh2*dh2);
		B[tid1] += (blockIdx.x == S-1) ? 0.0 : phi[tid2+1]*dt/(dh2*dh2);
	}
}

/* Check difference between iterations */
__global__ void calc_dif_iter(float* phi_new, float* phi_old, float* phi_bar,
	int N, int S) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < N*S) {
		phi_bar[tid] -= phi_new[tid];
		phi_new[tid] -= phi_old[tid];

		phi_bar[tid] *= phi_bar[tid];
		phi_new[tid] *= phi_new[tid];
	}
}

__global__ void transpose(float *iden, float *oden) {

	__shared__ float tile[TILE_WIDTH][TILE_WIDTH];

	int blockIdx_x, blockIdx_y;

	if ((N_COLS-1) == (N_ROWS-1)) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x+blockIdx.y) % gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid/gridDim.y)+blockIdx_y) % gridDim.x;
	}

	int tidx = threadIdx.x + blockIdx_x*TILE_WIDTH;
	int tidy = threadIdx.y + blockIdx_y*TILE_WIDTH;
	int index_in = tidx + tidy*(N_COLS-1);

	for (int i = 0; i < (TILE_WIDTH+SHARE_Y-1); i += SHARE_Y) {
		for (int j = 0; j < (TILE_WIDTH+SHARE_X-1); j += SHARE_X) {
			if (((tidx+j < N_COLS-1)
					&& (tidy+i < N_ROWS-1))
					&& ((i+threadIdx.y < TILE_WIDTH)
					&& (j+threadIdx.x < TILE_WIDTH))) {
				tile[threadIdx.y+i][threadIdx.x+j] = iden[index_in + j + i*(N_COLS-1)];
			}
		}
	}

	__syncthreads();

	tidx = threadIdx.x + blockIdx_y*TILE_WIDTH;
	tidy = threadIdx.y + blockIdx_x*TILE_WIDTH;
	int index_out = tidx + tidy*(N_ROWS-1);

	for (int i = 0; i < (TILE_WIDTH+SHARE_Y-1); i += SHARE_Y) {
		for (int j = 0; j < (TILE_WIDTH+SHARE_X-1); j += SHARE_X) {
			if (((tidx+j < N_ROWS-1)
					&& (tidy+i < N_COLS-1))
					&& ((i+threadIdx.y < TILE_WIDTH)
					&& (j+threadIdx.x < TILE_WIDTH))) {
				oden[index_out + j + i*(N_ROWS-1)] = tile[threadIdx.x+j][threadIdx.y+i];
			}
		}
	}
}

#ifdef OUTPUT

int main() {
	cout << "Hello World!" << endl;

	int N = 5;
	int S = 5;

	float phi[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
			1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	float rho[] = {EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0,
			EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0,
			EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0,
			EPSILON0, EPSILON0, EPSILON0, EPSILON0, EPSILON0};

	float* d_phi;
	float* d_rho;

	check_return(cudaMalloc((float**)&d_phi, N*S*sizeof(float)));
	check_return(cudaMalloc((float**)&d_rho, N*S*sizeof(float)));

	check_return(cudaMemcpy(d_phi, phi, N*S*sizeof(float), cudaMemcpyHostToDevice));
	check_return(cudaMemcpy(d_rho, rho, N*S*sizeof(float), cudaMemcpyHostToDevice));

	ADI* adi = new ADI(N, S);

	adi->adi_solver(phi, d_phi, d_rho);

	cudaFree(d_phi);
	cudaFree(d_rho);

	delete adi;

	return 0;
}

#endif
