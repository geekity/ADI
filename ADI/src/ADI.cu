/*
 * ADI.cu
 *
 *  Created on: 22 Oct 2013
 *      Author: geekity
 */

#include <iostream>
#include <cstdlib>
#include "ADI.h"
#include "constants/constants.h"
#include "constants/alloc.h"

using namespace std;

#define OUTPUT

/* ADI class public methods */

/* Constructors */
ADI::ADI(int N_tmp, int S_tmp) {
	S = S_tmp;
	N = N_tmp;
	pcrh = new PCR(N, S);
	pcrv = new PCR(S, N);

	h_phi_new = NULL;
	d_phi_new = NULL;
	h_phi_bar = NULL;
	d_phi_bar = NULL;
}

ADI::~ADI() {
	delete pcrh;
	delete pcrv;

	safe_free(h_phi_new);
	safe_free(h_phi_bar);

	check_return(cudaFree(d_phi_new));
	check_return(cudaFree(d_phi_bar));
}

__host__ void ADI::adi_solver(float* d_phi, float* d_rho) {
	float dt = 0.5;
	float dh1 = H;
	float dh2 = H;

	bool horizontal = true;

	calcAB<<<S, N>>>(pcrh->A1_arr(), pcrh->A2_arr(), pcrh->A3_arr(), pcrh->B_arr(),
			d_phi, d_rho, dt, dh1, dh2, N, S);

	cudaDeviceSynchronize();
}

/* ADI class private methods */

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

__global__ void density_transpose(float *iden, float *oden) {

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
	cout << "Hello World!";

	return 0;
}

#endif
