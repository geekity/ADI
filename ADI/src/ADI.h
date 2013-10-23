/*
 * ADI.h
 *
 *  Created on: 22 Oct 2013
 *      Author: geekity
 */

#ifndef ADI_H_
#define ADI_H_

#include "PCR/PCR.h"

class ADI {
private:
	int N;
	int S;

	PCR* pcrh;		/* Parallel cyclic reduction solver for horizontal iteration*/
	PCR* pcrv;		/* Parallel cyclic reduction solver for vertical iteration */

	float* h_phi_new; /* New value of phi after 2 double sweeps */
	float* d_phi_new;
	float* h_phi_bar; /* Value of phi after 1 double sweep of step size 2*dt */
	float* d_phi_bar;
public:
	ADI(int N_tmp, int S_tmp);
	~ADI();

	__host__ void adi_solver(float* d_phi, float* d_rho);
};

/* calculate tridiagonal matrix A and RHS vector B */
__global__ void calcAB(float* A1, float* A2, float* A3, float* B, float* phi,
	float* rho, float dt, float dh1, float dh2, int N, int S);

__global__ void recalcB(float* B, float* phi, float* rho, float dt, float dh1,
	float dh2, int N, int S);

/* Transpose density array */
__global__ void density_transpose(float *iden, float *oden);

#endif /* ADI_H_ */
