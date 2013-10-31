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

	PCR* pcr;		/* Parallel cyclic reduction solver*/

	float* h_phi_new; /* New value of phi after 2 double sweeps */
	float* d_phi_new;
	float* h_phi_bar; /* Value of phi after 1 double sweep of step size 2*dt */
	float* d_phi_bar;

	float* phi_trans; /* helper array for transpose phi */
	float* rho_trans; /* helper array for transpose rho*/

	void check_arrays();
	bool check_err(float* d_phi, float* dt, bool* accept);
	void double_sweep(float* phi_new, float* rho, float dt,
		float dh1, float dh2);
	void transposes(float* phi_new);
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

/* Check difference between iterations */
__global__ void calc_dif_iter(float* phi_new, float* phi_old, float* phi_bar, int N, int S);

/* Transpose density array */
__global__ void transpose(float *iden, float *oden, int N, int S);

#endif /* ADI_H_ */
