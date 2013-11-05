/*
 * ADI.h
 *
 *  Created on: 22 Oct 2013
 *      Author: geekity
 *
 *      This class is a GPU dynamic ADI solver for the 2 dimensional Poisson equation
 *      in time-differenced parabolic form:
 *
 *      d2f/dx2 + d2f/dy2 = df/dt - r/EPSILON0
 *
 *      By solving this equation for steady state, we remove the time derivative
 *      and the equation becomes the elliptic equation. The dt value is not actual
 *      timestep in PIC simulation, it is only a convergence parameter.
 *
 *      After rearranging into a form Ax = B, LHS consists of f in the direction
 *      currently being solved and RHS contains source term and f solutions from
 *      lines above.
 *
 *      More information on the procedure is given in:
 *      * W. F. Ames, Numerical Methods For Partial Differential Equations, 1977 Academic Press, Inc.
 *      * S. Doss, K. Miller, Dynamic ADI Methods for Elliptic Equations, SIAM J. Numer. Anal.,
 *        vol. 16, No. 5, October 1979
 *      * V. Vahedi, G. DiPeso, Simultaneous Potential and Circuit Solution for Two-Dimensional Bounded
 *        Plasma Simulation Codes, Jouranl of Computational Physics 131, 149-163, 1997
 *      * Z. Wei, B. Jang, Y. Zhang, Y. Jia, Parallelizing Alternating Direction Implicit Solver on
 *        GPUs, Procedia Computer Science 18 (2013) 389-398
 */

#ifndef ADI_H_
#define ADI_H_

#include "PCR/PCR.h"

class ADI {
private:
	int N;	/* # of equations */
	int S;	/* # of systems of equations */

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

/* Calculate new B vector only (case if dt is constant) */
__global__ void recalcB(float* B, float* phi, float* rho, float dt, float dh1,
	float dh2, int N, int S);

/* Check difference between iterations */
__global__ void calc_dif_iter(float* phi_new, float* phi_old, float* phi_bar, int N, int S);

/* Transpose density array */
__global__ void transpose(float *iden, float *oden, int N, int S);

#endif /* ADI_H_ */
