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

	TYPE_VAR* h_phi_new; /* New value of phi after 2 double sweeps */
	TYPE_VAR* d_phi_new;
	TYPE_VAR* d_phi_bar; /* Value of phi after 1 double sweep of step size 2*dt */
	TYPE_VAR* d_u; /* Old value of phi for conversion checking */
	TYPE_VAR* h_arr; /* Reduction helper array */

	TYPE_VAR* phi_trans; /* helper array for transpose phi */
	TYPE_VAR* rho_trans; /* helper array for transpose rho*/

	void check_arrays();
	bool check_err(TYPE_VAR* d_phi, TYPE_VAR* rho, TYPE_VAR* dt, bool* accept, TYPE_VAR dh1, TYPE_VAR dh2);
	void double_sweep(TYPE_VAR* phi_new, TYPE_VAR* rho, TYPE_VAR dt,
		TYPE_VAR dh1, TYPE_VAR dh2);
	void transposes(TYPE_VAR* phi_new);
	TYPE_VAR my_reduction(TYPE_VAR* d_arr);
	void assert_notnan(TYPE_VAR* d_arr);
public:
	ADI(int N_tmp, int S_tmp);
	~ADI();

	__host__ void adi_solver(TYPE_VAR* d_phi, TYPE_VAR* d_rho);
};

/* calculate tridiagonal matrix A and RHS vector B */
__global__ void calcAB(TYPE_VAR* A1, TYPE_VAR* A2, TYPE_VAR* A3, TYPE_VAR* B, TYPE_VAR* phi,
	TYPE_VAR* rho, TYPE_VAR dt, TYPE_VAR dh1, TYPE_VAR dh2, int N, int S);

/* Calculate new B vector only (case if dt is constant) */
__global__ void recalcB(TYPE_VAR* B, TYPE_VAR* phi, TYPE_VAR* rho, TYPE_VAR dt, TYPE_VAR dh1,
	TYPE_VAR dh2, int N, int S);

/* Check difference between iterations */
__global__ void calc_dif_iter(TYPE_VAR* phi_new, TYPE_VAR* phi_old, TYPE_VAR* phi_bar,
	int N, int S);

/* Transpose density array */
__global__ void transpose(TYPE_VAR *iden, TYPE_VAR *oden, int N, int S);

/* partial reduction in shared memory */
__global__ void shared_reduction(TYPE_VAR* arr, int size);

/* check for convergence */
__global__ void ADI_converge(TYPE_VAR* phi, TYPE_VAR* rho, int N, int S, TYPE_VAR dh1, TYPE_VAR dh2);

/* Rescale */
__global__ void ADI_rescale(TYPE_VAR* phi, TYPE_VAR scale, int N, int S);

/* Test function to see how well solution satisfies equation */
void ADI_test(TYPE_VAR* phi, TYPE_VAR* rho, int N, int S, TYPE_VAR dh1, TYPE_VAR dh2);

#endif /* ADI_H_ */
