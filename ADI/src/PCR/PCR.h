/*
 * PCR.h
 *
 *  Created on: 4 Oct 2013
 *      Author: geekity
 *
 *  Parallel Cyclic Reduction Solver. This solver solves
 *  the equation
 *  	Ax = b
 *  where A is a tridiagonal matrix, b is supplied and x is unknown,
 *  using parallel cyclic reduction method. It mostly uses ideas from
 *
 *  Y. Zhang, J. Cohen, A. Davidson, J. Owens GPU Computing Gems Jade
 *  Edition (2011), Chapter 11
 *
 *  and
 *
 *  Zhangping Wei, Byunghyun Jang, Yaoxin Zhang, Yafei Jia, Procedia
 *  Computer Science 18 (2013) 389 - 398, Section 4.2.1, method 2(b).
 */

#ifndef PCR_H_
#define PCR_H_

#define TYPE_VAR double

class PCR {
private:
	int N;	/* system dimension */
	int S;
	TYPE_VAR* A1;	/* below diagonal in tridiagonal system A */
	TYPE_VAR* A2;	/* diagonal of tridiagonal system A */
	TYPE_VAR* A3;	/* above diagonal in tridiagonal system A */
	TYPE_VAR* b;	/* source vector */

	/* Allocates device memory */
	__host__ void PCR_init(TYPE_VAR* A1_tmp, TYPE_VAR* A2_tmp, TYPE_VAR* A3_tmp,
		TYPE_VAR* b_tmp);
	/* Copies reduced matrix A' to host memory A for testing purposes */
	__host__ void PCR_A_tester(TYPE_VAR* A1_tmp, TYPE_VAR* A2_tmp, TYPE_VAR* A3_tmp,
		TYPE_VAR* b_tmp);
public:
	/* Constructors */
	PCR(int N_tmp, int S_tmp);

	/* Destructor */
	virtual ~PCR();

	/* PCR solver method */
	__host__ void PCR_solve(TYPE_VAR* A1_tmp, TYPE_VAR* A2_tmp, TYPE_VAR* A3_tmp,
		TYPE_VAR* b_tmp, TYPE_VAR* x_tmp);
	__host__ void PCR_solve(TYPE_VAR* x_tmp);

	/* Accessors */
	__host__ TYPE_VAR* A1_arr();
	__host__ TYPE_VAR* A2_arr();
	__host__ TYPE_VAR* A3_arr();
	__host__ TYPE_VAR* B_arr();

	/* ADI direction reverse */
	__host__ void ADI_flip(int N_tmp, int S_tmp);
};

/* Global solver function called from PCR Method PCR_solve(...) */
__global__ void PCR_solver(TYPE_VAR* A1, TYPE_VAR* A2, TYPE_VAR* A3, TYPE_VAR* B,
	int N);

/* Carries reduction on the system for a specified distance between equations (delta) */
__device__ void PCR_reduce(TYPE_VAR* A1, TYPE_VAR* A2, TYPE_VAR* A3, TYPE_VAR* B,
	int N, int chunks, int delta, int sys_offset);

/* Solves the 1 unknown system (obsolete) */
__device__ void PCR_solve_eqn(TYPE_VAR* A2, TYPE_VAR* B, int N, int chunks,
	int sys_offset);

#endif /* PCR_H_ */
