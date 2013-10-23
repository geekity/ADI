/*
 * constants.h
 *
 *  Created on: 28 Sep 2012
 *      Author: geekity
 *
 *  Note: Unless variable is specified under preprocessor directive #define
 *  it will not be accessible in the Kernel unless parsed. I will see if I
 *  can get around this later (hopefully).
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <cmath>
using namespace std;

/* Physical/Mathematical constants */
#define Q_E 1.602e-19f
#define ME 9.109e-31f
//#define MI 6.63e-27f  // Atomic mass of Argon
#define MI 6.67e-27f	// Atomic mass of Helium
#define KB 1.38e-23f
#define EPSILON0 8.85e-12f
#define M_ME MI/ME
#define P_ATM 101.325e3f
#define P_LOW 13.33f // 100 mTorr

/* Plasma parameters */
#define DENSITY 1.0e15f
#define T_E 30000.0f
#define T_I 300.0f
#define OMEGA sqrtf(DENSITY*Q_E*Q_E/(EPSILON0*ME))
#define LAMBDA sqrtf(EPSILON0*KB*T_E/(Q_E*Q_E*DENSITY))

/* Simulation constants */
#define WPDT 0.2
#define H (0.5f*LAMBDA)
#define DT (WPDT/OMEGA)

/* Simulation size */
#define N_ROWS 64
#define N_COLS 64
#define N_CELLS (N_ROWS*N_COLS)
#define N_PARTICLE 512
#define N_SPECIE 2

/* Derived parameters */
#define V_E (sqrtf(KB*T_E/ME)*DT/H)					/* Thermal Electron Velocity */
#define V_I (sqrtf(KB*T_I/MI)*DT/H)
#define N_S (DENSITY*H/N_PARTICLE)
#define Q_S (Q_E*N_S)
#define ANORM (Q_E*DT*DT/(2*ME*H*H)*Q_S/EPSILON0)
#define N_G (P_LOW/(KB*T_I)*DT)
//#define N_G 0.01f
//#define N_G 3.55e21*(H*H*H)


/* CUDA thread/block sizes */
/* specific to multiprocessor count on the card we are running on */
/*#if (N_CELL <= 512*4)
#define BLOCKS 4
#define THREADS ( (N_CELL + BLOCKS - 1)/BLOCKS )
#else
#define THREADS 512
#define BLOCKS ( (N_CELL + THREADS - 1)/THREADS )
#endif*/
#define SHARE_X 8
#define SHARE_Y 8
#define TILE_WIDTH (SHARE_X >= SHARE_Y) ? SHARE_X : SHARE_Y

#define THREADS dim3(SHARE_X, SHARE_Y, 1)
#define BLOCKS dim3((N_COLS + SHARE_X - 1)/SHARE_X, (N_ROWS + SHARE_Y - 1)/SHARE_Y, 1)

#define BLOCK_SIZE_X ((N_COLS + SHARE_X - 1)/SHARE_X)
#define BLOCK_SIZE_Y ((N_ROWS + SHARE_Y - 1)/SHARE_Y)

/* Specie parameters */
#if 1
const float V_T[2] = {(sqrtf(KB*T_E/ME)*DT/H), (sqrtf(KB*T_I/MI)*DT/H)};
const float CHARGE[2] = {-1.0, 1.0};
const float MASS[2] = {ME, MI};
const float MU[2] = {9.0987584e-31, 3.335e-27};
//const float MU[2] = {3.35785417e-22, 1.230615e-17};
const float SCALE_FACTOR[2] = {0.0, 0.0};
#endif

/* Periodic boundary condition */
#if defined(PERIODIC)
#undef(PERIODIC)
#endif

/* Output of potential data */
#define FIELD_OUTPUT

/* Collision data */
//#define NU_C 0.02f
#define P_IONIZ 0.00
#define P_CAP 0.0
#define EN_I 0.1


#endif /* CONSTANTS_H_ */
