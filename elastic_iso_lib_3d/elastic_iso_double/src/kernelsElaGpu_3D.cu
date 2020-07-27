#include "varElaDeclare_3D.h"
#include <stdio.h>




/******************************************************************************/
/******************************* Forward stepper ******************************/
/******************************************************************************/
/* Forward stepper (no damping) */
__global__ void stepFwdGpu_3D(double *dev_o_vx, double *dev_o_vy, double *dev_o_vz, double *dev_o_sigmaxx, double *dev_o_sigmayy, double *dev_o_sigmazz, double *dev_o_sigmaxy, double *dev_o_sigmaxz, double *dev_o_sigmayz, double *dev_c_vx, double *dev_c_vy, double *dev_c_vz, double *dev_c_sigmaxx, double *dev_c_sigmayy, double *dev_c_sigmazz, double *dev_c_sigmaxy, double *dev_c_sigmaxz, double *dev_c_sigmayz, double *dev_n_vx, double *dev_n_vy, double *dev_n_vz, double *dev_n_sigmaxx, double *dev_n_sigmayy, double *dev_n_sigmazz, double *dev_n_sigmaxy, double *dev_n_sigmaxz, double *dev_n_sigmayz, double* dev_rhoxDtw, double* dev_rhoyDtw, double* dev_rhozDtw, double* dev_lamb2MuDtw, double* dev_lambDtw, double* dev_muxzDtw, double* dev_muxyDtw, double* dev_muyzDtw){

	// Allocate shared memory for a specific block
	__shared__ double shared_c_vx[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vx wavefield y-slice block
	__shared__ double shared_c_vy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vy wavefield y-slice block
	__shared__ double shared_c_vz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vz wavefield y-slice block
	__shared__ double shared_c_sigmaxx[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxx wavefield y-slice block
	__shared__ double shared_c_sigmayy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmayy wavefield y-slice block
	__shared__ double shared_c_sigmazz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmazz wavefield y-slice block
	__shared__ double shared_c_sigmaxz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxz wavefield y-slice block
	__shared__ double shared_c_sigmaxy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxy wavefield y-slice block
	__shared__ double shared_c_sigmayz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmayz wavefield y-slice block

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis
	// Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

	// Allocate the arrays that will store the wavefield values in the y-direction
	// Only some components need the be used for derivative computation along the y axis
	double dev_c_vx_y[2*FAT-1];
	double dev_c_vy_y[2*FAT-1];
	double dev_c_vz_y[2*FAT-1];
	double dev_c_sigmayy_y[2*FAT-1];
	double dev_c_sigmaxy_y[2*FAT-1];
	double dev_c_sigmayz_y[2*FAT-1];

	// Number of elements in one y-slice
	long long yStride = dev_nz * dev_nx;

	// Global index of the first element at which we are going to compute the Laplacian
	// Skip the first FAT elements on the y-axis
	long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

	// Global index of the element with the smallest y-position needed to compute derivatives at iGlobal
	long long iGlobalTemp = iGlobal - FAT * yStride;

	// Loading stride for Vx along the y-direction (backward derivative)
	dev_c_vx_y[1] = dev_c_vx[iGlobalTemp]; // iy = 0
	dev_c_vx_y[2] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 1
	dev_c_vx_y[3] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 2
	shared_c_vx[ixLocal][izLocal] = dev_c_vx[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_vx_y[4] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vx_y[5] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 5
	dev_c_vx_y[6] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 6

	// Loading for Vy along the y-direction (forward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_vy_y[1] = dev_c_vy[iGlobalTemp]; // iy = 0
	dev_c_vy_y[2] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 1
	shared_c_vy[ixLocal][izLocal] = dev_c_vy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_vy_y[3] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 3
	dev_c_vy_y[4] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vy_y[5] = dev_c_vy[iGlobalTemp+=yStride];// iy = 5
	dev_c_vy_y[6] = dev_c_vy[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loading for Vz along the y-direction (backward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_vz_y[1] = dev_c_vz[iGlobalTemp]; // iy = 0
	dev_c_vz_y[2] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 1
	dev_c_vz_y[3] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 2
	shared_c_vz[ixLocal][izLocal] = dev_c_vz[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_vz_y[4] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vz_y[5] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 5
	dev_c_vz_y[6] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 6

	// Loading for Sigmaxy along the y-direction (forward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_sigmaxy_y[1] = dev_c_sigmaxy[iGlobalTemp]; // iy = 0
	dev_c_sigmaxy_y[2] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 1
	shared_c_sigmaxy[ixLocal][izLocal] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_sigmaxy_y[3] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 3
	dev_c_sigmaxy_y[4] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmaxy_y[5] = dev_c_sigmaxy[iGlobalTemp+=yStride];// iy = 5
	dev_c_sigmaxy_y[6] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loading for Sigmayy along the y-direction (backward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_sigmayy_y[1] = dev_c_sigmayy[iGlobalTemp]; // iy = 0
	dev_c_sigmayy_y[2] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 1
	dev_c_sigmayy_y[3] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 2
	shared_c_sigmayy[ixLocal][izLocal] = dev_c_sigmayy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_sigmayy_y[4] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmayy_y[5] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 5
	dev_c_sigmayy_y[6] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 6

	// Loading for Sigmayz along the y-direction (forward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_sigmayz_y[1] = dev_c_sigmayz[iGlobalTemp]; // iy = 0
	dev_c_sigmayz_y[2] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 1
	shared_c_sigmaxz[ixLocal][izLocal] = dev_c_sigmayz[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_sigmayz_y[3] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 3
	dev_c_sigmayz_y[4] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmayz_y[5] = dev_c_sigmayz[iGlobalTemp+=yStride];// iy = 5
	dev_c_sigmayz_y[6] = dev_c_sigmayz[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loop over y
	for (long long iy=FAT; iy<dev_ny-FAT; iy++){
		// Update Vx values along the y-axis
		dev_c_vx_y[0] = dev_c_vx_y[1];
		dev_c_vx_y[1] = dev_c_vx_y[2];
		dev_c_vx_y[2] = dev_c_vx_y[3];
		dev_c_vx_y[3] = shared_c_vx[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
		shared_c_vx[ixLocal][izLocal] = dev_c_vx_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vx_y[4] = dev_c_vx_y[5];
		dev_c_vx_y[5] = dev_c_vx_y[6];
		dev_c_vx_y[6] = dev_c_vx_y[iGlobalTemp+yStride];

		// Update Vy values along the y-axis
		dev_c_vy_y[0] = dev_c_vy_y[1];
		dev_c_vy_y[1] = dev_c_vy_y[2];
		dev_c_vy_y[2] = shared_c_vy[ixLocal][izLocal];
		shared_c_vy[ixLocal][izLocal] = dev_c_vy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vy_y[3] = dev_c_vy_y[4];
		dev_c_vy_y[4] = dev_c_vy_y[5];
		dev_c_vy_y[5] = dev_c_vy_y[6];
		dev_c_vy_y[6] = dev_c_vy_y[iGlobalTemp+yStride];

		// Update Vz values along the y-axis
		dev_c_vz_y[0] = dev_c_vz_y[1];
		dev_c_vz_y[1] = dev_c_vz_y[2];
		dev_c_vz_y[2] = dev_c_vz_y[3];
		dev_c_vz_y[3] = shared_c_vz[ixLocal][izLocal];
		shared_c_vz[ixLocal][izLocal] = dev_c_vz_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vz_y[4] = dev_c_vz_y[5];
		dev_c_vz_y[5] = dev_c_vz_y[6];
		dev_c_vz_y[6] = dev_c_vz_y[iGlobalTemp+yStride];

		// Update Sigmaxy values along the y-axis
		dev_c_sigmaxy_y[0] = dev_c_sigmaxy_y[1];
		dev_c_sigmaxy_y[1] = dev_c_sigmaxy_y[2];
		dev_c_sigmaxy_y[2] = shared_c_sigmaxy[ixLocal][izLocal];
		shared_c_sigmaxy[ixLocal][izLocal] = dev_c_sigmaxy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmaxy_y[3] = dev_c_sigmaxy_y[4];
		dev_c_sigmaxy_y[4] = dev_c_sigmaxy_y[5];
		dev_c_sigmaxy_y[5] = dev_c_sigmaxy_y[6];
		dev_c_sigmaxy_y[6] = dev_c_sigmaxy_y[iGlobalTemp+yStride];

		// Update Sigmayy values along the y-axis
		dev_c_sigmayy_y[0] = dev_c_sigmayy_y[1];
		dev_c_sigmayy_y[1] = dev_c_sigmayy_y[2];
		dev_c_sigmayy_y[2] = dev_c_sigmayy_y[3];
		dev_c_sigmayy_y[3] = shared_c_sigmayy[ixLocal][izLocal];
		shared_c_sigmayy[ixLocal][izLocal] = dev_c_sigmayy_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmayy_y[4] = dev_c_sigmayy_y[5];
		dev_c_sigmayy_y[5] = dev_c_sigmayy_y[6];
		dev_c_sigmayy_y[6] = dev_c_sigmayy_y[iGlobalTemp+yStride];

		// Update Sigmaxz values along the y-axis
		dev_c_sigmayz_y[0] = dev_c_sigmayz_y[1];
		dev_c_sigmayz_y[1] = dev_c_sigmayz_y[2];
		dev_c_sigmayz_y[2] = shared_c_sigmayz[ixLocal][izLocal];
		shared_c_sigmayz[ixLocal][izLocal] = dev_c_sigmayy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmayz_y[3] = dev_c_sigmayz_y[4];
		dev_c_sigmayz_y[4] = dev_c_sigmayz_y[5];
		dev_c_sigmayz_y[5] = dev_c_sigmayz_y[6];
		dev_c_sigmayz_y[6] = dev_c_sigmayz_y[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

		// Load the halos in the x-direction
		// Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
		if (threadIdx.y < FAT) {
			shared_c_vx[threadIdx.y][izLocal] = shared_c_vx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vy[threadIdx.y][izLocal] = shared_c_vy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vz[threadIdx.y][izLocal] = shared_c_vz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxx[threadIdx.y][izLocal] = shared_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayy[threadIdx.y][izLocal] = shared_c_sigmayy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmazz[threadIdx.y][izLocal] = shared_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxz[threadIdx.y][izLocal] = shared_c_sigmaxz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxy[threadIdx.y][izLocal] = shared_c_sigmaxy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayz[threadIdx.y][izLocal] = shared_c_sigmayz[iGlobal-dev_nz*FAT]; // Left side

			shared_c_vx[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxx[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmayy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmazz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmayz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
		}

		// Load the halos in the z-direction
		if (threadIdx.x < FAT) {
			shared_c_vx[ixLocal][threadIdx.x] = shared_c_vx[iGlobal-FAT]; // Up
			shared_c_vy[ixLocal][threadIdx.x] = shared_c_vy[iGlobal-FAT]; // Up
			shared_c_vz[ixLocal][threadIdx.x] = shared_c_vz[iGlobal-FAT]; // Up
			shared_c_sigmaxx[ixLocal][threadIdx.x] = shared_c_sigmaxx[iGlobal-FAT]; // Up
			shared_c_sigmayy[ixLocal][threadIdx.x] = shared_c_sigmayy[iGlobal-FAT]; // Up
			shared_c_sigmazz[ixLocal][threadIdx.x] = shared_c_sigmazz[iGlobal-FAT]; // Up
			shared_c_sigmaxz[ixLocal][threadIdx.x] = shared_c_sigmaxz[iGlobal-FAT]; // Up
			shared_c_sigmaxy[ixLocal][threadIdx.x] = shared_c_sigmaxy[iGlobal-FAT]; // Up
			shared_c_sigmayz[ixLocal][threadIdx.x] = shared_c_sigmayz[iGlobal-FAT]; // Up

			shared_c_vx[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxx[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmayy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmazz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmazz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmayz[iGlobal+BLOCK_SIZE_Z]; // Down
		}

		// Wait until all threads of this block have loaded the slice y-slice into shared memory
		__syncthreads(); // Synchronise all threads within each block

		// Computing common derivative terms
		// dvx/dx (+)
		double dvx_dx = dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
    								dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
    								dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
    								dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal]);
		// dvy/dx (+)
		double dvy_dy = dev_yCoeff[0]*(dev_c_vx_y[3]-shared_c_vy[ixLocal][izLocal]) +
    								dev_yCoeff[1]*(dev_c_vx_y[4]-dev_c_vx_y[2])+
    								dev_yCoeff[2]*(dev_c_vx_y[5]-dev_c_vx_y[1])+
    								dev_yCoeff[3]*(dev_c_vx_y[6]-dev_c_vx_y[0]);
		// dvz/dx (+)
		double dvz_dz = dev_zCoeff[0]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal][izLocal])  +
    								dev_zCoeff[1]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-1][izLocal])+
    								dev_zCoeff[2]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
    								dev_zCoeff[3]*(shared_c_vz[ixLocal+4][izLocal]-shared_c_vz[ixLocal-3][izLocal]);

		// Updating particle velocity fields
		// Vx
		dev_n_vx[iGlobal] = dev_o_vx[iGlobal] + dev_rhoxDtw[iGlobal] * (
													// dsigmaxx/dx (-)
													dev_xCoeff[0]*(shared_c_sigmaxx[ixLocal][izLocal]-shared_c_sigmaxx[ixLocal-1][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxx[ixLocal+1][izLocal]-shared_c_sigmaxx[ixLocal-2][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxx[ixLocal+2][izLocal]-shared_c_sigmaxx[ixLocal-3][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxx[ixLocal+3][izLocal]-shared_c_sigmaxx[ixLocal-4][izLocal]) +
													// dsigmaxy/dy (+)
													dev_yCoeff[0]*(dev_c_sigmaxy_y[3]-shared_c_sigmaxy[ixLocal][izLocal]) +
											    dev_yCoeff[1]*(dev_c_sigmaxy_y[4]-dev_c_sigmaxy_y[2])+
											    dev_yCoeff[2]*(dev_c_sigmaxy_y[5]-dev_c_sigmaxy_y[1])+
											    dev_yCoeff[3]*(dev_c_sigmaxy_y[6]-dev_c_sigmaxy_y[0])+
													// dsigmaxz/dz (+)
													dev_zCoeff[0]*(shared_c_sigmaxz[ixLocal][izLocal+1]-shared_c_sigmaxz[ixLocal][izLocal])  +
										      dev_zCoeff[1]*(shared_c_sigmaxz[ixLocal][izLocal+2]-shared_c_sigmaxz[ixLocal][izLocal-1])+
										      dev_zCoeff[2]*(shared_c_sigmaxz[ixLocal][izLocal+3]-shared_c_sigmaxz[ixLocal][izLocal-2])+
										      dev_zCoeff[3]*(shared_c_sigmaxz[ixLocal][izLocal+4]-shared_c_sigmaxz[ixLocal][izLocal-3])
												);
		// Vy
		dev_n_vy[iGlobal] = dev_o_vy[iGlobal] + dev_rhoyDtw[iGlobal] * (
													// dsigmaxy/dx (+)
													dev_xCoeff[0]*(shared_c_sigmaxy[ixLocal+1][izLocal]-shared_c_sigmaxy[ixLocal][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxy[ixLocal+2][izLocal]-shared_c_sigmaxy[ixLocal-1][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxy[ixLocal+3][izLocal]-shared_c_sigmaxy[ixLocal-2][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxy[ixLocal+4][izLocal]-shared_c_sigmaxy[ixLocal-3][izLocal]) +
													// dsigmayy/dy (-)
													dev_yCoeff[0]*(shared_c_sigmayy[ixLocal][izLocal]-dev_c_sigmayy_y[3]) +
											    dev_yCoeff[1]*(dev_c_sigmayy_y[4]-dev_c_sigmayy_y[2])+
											    dev_yCoeff[2]*(dev_c_sigmayy_y[5]-dev_c_sigmayy_y[1])+
											    dev_yCoeff[3]*(dev_c_sigmayy_y[6]-dev_c_sigmayy_y[0])+
													// dsigmayz/dz (+)
													dev_zCoeff[0]*(shared_c_sigmayz[ixLocal][izLocal+1]-shared_c_sigmayz[ixLocal][izLocal])  +
										      dev_zCoeff[1]*(shared_c_sigmayz[ixLocal][izLocal+2]-shared_c_sigmayz[ixLocal][izLocal-1])+
										      dev_zCoeff[2]*(shared_c_sigmayz[ixLocal][izLocal+3]-shared_c_sigmayz[ixLocal][izLocal-2])+
										      dev_zCoeff[3]*(shared_c_sigmayz[ixLocal][izLocal+4]-shared_c_sigmayz[ixLocal][izLocal-3])
												);
		// Vz
		dev_n_vz[iGlobal] = dev_o_vz[iGlobal] + dev_rhozDtw[iGlobal] * (
													// dsigmaxz/dx (+)
													dev_xCoeff[0]*(shared_c_sigmaxz[ixLocal+1][izLocal]-shared_c_sigmaxz[ixLocal][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxz[ixLocal+2][izLocal]-shared_c_sigmaxz[ixLocal-1][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxz[ixLocal+3][izLocal]-shared_c_sigmaxz[ixLocal-2][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxz[ixLocal+4][izLocal]-shared_c_sigmaxz[ixLocal-3][izLocal]) +
													// dsigmayz/dy (+)
													// dsigmazz/dz (-)
													dev_zCoeff[0]*(shared_c_sigmazz[ixLocal][izLocal]-shared_c_sigmazz[ixLocal][izLocal-1])  +
										      dev_zCoeff[1]*(shared_c_sigmazz[ixLocal][izLocal+1]-shared_c_sigmazz[ixLocal][izLocal-2])+
										      dev_zCoeff[2]*(shared_c_sigmazz[ixLocal][izLocal+2]-shared_c_sigmazz[ixLocal][izLocal-3])+
										      dev_zCoeff[3]*(shared_c_sigmazz[ixLocal][izLocal+3]-shared_c_sigmazz[ixLocal][izLocal-4])
												);

		// Updating stress fields
		// Sigmaxx
		dev_n_sigmaxx[iGlobal] = dev_o_sigmaxx[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvx_dx
													 + dev_lambDtw[iGlobal] * (dvy_dy + dvz_dz);
		// Sigmayy
		dev_n_sigmayy[iGlobal] = dev_o_sigmayy[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvy_dy
													 + dev_lambDtw[iGlobal] * (dvx_dx + dvz_dz);
		// Sigmazz
		dev_n_sigmazz[iGlobal] = dev_o_sigmazz[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvz_dz
													 + dev_lambDtw[iGlobal] * (dvx_dx + dvy_dy);
		// Sigmaxy
		dev_n_sigmaxy[iGlobal] = dev_o_sigmaxy[iGlobal]
													 + dev_muxyDtw[iGlobal] * (
															 // dvx_dy (-)
															 dev_yCoeff[0]*(shared_c_vx[ixLocal][izLocal]-dev_c_vx_y[3])  +
													     dev_yCoeff[1]*(dev_c_vx_y[4]-dev_c_vx_y[2])+
													     dev_yCoeff[2]*(dev_c_vx_y[5]-dev_c_vx_y[1])+
													     dev_yCoeff[3]*(dev_c_vx_y[6]-dev_c_vx_y[0])
															 // dvy_dx (-)
															 dev_xCoeff[0]*(shared_c_vy[ixLocal][izLocal]-shared_c_vy[ixLocal-1][izLocal])  +
													     dev_xCoeff[1]*(shared_c_vy[ixLocal+1][izLocal]-shared_c_vy[ixLocal-2][izLocal])+
													     dev_xCoeff[2]*(shared_c_vy[ixLocal+2][izLocal]-shared_c_vy[ixLocal-3][izLocal])+
													     dev_xCoeff[3]*(shared_c_vy[ixLocal+3][izLocal]-shared_c_vy[ixLocal-4][izLocal])
													   );
		// Sigmaxz
		dev_n_sigmaxz[iGlobal] = dev_o_sigmaxy[iGlobal]
													 + dev_muxzDtw[iGlobal] * (
															 // dvx_dz (-)
															 dev_zCoeff[0]*(shared_c_vx[ixLocal][izLocal]-shared_c_vx[ixLocal-1][izLocal])  +
													     dev_zCoeff[1]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
													     dev_zCoeff[2]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-3][izLocal])+
													     dev_zCoeff[3]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-4][izLocal])
															 // dvz_dx (-)
															 dev_xCoeff[0]*(shared_c_vz[ixLocal][izLocal]-shared_c_vz[ixLocal-1][izLocal])  +
													     dev_xCoeff[1]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
													     dev_xCoeff[2]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-3][izLocal])+
													     dev_xCoeff[3]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-4][izLocal])
													   );
		// Sigmayz
		dev_n_sigmayz[iGlobal] = dev_o_sigmaxz[iGlobal]
													 + dev_muyzDtw[iGlobal] * (
															 // dvz_dy (-)
															 dev_yCoeff[0]*(shared_c_vy[ixLocal][izLocal]-dev_c_vz_y[3])  +
													     dev_yCoeff[1]*(dev_c_vz_y[4]-dev_c_vz_y[2])+
													     dev_yCoeff[2]*(dev_c_vz_y[5]-dev_c_vz_y[1])+
													     dev_yCoeff[3]*(dev_c_vz_y[6]-dev_c_vz_y[0])
															 // dvy_dz (-)
															 dev_zCoeff[0]*(shared_c_vy[ixLocal][izLocal]-shared_c_vy[ixLocal-1][izLocal])  +
													     dev_zCoeff[1]*(shared_c_vy[ixLocal+1][izLocal]-shared_c_vy[ixLocal-2][izLocal])+
													     dev_zCoeff[2]*(shared_c_vy[ixLocal+2][izLocal]-shared_c_vy[ixLocal-3][izLocal])+
													     dev_zCoeff[3]*(shared_c_vy[ixLocal+3][izLocal]-shared_c_vy[ixLocal-4][izLocal])
													   );


		// Move forward one grid point in the y-direction
		iGlobal += yStride;

	}

}

/* Forward stepper for updating particle-velocity fields (no damping) */
__global__ void stepFwdGpu_vel_3D(double *dev_o_vx, double *dev_o_vy, double *dev_o_vz, double *dev_o_sigmaxx, double *dev_o_sigmayy, double *dev_o_sigmazz, double *dev_o_sigmaxy, double *dev_o_sigmaxz, double *dev_o_sigmayz, double *dev_c_vx, double *dev_c_vy, double *dev_c_vz, double *dev_c_sigmaxx, double *dev_c_sigmayy, double *dev_c_sigmazz, double *dev_c_sigmaxy, double *dev_c_sigmaxz, double *dev_c_sigmayz, double *dev_n_vx, double *dev_n_vy, double *dev_n_vz, double *dev_n_sigmaxx, double *dev_n_sigmayy, double *dev_n_sigmazz, double *dev_n_sigmaxy, double *dev_n_sigmaxz, double *dev_n_sigmayz, double* dev_rhoxDtw, double* dev_rhoyDtw, double* dev_rhozDtw, double* dev_lamb2MuDtw, double* dev_lambDtw, double* dev_muxzDtw, double* dev_muxyDtw, double* dev_muyzDtw){

	// Allocate shared memory for a specific block
	__shared__ double shared_c_sigmaxx[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxx wavefield y-slice block
	__shared__ double shared_c_sigmayy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmayy wavefield y-slice block
	__shared__ double shared_c_sigmazz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmazz wavefield y-slice block
	__shared__ double shared_c_sigmaxz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxz wavefield y-slice block
	__shared__ double shared_c_sigmaxy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxy wavefield y-slice block
	__shared__ double shared_c_sigmayz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmayz wavefield y-slice block

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis
	// Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

	// Allocate the arrays that will store the wavefield values in the y-direction
	// Only some components need the be used for derivative computation along the y axis
	double dev_c_sigmayy_y[2*FAT-1];
	double dev_c_sigmaxy_y[2*FAT-1];
	double dev_c_sigmayz_y[2*FAT-1];

	// Number of elements in one y-slice
	long long yStride = dev_nz * dev_nx;

	// Global index of the first element at which we are going to compute the Laplacian
	// Skip the first FAT elements on the y-axis
	long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

	// Global index of the element with the smallest y-position needed to compute derivatives at iGlobal
	long long iGlobalTemp = iGlobal - FAT * yStride;

	// Loading for Sigmaxy along the y-direction (forward derivative)
	dev_c_sigmaxy_y[1] = dev_c_sigmaxy[iGlobalTemp]; // iy = 0
	dev_c_sigmaxy_y[2] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 1
	shared_c_sigmaxy[ixLocal][izLocal] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_sigmaxy_y[3] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 3
	dev_c_sigmaxy_y[4] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmaxy_y[5] = dev_c_sigmaxy[iGlobalTemp+=yStride];// iy = 5
	dev_c_sigmaxy_y[6] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loading for Sigmayy along the y-direction (backward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_sigmayy_y[1] = dev_c_sigmayy[iGlobalTemp]; // iy = 0
	dev_c_sigmayy_y[2] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 1
	dev_c_sigmayy_y[3] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 2
	shared_c_sigmayy[ixLocal][izLocal] = dev_c_sigmayy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_sigmayy_y[4] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmayy_y[5] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 5
	dev_c_sigmayy_y[6] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 6

	// Loading for Sigmayz along the y-direction (forward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_sigmayz_y[1] = dev_c_sigmayz[iGlobalTemp]; // iy = 0
	dev_c_sigmayz_y[2] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 1
	shared_c_sigmaxz[ixLocal][izLocal] = dev_c_sigmayz[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_sigmayz_y[3] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 3
	dev_c_sigmayz_y[4] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmayz_y[5] = dev_c_sigmayz[iGlobalTemp+=yStride];// iy = 5
	dev_c_sigmayz_y[6] = dev_c_sigmayz[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loop over y
	for (long long iy=FAT; iy<dev_ny-FAT; iy++){

		// Update Sigmaxy values along the y-axis
		dev_c_sigmaxy_y[0] = dev_c_sigmaxy_y[1];
		dev_c_sigmaxy_y[1] = dev_c_sigmaxy_y[2];
		dev_c_sigmaxy_y[2] = shared_c_sigmaxy[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
		shared_c_sigmaxy[ixLocal][izLocal] = dev_c_sigmaxy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmaxy_y[3] = dev_c_sigmaxy_y[4];
		dev_c_sigmaxy_y[4] = dev_c_sigmaxy_y[5];
		dev_c_sigmaxy_y[5] = dev_c_sigmaxy_y[6];
		dev_c_sigmaxy_y[6] = dev_c_sigmaxy_y[iGlobalTemp+yStride];

		// Update Sigmayy values along the y-axis
		dev_c_sigmayy_y[0] = dev_c_sigmayy_y[1];
		dev_c_sigmayy_y[1] = dev_c_sigmayy_y[2];
		dev_c_sigmayy_y[2] = dev_c_sigmayy_y[3];
		dev_c_sigmayy_y[3] = shared_c_sigmayy[ixLocal][izLocal];
		shared_c_sigmayy[ixLocal][izLocal] = dev_c_sigmayy_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmayy_y[4] = dev_c_sigmayy_y[5];
		dev_c_sigmayy_y[5] = dev_c_sigmayy_y[6];
		dev_c_sigmayy_y[6] = dev_c_sigmayy_y[iGlobalTemp+yStride];

		// Update Sigmaxz values along the y-axis
		dev_c_sigmayz_y[0] = dev_c_sigmayz_y[1];
		dev_c_sigmayz_y[1] = dev_c_sigmayz_y[2];
		dev_c_sigmayz_y[2] = shared_c_sigmayz[ixLocal][izLocal];
		shared_c_sigmayz[ixLocal][izLocal] = dev_c_sigmayy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmayz_y[3] = dev_c_sigmayz_y[4];
		dev_c_sigmayz_y[4] = dev_c_sigmayz_y[5];
		dev_c_sigmayz_y[5] = dev_c_sigmayz_y[6];
		dev_c_sigmayz_y[6] = dev_c_sigmayz_y[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

		// Load the halos in the x-direction
		// Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
		if (threadIdx.y < FAT) {
			shared_c_sigmaxx[threadIdx.y][izLocal] = shared_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayy[threadIdx.y][izLocal] = shared_c_sigmayy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmazz[threadIdx.y][izLocal] = shared_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxz[threadIdx.y][izLocal] = shared_c_sigmaxz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxy[threadIdx.y][izLocal] = shared_c_sigmaxy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayz[threadIdx.y][izLocal] = shared_c_sigmayz[iGlobal-dev_nz*FAT]; // Left side

			shared_c_sigmaxx[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmayy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmazz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmayz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
		}

		// Load the halos in the z-direction
		if (threadIdx.x < FAT) {
			shared_c_sigmaxx[ixLocal][threadIdx.x] = shared_c_sigmaxx[iGlobal-FAT]; // Up
			shared_c_sigmayy[ixLocal][threadIdx.x] = shared_c_sigmayy[iGlobal-FAT]; // Up
			shared_c_sigmazz[ixLocal][threadIdx.x] = shared_c_sigmazz[iGlobal-FAT]; // Up
			shared_c_sigmaxz[ixLocal][threadIdx.x] = shared_c_sigmaxz[iGlobal-FAT]; // Up
			shared_c_sigmaxy[ixLocal][threadIdx.x] = shared_c_sigmaxy[iGlobal-FAT]; // Up
			shared_c_sigmayz[ixLocal][threadIdx.x] = shared_c_sigmayz[iGlobal-FAT]; // Up

			shared_c_sigmaxx[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmayy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmazz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmazz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmayz[iGlobal+BLOCK_SIZE_Z]; // Down
		}

		// Wait until all threads of this block have loaded the slice y-slice into shared memory
		__syncthreads(); // Synchronise all threads within each block

		// Updating stress fields
		// Sigmaxx
		dev_n_sigmaxx[iGlobal] = dev_o_sigmaxx[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvx_dx
													 + dev_lambDtw[iGlobal] * (dvy_dy + dvz_dz);
		// Sigmayy
		dev_n_sigmayy[iGlobal] = dev_o_sigmayy[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvy_dy
													 + dev_lambDtw[iGlobal] * (dvx_dx + dvz_dz);
		// Sigmazz
		dev_n_sigmazz[iGlobal] = dev_o_sigmazz[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvz_dz
													 + dev_lambDtw[iGlobal] * (dvx_dx + dvy_dy);
		// Sigmaxy
		dev_n_sigmaxy[iGlobal] = dev_o_sigmaxy[iGlobal]
													 + dev_muxyDtw[iGlobal] * (
															 // dvx_dy (-)
															 dev_yCoeff[0]*(shared_c_vx[ixLocal][izLocal]-dev_c_vx_y[3])  +
													     dev_yCoeff[1]*(dev_c_vx_y[4]-dev_c_vx_y[2])+
													     dev_yCoeff[2]*(dev_c_vx_y[5]-dev_c_vx_y[1])+
													     dev_yCoeff[3]*(dev_c_vx_y[6]-dev_c_vx_y[0])
															 // dvy_dx (-)
															 dev_xCoeff[0]*(shared_c_vy[ixLocal][izLocal]-shared_c_vy[ixLocal-1][izLocal])  +
													     dev_xCoeff[1]*(shared_c_vy[ixLocal+1][izLocal]-shared_c_vy[ixLocal-2][izLocal])+
													     dev_xCoeff[2]*(shared_c_vy[ixLocal+2][izLocal]-shared_c_vy[ixLocal-3][izLocal])+
													     dev_xCoeff[3]*(shared_c_vy[ixLocal+3][izLocal]-shared_c_vy[ixLocal-4][izLocal])
													   );
		// Sigmaxz
		dev_n_sigmaxz[iGlobal] = dev_o_sigmaxy[iGlobal]
													 + dev_muxzDtw[iGlobal] * (
															 // dvx_dz (-)
															 dev_zCoeff[0]*(shared_c_vx[ixLocal][izLocal]-shared_c_vx[ixLocal-1][izLocal])  +
													     dev_zCoeff[1]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
													     dev_zCoeff[2]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-3][izLocal])+
													     dev_zCoeff[3]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-4][izLocal])
															 // dvz_dx (-)
															 dev_xCoeff[0]*(shared_c_vz[ixLocal][izLocal]-shared_c_vz[ixLocal-1][izLocal])  +
													     dev_xCoeff[1]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
													     dev_xCoeff[2]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-3][izLocal])+
													     dev_xCoeff[3]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-4][izLocal])
													   );
		// Sigmayz
		dev_n_sigmayz[iGlobal] = dev_o_sigmaxz[iGlobal]
													 + dev_muyzDtw[iGlobal] * (
															 // dvz_dy (-)
															 dev_yCoeff[0]*(shared_c_vy[ixLocal][izLocal]-dev_c_vz_y[3])  +
													     dev_yCoeff[1]*(dev_c_vz_y[4]-dev_c_vz_y[2])+
													     dev_yCoeff[2]*(dev_c_vz_y[5]-dev_c_vz_y[1])+
													     dev_yCoeff[3]*(dev_c_vz_y[6]-dev_c_vz_y[0])
															 // dvy_dz (-)
															 dev_zCoeff[0]*(shared_c_vy[ixLocal][izLocal]-shared_c_vy[ixLocal-1][izLocal])  +
													     dev_zCoeff[1]*(shared_c_vy[ixLocal+1][izLocal]-shared_c_vy[ixLocal-2][izLocal])+
													     dev_zCoeff[2]*(shared_c_vy[ixLocal+2][izLocal]-shared_c_vy[ixLocal-3][izLocal])+
													     dev_zCoeff[3]*(shared_c_vy[ixLocal+3][izLocal]-shared_c_vy[ixLocal-4][izLocal])
													   );


		// Move forward one grid point in the y-direction
		iGlobal += yStride;

	}

}

/* Forward stepper for updating stress fields (no damping) */
__global__ void stepFwdGpu_stress_3D(double *dev_o_vx, double *dev_o_vy, double *dev_o_vz, double *dev_o_sigmaxx, double *dev_o_sigmayy, double *dev_o_sigmazz, double *dev_o_sigmaxy, double *dev_o_sigmaxz, double *dev_o_sigmayz, double *dev_c_vx, double *dev_c_vy, double *dev_c_vz, double *dev_c_sigmaxx, double *dev_c_sigmayy, double *dev_c_sigmazz, double *dev_c_sigmaxy, double *dev_c_sigmaxz, double *dev_c_sigmayz, double *dev_n_vx, double *dev_n_vy, double *dev_n_vz, double *dev_n_sigmaxx, double *dev_n_sigmayy, double *dev_n_sigmazz, double *dev_n_sigmaxy, double *dev_n_sigmaxz, double *dev_n_sigmayz, double* dev_rhoxDtw, double* dev_rhoyDtw, double* dev_rhozDtw, double* dev_lamb2MuDtw, double* dev_lambDtw, double* dev_muxzDtw, double* dev_muxyDtw, double* dev_muyzDtw){

	// Allocate shared memory for a specific block
	__shared__ double shared_c_vx[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vx wavefield y-slice block
	__shared__ double shared_c_vy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vy wavefield y-slice block
	__shared__ double shared_c_vz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vz wavefield y-slice block

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis
	// Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

	// Allocate the arrays that will store the wavefield values in the y-direction
	// Only some components need the be used for derivative computation along the y axis
	double dev_c_vx_y[2*FAT-1];
	double dev_c_vy_y[2*FAT-1];
	double dev_c_vz_y[2*FAT-1];

	// Number of elements in one y-slice
	long long yStride = dev_nz * dev_nx;

	// Global index of the first element at which we are going to compute the Laplacian
	// Skip the first FAT elements on the y-axis
	long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

	// Global index of the element with the smallest y-position needed to compute derivatives at iGlobal
	long long iGlobalTemp = iGlobal - FAT * yStride;

	// Loading stride for Vx along the y-direction (backward derivative)
	dev_c_vx_y[1] = dev_c_vx[iGlobalTemp]; // iy = 0
	dev_c_vx_y[2] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 1
	dev_c_vx_y[3] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 2
	shared_c_vx[ixLocal][izLocal] = dev_c_vx[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_vx_y[4] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vx_y[5] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 5
	dev_c_vx_y[6] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 6

	// Loading for Vy along the y-direction (forward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_vy_y[1] = dev_c_vy[iGlobalTemp]; // iy = 0
	dev_c_vy_y[2] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 1
	shared_c_vy[ixLocal][izLocal] = dev_c_vy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_vy_y[3] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 3
	dev_c_vy_y[4] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vy_y[5] = dev_c_vy[iGlobalTemp+=yStride];// iy = 5
	dev_c_vy_y[6] = dev_c_vy[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loading for Vz along the y-direction (backward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_vz_y[1] = dev_c_vz[iGlobalTemp]; // iy = 0
	dev_c_vz_y[2] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 1
	dev_c_vz_y[3] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 2
	shared_c_vz[ixLocal][izLocal] = dev_c_vz[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_vz_y[4] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vz_y[5] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 5
	dev_c_vz_y[6] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 6

	// Loop over y
	for (long long iy=FAT; iy<dev_ny-FAT; iy++){
		// Update Vx values along the y-axis
		dev_c_vx_y[0] = dev_c_vx_y[1];
		dev_c_vx_y[1] = dev_c_vx_y[2];
		dev_c_vx_y[2] = dev_c_vx_y[3];
		dev_c_vx_y[3] = shared_c_vx[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
		shared_c_vx[ixLocal][izLocal] = dev_c_vx_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vx_y[4] = dev_c_vx_y[5];
		dev_c_vx_y[5] = dev_c_vx_y[6];
		dev_c_vx_y[6] = dev_c_vx_y[iGlobalTemp+yStride];

		// Update Vy values along the y-axis
		dev_c_vy_y[0] = dev_c_vy_y[1];
		dev_c_vy_y[1] = dev_c_vy_y[2];
		dev_c_vy_y[2] = shared_c_vy[ixLocal][izLocal];
		shared_c_vy[ixLocal][izLocal] = dev_c_vy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vy_y[3] = dev_c_vy_y[4];
		dev_c_vy_y[4] = dev_c_vy_y[5];
		dev_c_vy_y[5] = dev_c_vy_y[6];
		dev_c_vy_y[6] = dev_c_vy_y[iGlobalTemp+yStride];

		// Update Vz values along the y-axis
		dev_c_vz_y[0] = dev_c_vz_y[1];
		dev_c_vz_y[1] = dev_c_vz_y[2];
		dev_c_vz_y[2] = dev_c_vz_y[3];
		dev_c_vz_y[3] = shared_c_vz[ixLocal][izLocal];
		shared_c_vz[ixLocal][izLocal] = dev_c_vz_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vz_y[4] = dev_c_vz_y[5];
		dev_c_vz_y[5] = dev_c_vz_y[6];
		dev_c_vz_y[6] = dev_c_vz_y[iGlobalTemp+=yStride];


		// Load the halos in the x-direction
		// Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
		if (threadIdx.y < FAT) {
			shared_c_vx[threadIdx.y][izLocal] = shared_c_vx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vy[threadIdx.y][izLocal] = shared_c_vy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vz[threadIdx.y][izLocal] = shared_c_vz[iGlobal-dev_nz*FAT]; // Left side

			shared_c_vx[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
		}

		// Load the halos in the z-direction
		if (threadIdx.x < FAT) {
			shared_c_vx[ixLocal][threadIdx.x] = shared_c_vx[iGlobal-FAT]; // Up
			shared_c_vy[ixLocal][threadIdx.x] = shared_c_vy[iGlobal-FAT]; // Up
			shared_c_vz[ixLocal][threadIdx.x] = shared_c_vz[iGlobal-FAT]; // Up

			shared_c_vx[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vz[iGlobal+BLOCK_SIZE_Z]; // Down
		}

		// Wait until all threads of this block have loaded the slice y-slice into shared memory
		__syncthreads(); // Synchronise all threads within each block

		// Computing common derivative terms
		// dvx/dx (+)
		double dvx_dx = dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
    								dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
    								dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
    								dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal]);
		// dvy/dx (+)
		double dvy_dy = dev_yCoeff[0]*(dev_c_vx_y[3]-shared_c_vy[ixLocal][izLocal]) +
    								dev_yCoeff[1]*(dev_c_vx_y[4]-dev_c_vx_y[2])+
    								dev_yCoeff[2]*(dev_c_vx_y[5]-dev_c_vx_y[1])+
    								dev_yCoeff[3]*(dev_c_vx_y[6]-dev_c_vx_y[0]);
		// dvz/dx (+)
		double dvz_dz = dev_zCoeff[0]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal][izLocal])  +
    								dev_zCoeff[1]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-1][izLocal])+
    								dev_zCoeff[2]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
    								dev_zCoeff[3]*(shared_c_vz[ixLocal+4][izLocal]-shared_c_vz[ixLocal-3][izLocal]);

		// Updating particle velocity fields
		// Vx
		dev_n_vx[iGlobal] = dev_o_vx[iGlobal] + dev_rhoxDtw[iGlobal] * (
													// dsigmaxx/dx (-)
													dev_xCoeff[0]*(shared_c_sigmaxx[ixLocal][izLocal]-shared_c_sigmaxx[ixLocal-1][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxx[ixLocal+1][izLocal]-shared_c_sigmaxx[ixLocal-2][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxx[ixLocal+2][izLocal]-shared_c_sigmaxx[ixLocal-3][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxx[ixLocal+3][izLocal]-shared_c_sigmaxx[ixLocal-4][izLocal]) +
													// dsigmaxy/dy (+)
													dev_yCoeff[0]*(dev_c_sigmaxy_y[3]-shared_c_sigmaxy[ixLocal][izLocal]) +
											    dev_yCoeff[1]*(dev_c_sigmaxy_y[4]-dev_c_sigmaxy_y[2])+
											    dev_yCoeff[2]*(dev_c_sigmaxy_y[5]-dev_c_sigmaxy_y[1])+
											    dev_yCoeff[3]*(dev_c_sigmaxy_y[6]-dev_c_sigmaxy_y[0])+
													// dsigmaxz/dz (+)
													dev_zCoeff[0]*(shared_c_sigmaxz[ixLocal][izLocal+1]-shared_c_sigmaxz[ixLocal][izLocal])  +
										      dev_zCoeff[1]*(shared_c_sigmaxz[ixLocal][izLocal+2]-shared_c_sigmaxz[ixLocal][izLocal-1])+
										      dev_zCoeff[2]*(shared_c_sigmaxz[ixLocal][izLocal+3]-shared_c_sigmaxz[ixLocal][izLocal-2])+
										      dev_zCoeff[3]*(shared_c_sigmaxz[ixLocal][izLocal+4]-shared_c_sigmaxz[ixLocal][izLocal-3])
												);
		// Vy
		dev_n_vy[iGlobal] = dev_o_vy[iGlobal] + dev_rhoyDtw[iGlobal] * (
													// dsigmaxy/dx (+)
													dev_xCoeff[0]*(shared_c_sigmaxy[ixLocal+1][izLocal]-shared_c_sigmaxy[ixLocal][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxy[ixLocal+2][izLocal]-shared_c_sigmaxy[ixLocal-1][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxy[ixLocal+3][izLocal]-shared_c_sigmaxy[ixLocal-2][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxy[ixLocal+4][izLocal]-shared_c_sigmaxy[ixLocal-3][izLocal]) +
													// dsigmayy/dy (-)
													dev_yCoeff[0]*(shared_c_sigmayy[ixLocal][izLocal]-dev_c_sigmayy_y[3]) +
											    dev_yCoeff[1]*(dev_c_sigmayy_y[4]-dev_c_sigmayy_y[2])+
											    dev_yCoeff[2]*(dev_c_sigmayy_y[5]-dev_c_sigmayy_y[1])+
											    dev_yCoeff[3]*(dev_c_sigmayy_y[6]-dev_c_sigmayy_y[0])+
													// dsigmayz/dz (+)
													dev_zCoeff[0]*(shared_c_sigmayz[ixLocal][izLocal+1]-shared_c_sigmayz[ixLocal][izLocal])  +
										      dev_zCoeff[1]*(shared_c_sigmayz[ixLocal][izLocal+2]-shared_c_sigmayz[ixLocal][izLocal-1])+
										      dev_zCoeff[2]*(shared_c_sigmayz[ixLocal][izLocal+3]-shared_c_sigmayz[ixLocal][izLocal-2])+
										      dev_zCoeff[3]*(shared_c_sigmayz[ixLocal][izLocal+4]-shared_c_sigmayz[ixLocal][izLocal-3])
												);
		// Vz
		dev_n_vz[iGlobal] = dev_o_vz[iGlobal] + dev_rhozDtw[iGlobal] * (
													// dsigmaxz/dx (+)
													dev_xCoeff[0]*(shared_c_sigmaxz[ixLocal+1][izLocal]-shared_c_sigmaxz[ixLocal][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxz[ixLocal+2][izLocal]-shared_c_sigmaxz[ixLocal-1][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxz[ixLocal+3][izLocal]-shared_c_sigmaxz[ixLocal-2][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxz[ixLocal+4][izLocal]-shared_c_sigmaxz[ixLocal-3][izLocal]) +
													// dsigmayz/dy (+)
													// dsigmazz/dz (-)
													dev_zCoeff[0]*(shared_c_sigmazz[ixLocal][izLocal]-shared_c_sigmazz[ixLocal][izLocal-1])  +
										      dev_zCoeff[1]*(shared_c_sigmazz[ixLocal][izLocal+1]-shared_c_sigmazz[ixLocal][izLocal-2])+
										      dev_zCoeff[2]*(shared_c_sigmazz[ixLocal][izLocal+2]-shared_c_sigmazz[ixLocal][izLocal-3])+
										      dev_zCoeff[3]*(shared_c_sigmazz[ixLocal][izLocal+3]-shared_c_sigmazz[ixLocal][izLocal-4])
												);

		// Move forward one grid point in the y-direction
		iGlobal += yStride;

	}

}


/* Adjoint stepper (no damping) */
__global__ void stepAdjGpu_3D(double *dev_o_vx, double *dev_o_vy, double *dev_o_vz, double *dev_o_sigmaxx, double *dev_o_sigmayy, double *dev_o_sigmazz, double *dev_o_sigmaxy, double *dev_o_sigmaxz, double *dev_o_sigmayz, double *dev_c_vx, double *dev_c_vy, double *dev_c_vz, double *dev_c_sigmaxx, double *dev_c_sigmayy, double *dev_c_sigmazz, double *dev_c_sigmaxy, double *dev_c_sigmaxz, double *dev_c_sigmayz, double *dev_n_vx, double *dev_n_vy, double *dev_n_vz, double *dev_n_sigmaxx, double *dev_n_sigmayy, double *dev_n_sigmazz, double *dev_n_sigmaxy, double *dev_n_sigmaxz, double *dev_n_sigmayz, double* dev_rhoxDtw, double* dev_rhoyDtw, double* dev_rhozDtw, double* dev_lamb2MuDtw, double* dev_lambDtw, double* dev_muxzDtw, double* dev_muxyDtw, double* dev_muyzDtw){

	// Allocate shared memory for a specific block
	__shared__ double shared_c_vx[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vx wavefield y-slice block
	__shared__ double shared_c_vy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vy wavefield y-slice block
	__shared__ double shared_c_vz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Vz wavefield y-slice block
	__shared__ double shared_c_sigmaxx[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxx wavefield y-slice block
	__shared__ double shared_c_sigmayy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmayy wavefield y-slice block
	__shared__ double shared_c_sigmazz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmazz wavefield y-slice block
	__shared__ double shared_c_sigmaxz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxz wavefield y-slice block
	__shared__ double shared_c_sigmaxy[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmaxy wavefield y-slice block
	__shared__ double shared_c_sigmayz[BLOCK_SIZE_X+2*FAT][BLOCK_SIZE_Z+2*FAT];  // Current Sigmayz wavefield y-slice block

	// Global coordinates for the faster two axes (z and x)
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Coordinate of current thread on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Coordinate of current thread on the x-axis
	// Local coordinates for the fastest two axes
	long long izLocal = FAT + threadIdx.x; // z-coordinate on the local grid stored in shared memory
	long long ixLocal = FAT + threadIdx.y; // x-coordinate on the local grid stored in shared memory

	// Allocate the arrays that will store the wavefield values in the y-direction
	// Only some components need the be used for derivative computation along the y axis
	double dev_c_vx_y[2*FAT-1];
	double dev_c_vy_y[2*FAT-1];
	double dev_c_vz_y[2*FAT-1];
	double dev_c_sigmayy_y[2*FAT-1];
	double dev_c_sigmaxy_y[2*FAT-1];
	double dev_c_sigmayz_y[2*FAT-1];

	// Number of elements in one y-slice
	long long yStride = dev_nz * dev_nx;

	// Global index of the first element at which we are going to compute the Laplacian
	// Skip the first FAT elements on the y-axis
	long long iGlobal = FAT * yStride + dev_nz * ixGlobal + izGlobal;

	// Global index of the element with the smallest y-position needed to compute derivatives at iGlobal
	long long iGlobalTemp = iGlobal - FAT * yStride;

	// Loading stride for Vx along the y-direction (backward derivative)
	dev_c_vx_y[1] = dev_c_vx[iGlobalTemp]; // iy = 0
	dev_c_vx_y[2] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 1
	dev_c_vx_y[3] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 2
	shared_c_vx[ixLocal][izLocal] = dev_c_vx[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_vx_y[4] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vx_y[5] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 5
	dev_c_vx_y[6] = dev_c_vx[iGlobalTemp+=yStride]; // iy = 6

	// Loading for Vy along the y-direction (forward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_vy_y[1] = dev_c_vy[iGlobalTemp]; // iy = 0
	dev_c_vy_y[2] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 1
	shared_c_vy[ixLocal][izLocal] = dev_c_vy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_vy_y[3] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 3
	dev_c_vy_y[4] = dev_c_vy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vy_y[5] = dev_c_vy[iGlobalTemp+=yStride];// iy = 5
	dev_c_vy_y[6] = dev_c_vy[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loading for Vz along the y-direction (backward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_vz_y[1] = dev_c_vz[iGlobalTemp]; // iy = 0
	dev_c_vz_y[2] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 1
	dev_c_vz_y[3] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 2
	shared_c_vz[ixLocal][izLocal] = dev_c_vz[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_vz_y[4] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 4
	dev_c_vz_y[5] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 5
	dev_c_vz_y[6] = dev_c_vz[iGlobalTemp+=yStride]; // iy = 6

	// Loading for Sigmaxy along the y-direction (forward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_sigmaxy_y[1] = dev_c_sigmaxy[iGlobalTemp]; // iy = 0
	dev_c_sigmaxy_y[2] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 1
	shared_c_sigmaxy[ixLocal][izLocal] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_sigmaxy_y[3] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 3
	dev_c_sigmaxy_y[4] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmaxy_y[5] = dev_c_sigmaxy[iGlobalTemp+=yStride];// iy = 5
	dev_c_sigmaxy_y[6] = dev_c_sigmaxy[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loading for Sigmayy along the y-direction (backward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_sigmayy_y[1] = dev_c_sigmayy[iGlobalTemp]; // iy = 0
	dev_c_sigmayy_y[2] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 1
	dev_c_sigmayy_y[3] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 2
	shared_c_sigmayy[ixLocal][izLocal] = dev_c_sigmayy[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 3
	dev_c_sigmayy_y[4] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmayy_y[5] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 5
	dev_c_sigmayy_y[6] = dev_c_sigmayy[iGlobalTemp+=yStride]; // iy = 6

	// Loading for Sigmayz along the y-direction (forward derivative)
	iGlobalTemp = iGlobal - FAT * yStride;
	dev_c_sigmayz_y[1] = dev_c_sigmayz[iGlobalTemp]; // iy = 0
	dev_c_sigmayz_y[2] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 1
	shared_c_sigmaxz[ixLocal][izLocal] = dev_c_sigmayz[iGlobalTemp+=yStride]; // Only the central point on the y-axis is stored in the shared memory // iy = 2
	dev_c_sigmayz_y[3] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 3
	dev_c_sigmayz_y[4] = dev_c_sigmayz[iGlobalTemp+=yStride]; // iy = 4
	dev_c_sigmayz_y[5] = dev_c_sigmayz[iGlobalTemp+=yStride];// iy = 5
	dev_c_sigmayz_y[6] = dev_c_sigmayz[iGlobalTemp+=yStride]; // At that point, iyTemp = 2*FAT-1 // iy = 6

	// Loop over y
	for (long long iy=FAT; iy<dev_ny-FAT; iy++){
		// Update Vx values along the y-axis
		dev_c_vx_y[0] = dev_c_vx_y[1];
		dev_c_vx_y[1] = dev_c_vx_y[2];
		dev_c_vx_y[2] = dev_c_vx_y[3];
		dev_c_vx_y[3] = shared_c_vx[ixLocal][izLocal];
		__syncthreads(); // Synchronise all threads within each block before updating the value of the shared memory at ixLocal, izLocal
		shared_c_vx[ixLocal][izLocal] = dev_c_vx_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vx_y[4] = dev_c_vx_y[5];
		dev_c_vx_y[5] = dev_c_vx_y[6];
		dev_c_vx_y[6] = dev_c_vx_y[iGlobalTemp+yStride];

		// Update Vy values along the y-axis
		dev_c_vy_y[0] = dev_c_vy_y[1];
		dev_c_vy_y[1] = dev_c_vy_y[2];
		dev_c_vy_y[2] = shared_c_vy[ixLocal][izLocal];
		shared_c_vy[ixLocal][izLocal] = dev_c_vy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vy_y[3] = dev_c_vy_y[4];
		dev_c_vy_y[4] = dev_c_vy_y[5];
		dev_c_vy_y[5] = dev_c_vy_y[6];
		dev_c_vy_y[6] = dev_c_vy_y[iGlobalTemp+yStride];

		// Update Vz values along the y-axis
		dev_c_vz_y[0] = dev_c_vz_y[1];
		dev_c_vz_y[1] = dev_c_vz_y[2];
		dev_c_vz_y[2] = dev_c_vz_y[3];
		dev_c_vz_y[3] = shared_c_vz[ixLocal][izLocal];
		shared_c_vz[ixLocal][izLocal] = dev_c_vz_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_vz_y[4] = dev_c_vz_y[5];
		dev_c_vz_y[5] = dev_c_vz_y[6];
		dev_c_vz_y[6] = dev_c_vz_y[iGlobalTemp+yStride];

		// Update Sigmaxy values along the y-axis
		dev_c_sigmaxy_y[0] = dev_c_sigmaxy_y[1];
		dev_c_sigmaxy_y[1] = dev_c_sigmaxy_y[2];
		dev_c_sigmaxy_y[2] = shared_c_sigmaxy[ixLocal][izLocal];
		shared_c_sigmaxy[ixLocal][izLocal] = dev_c_sigmaxy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmaxy_y[3] = dev_c_sigmaxy_y[4];
		dev_c_sigmaxy_y[4] = dev_c_sigmaxy_y[5];
		dev_c_sigmaxy_y[5] = dev_c_sigmaxy_y[6];
		dev_c_sigmaxy_y[6] = dev_c_sigmaxy_y[iGlobalTemp+yStride];

		// Update Sigmayy values along the y-axis
		dev_c_sigmayy_y[0] = dev_c_sigmayy_y[1];
		dev_c_sigmayy_y[1] = dev_c_sigmayy_y[2];
		dev_c_sigmayy_y[2] = dev_c_sigmayy_y[3];
		dev_c_sigmayy_y[3] = shared_c_sigmayy[ixLocal][izLocal];
		shared_c_sigmayy[ixLocal][izLocal] = dev_c_sigmayy_y[4]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmayy_y[4] = dev_c_sigmayy_y[5];
		dev_c_sigmayy_y[5] = dev_c_sigmayy_y[6];
		dev_c_sigmayy_y[6] = dev_c_sigmayy_y[iGlobalTemp+yStride];

		// Update Sigmaxz values along the y-axis
		dev_c_sigmayz_y[0] = dev_c_sigmayz_y[1];
		dev_c_sigmayz_y[1] = dev_c_sigmayz_y[2];
		dev_c_sigmayz_y[2] = shared_c_sigmayz[ixLocal][izLocal];
		shared_c_sigmayz[ixLocal][izLocal] = dev_c_sigmayy_y[3]; // Store the middle one in the shared memory (it will be re-used to compute the derivatives in the z- and x-directions)
		dev_c_sigmayz_y[3] = dev_c_sigmayz_y[4];
		dev_c_sigmayz_y[4] = dev_c_sigmayz_y[5];
		dev_c_sigmayz_y[5] = dev_c_sigmayz_y[6];
		dev_c_sigmayz_y[6] = dev_c_sigmayz_y[iGlobalTemp+=yStride]; // The last point of the stencil now points to the next y-slice

		// Load the halos in the x-direction
		// Threads with x-index ranging from 0,...,FAT will load the first and last FAT elements of the block on the x-axis to shared memory
		if (threadIdx.y < FAT) {
			shared_c_vx[threadIdx.y][izLocal] = shared_c_vx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vy[threadIdx.y][izLocal] = shared_c_vy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vz[threadIdx.y][izLocal] = shared_c_vz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxx[threadIdx.y][izLocal] = shared_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayy[threadIdx.y][izLocal] = shared_c_sigmayy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmazz[threadIdx.y][izLocal] = shared_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxz[threadIdx.y][izLocal] = shared_c_sigmaxz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxy[threadIdx.y][izLocal] = shared_c_sigmaxy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayz[threadIdx.y][izLocal] = shared_c_sigmayz[iGlobal-dev_nz*FAT]; // Left side

			shared_c_vx[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_vz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxx[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmayy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmazz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxy[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmaxy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayz[ixLocal+BLOCK_SIZE_X][izLocal] = shared_c_sigmayz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
		}

		// Load the halos in the z-direction
		if (threadIdx.x < FAT) {
			shared_c_vx[ixLocal][threadIdx.x] = shared_c_vx[iGlobal-FAT]; // Up
			shared_c_vy[ixLocal][threadIdx.x] = shared_c_vy[iGlobal-FAT]; // Up
			shared_c_vz[ixLocal][threadIdx.x] = shared_c_vz[iGlobal-FAT]; // Up
			shared_c_sigmaxx[ixLocal][threadIdx.x] = shared_c_sigmaxx[iGlobal-FAT]; // Up
			shared_c_sigmayy[ixLocal][threadIdx.x] = shared_c_sigmayy[iGlobal-FAT]; // Up
			shared_c_sigmazz[ixLocal][threadIdx.x] = shared_c_sigmazz[iGlobal-FAT]; // Up
			shared_c_sigmaxz[ixLocal][threadIdx.x] = shared_c_sigmaxz[iGlobal-FAT]; // Up
			shared_c_sigmaxy[ixLocal][threadIdx.x] = shared_c_sigmaxy[iGlobal-FAT]; // Up
			shared_c_sigmayz[ixLocal][threadIdx.x] = shared_c_sigmayz[iGlobal-FAT]; // Up

			shared_c_vx[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_vz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxx[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmayy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmazz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmazz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxy[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmaxy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayz[ixLocal][izLocal+BLOCK_SIZE_Z] = shared_c_sigmayz[iGlobal+BLOCK_SIZE_Z]; // Down
		}

		// Wait until all threads of this block have loaded the slice y-slice into shared memory
		__syncthreads(); // Synchronise all threads within each block

		// Computing common derivative terms
		// dvx/dx (+)
		double dvx_dx = dev_xCoeff[0]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal][izLocal])  +
    								dev_xCoeff[1]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-1][izLocal])+
    								dev_xCoeff[2]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
    								dev_xCoeff[3]*(shared_c_vx[ixLocal+4][izLocal]-shared_c_vx[ixLocal-3][izLocal]);
		// dvy/dx (+)
		double dvy_dy = dev_yCoeff[0]*(dev_c_vx_y[3]-shared_c_vy[ixLocal][izLocal]) +
    								dev_yCoeff[1]*(dev_c_vx_y[4]-dev_c_vx_y[2])+
    								dev_yCoeff[2]*(dev_c_vx_y[5]-dev_c_vx_y[1])+
    								dev_yCoeff[3]*(dev_c_vx_y[6]-dev_c_vx_y[0]);
		// dvz/dx (+)
		double dvz_dz = dev_zCoeff[0]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal][izLocal])  +
    								dev_zCoeff[1]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-1][izLocal])+
    								dev_zCoeff[2]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
    								dev_zCoeff[3]*(shared_c_vz[ixLocal+4][izLocal]-shared_c_vz[ixLocal-3][izLocal]);

		// Updating particle velocity fields
		// Vx
		dev_n_vx[iGlobal] = dev_o_vx[iGlobal] + dev_rhoxDtw[iGlobal] * (
													// dsigmaxx/dx (-)
													dev_xCoeff[0]*(shared_c_sigmaxx[ixLocal][izLocal]-shared_c_sigmaxx[ixLocal-1][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxx[ixLocal+1][izLocal]-shared_c_sigmaxx[ixLocal-2][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxx[ixLocal+2][izLocal]-shared_c_sigmaxx[ixLocal-3][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxx[ixLocal+3][izLocal]-shared_c_sigmaxx[ixLocal-4][izLocal]) +
													// dsigmaxy/dy (+)
													dev_yCoeff[0]*(dev_c_sigmaxy_y[3]-shared_c_sigmaxy[ixLocal][izLocal]) +
											    dev_yCoeff[1]*(dev_c_sigmaxy_y[4]-dev_c_sigmaxy_y[2])+
											    dev_yCoeff[2]*(dev_c_sigmaxy_y[5]-dev_c_sigmaxy_y[1])+
											    dev_yCoeff[3]*(dev_c_sigmaxy_y[6]-dev_c_sigmaxy_y[0])+
													// dsigmaxz/dz (+)
													dev_zCoeff[0]*(shared_c_sigmaxz[ixLocal][izLocal+1]-shared_c_sigmaxz[ixLocal][izLocal])  +
										      dev_zCoeff[1]*(shared_c_sigmaxz[ixLocal][izLocal+2]-shared_c_sigmaxz[ixLocal][izLocal-1])+
										      dev_zCoeff[2]*(shared_c_sigmaxz[ixLocal][izLocal+3]-shared_c_sigmaxz[ixLocal][izLocal-2])+
										      dev_zCoeff[3]*(shared_c_sigmaxz[ixLocal][izLocal+4]-shared_c_sigmaxz[ixLocal][izLocal-3])
												);
		// Vy
		dev_n_vy[iGlobal] = dev_o_vy[iGlobal] + dev_rhoyDtw[iGlobal] * (
													// dsigmaxy/dx (+)
													dev_xCoeff[0]*(shared_c_sigmaxy[ixLocal+1][izLocal]-shared_c_sigmaxy[ixLocal][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxy[ixLocal+2][izLocal]-shared_c_sigmaxy[ixLocal-1][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxy[ixLocal+3][izLocal]-shared_c_sigmaxy[ixLocal-2][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxy[ixLocal+4][izLocal]-shared_c_sigmaxy[ixLocal-3][izLocal]) +
													// dsigmayy/dy (-)
													dev_yCoeff[0]*(shared_c_sigmayy[ixLocal][izLocal]-dev_c_sigmayy_y[3]) +
											    dev_yCoeff[1]*(dev_c_sigmayy_y[4]-dev_c_sigmayy_y[2])+
											    dev_yCoeff[2]*(dev_c_sigmayy_y[5]-dev_c_sigmayy_y[1])+
											    dev_yCoeff[3]*(dev_c_sigmayy_y[6]-dev_c_sigmayy_y[0])+
													// dsigmayz/dz (+)
													dev_zCoeff[0]*(shared_c_sigmayz[ixLocal][izLocal+1]-shared_c_sigmayz[ixLocal][izLocal])  +
										      dev_zCoeff[1]*(shared_c_sigmayz[ixLocal][izLocal+2]-shared_c_sigmayz[ixLocal][izLocal-1])+
										      dev_zCoeff[2]*(shared_c_sigmayz[ixLocal][izLocal+3]-shared_c_sigmayz[ixLocal][izLocal-2])+
										      dev_zCoeff[3]*(shared_c_sigmayz[ixLocal][izLocal+4]-shared_c_sigmayz[ixLocal][izLocal-3])
												);
		// Vz
		dev_n_vz[iGlobal] = dev_o_vz[iGlobal] + dev_rhozDtw[iGlobal] * (
													// dsigmaxz/dx (+)
													dev_xCoeff[0]*(shared_c_sigmaxz[ixLocal+1][izLocal]-shared_c_sigmaxz[ixLocal][izLocal])+
										      dev_xCoeff[1]*(shared_c_sigmaxz[ixLocal+2][izLocal]-shared_c_sigmaxz[ixLocal-1][izLocal])+
										      dev_xCoeff[2]*(shared_c_sigmaxz[ixLocal+3][izLocal]-shared_c_sigmaxz[ixLocal-2][izLocal])+
										      dev_xCoeff[3]*(shared_c_sigmaxz[ixLocal+4][izLocal]-shared_c_sigmaxz[ixLocal-3][izLocal]) +
													// dsigmayz/dy (+)
													// dsigmazz/dz (-)
													dev_zCoeff[0]*(shared_c_sigmazz[ixLocal][izLocal]-shared_c_sigmazz[ixLocal][izLocal-1])  +
										      dev_zCoeff[1]*(shared_c_sigmazz[ixLocal][izLocal+1]-shared_c_sigmazz[ixLocal][izLocal-2])+
										      dev_zCoeff[2]*(shared_c_sigmazz[ixLocal][izLocal+2]-shared_c_sigmazz[ixLocal][izLocal-3])+
										      dev_zCoeff[3]*(shared_c_sigmazz[ixLocal][izLocal+3]-shared_c_sigmazz[ixLocal][izLocal-4])
												);

		// Updating stress fields
		// Sigmaxx
		dev_n_sigmaxx[iGlobal] = dev_o_sigmaxx[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvx_dx
													 + dev_lambDtw[iGlobal] * (dvy_dy + dvz_dz);
		// Sigmayy
		dev_n_sigmayy[iGlobal] = dev_o_sigmayy[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvy_dy
													 + dev_lambDtw[iGlobal] * (dvx_dx + dvz_dz);
		// Sigmazz
		dev_n_sigmazz[iGlobal] = dev_o_sigmazz[iGlobal]
													 + dev_lamb2MuDtw[iGlobal] * dvz_dz
													 + dev_lambDtw[iGlobal] * (dvx_dx + dvy_dy);
		// Sigmaxy
		dev_n_sigmaxy[iGlobal] = dev_o_sigmaxy[iGlobal]
													 + dev_muxyDtw[iGlobal] * (
															 // dvx_dy (-)
															 dev_yCoeff[0]*(shared_c_vx[ixLocal][izLocal]-dev_c_vx_y[3])  +
													     dev_yCoeff[1]*(dev_c_vx_y[4]-dev_c_vx_y[2])+
													     dev_yCoeff[2]*(dev_c_vx_y[5]-dev_c_vx_y[1])+
													     dev_yCoeff[3]*(dev_c_vx_y[6]-dev_c_vx_y[0])
															 // dvy_dx (-)
															 dev_xCoeff[0]*(shared_c_vy[ixLocal][izLocal]-shared_c_vy[ixLocal-1][izLocal])  +
													     dev_xCoeff[1]*(shared_c_vy[ixLocal+1][izLocal]-shared_c_vy[ixLocal-2][izLocal])+
													     dev_xCoeff[2]*(shared_c_vy[ixLocal+2][izLocal]-shared_c_vy[ixLocal-3][izLocal])+
													     dev_xCoeff[3]*(shared_c_vy[ixLocal+3][izLocal]-shared_c_vy[ixLocal-4][izLocal])
													   );
		// Sigmaxz
		dev_n_sigmaxz[iGlobal] = dev_o_sigmaxy[iGlobal]
													 + dev_muxzDtw[iGlobal] * (
															 // dvx_dz (-)
															 dev_zCoeff[0]*(shared_c_vx[ixLocal][izLocal]-shared_c_vx[ixLocal-1][izLocal])  +
													     dev_zCoeff[1]*(shared_c_vx[ixLocal+1][izLocal]-shared_c_vx[ixLocal-2][izLocal])+
													     dev_zCoeff[2]*(shared_c_vx[ixLocal+2][izLocal]-shared_c_vx[ixLocal-3][izLocal])+
													     dev_zCoeff[3]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-4][izLocal])
															 // dvz_dx (-)
															 dev_xCoeff[0]*(shared_c_vz[ixLocal][izLocal]-shared_c_vz[ixLocal-1][izLocal])  +
													     dev_xCoeff[1]*(shared_c_vz[ixLocal+1][izLocal]-shared_c_vz[ixLocal-2][izLocal])+
													     dev_xCoeff[2]*(shared_c_vz[ixLocal+2][izLocal]-shared_c_vz[ixLocal-3][izLocal])+
													     dev_xCoeff[3]*(shared_c_vz[ixLocal+3][izLocal]-shared_c_vz[ixLocal-4][izLocal])
													   );
		// Sigmayz
		dev_n_sigmayz[iGlobal] = dev_o_sigmaxz[iGlobal]
													 + dev_muyzDtw[iGlobal] * (
															 // dvz_dy (-)
															 dev_yCoeff[0]*(shared_c_vy[ixLocal][izLocal]-dev_c_vz_y[3])  +
													     dev_yCoeff[1]*(dev_c_vz_y[4]-dev_c_vz_y[2])+
													     dev_yCoeff[2]*(dev_c_vz_y[5]-dev_c_vz_y[1])+
													     dev_yCoeff[3]*(dev_c_vz_y[6]-dev_c_vz_y[0])
															 // dvy_dz (-)
															 dev_zCoeff[0]*(shared_c_vy[ixLocal][izLocal]-shared_c_vy[ixLocal-1][izLocal])  +
													     dev_zCoeff[1]*(shared_c_vy[ixLocal+1][izLocal]-shared_c_vy[ixLocal-2][izLocal])+
													     dev_zCoeff[2]*(shared_c_vy[ixLocal+2][izLocal]-shared_c_vy[ixLocal-3][izLocal])+
													     dev_zCoeff[3]*(shared_c_vy[ixLocal+3][izLocal]-shared_c_vy[ixLocal-4][izLocal])
													   );


		// Move forward one grid point in the y-direction
		iGlobal += yStride;

	}

}
