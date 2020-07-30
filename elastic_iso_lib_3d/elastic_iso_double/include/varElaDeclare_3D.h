#ifndef VAR_DECLARE_3D_H
#define VAR_DECLARE_3D_H 1

#include <math.h>
#define BLOCK_SIZE_Z 16
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE 16
#define BLOCK_SIZE_DATA 128
#define FAT 4

#define PI_CUDA M_PI // Import the number "Pi" from the math library
#define PAD_MAX 400 // Maximum number of points for padding (on one side)
#define SUB_MAX 100 // Maximum subsampling value for time
#define N_GPU_MAX 16 // Maximum number of GPUs allowed to run in parallel

#define min2(v1,v2) (((v1)<(v2))?(v1):(v2)) /* Minimum function */
#define max2(v1,v2) (((v1)>(v2))?(v1):(v2)) /* Minimum function */

#define COEFF_SIZE 4 // Derivative coefficient array for 8th order


#if __CUDACC__
	/************************************* DEVICE DECLARATION *******************************/
	// Device function
	__device__ int min4(int v1,int v2,int v3,int v4){return min2(min2(v1,v2),min2(v3,v4));}

	// Constant memory variables
	// 8th-order derivative coefficients on Device
	__constant__ double dev_xCoeff[COEFF_SIZE];
	__constant__ double dev_yCoeff[COEFF_SIZE];
	__constant__ double dev_zCoeff[COEFF_SIZE];

	// Constant memory
	__constant__ double dev_coeff[COEFF_SIZE];
	__constant__ int dev_nTimeInterpFilter; // Time interpolation filter length
	__constant__ int dev_hTimeInterpFilter; // Time interpolation filter half-length
	__constant__ double dev_timeInterpFilter[2*(SUB_MAX+1)]; // Time interpolation filter stored in constant memory

	__constant__ int dev_nx; // nx on Device
	__constant__ int dev_ny; // ny on Device
	__constant__ int dev_nz; // nz on Device
	__constant__ long long dev_yStride; // nz * nx on Device
	__constant__ unsigned long long dev_nModel; // nz * nx * ny on Device

	__constant__ int dev_nts; // Number of time steps at the coarse time sampling on Device
	__constant__ int dev_ntw; // Number of time steps at the fine time sampling on Device
	__constant__ int dev_sub; // Subsampling in time
	__constant__ double dev_dts_inv; // 1/dts for computing time derivative on device
	__constant__ double dev_dtw; // dtw

	/************************************* HOST DECLARATION *********************************/
	long long host_nx; // Includes padding + FAT
	long long host_ny;
	long long host_nz;
	long long host_yStride;
	unsigned long long host_nModel;

	double host_dx;
	double host_dy;
	double host_dz;

	int host_nts;
	double host_dts;
	int host_ntw;
	int host_sub;



#endif

#endif
