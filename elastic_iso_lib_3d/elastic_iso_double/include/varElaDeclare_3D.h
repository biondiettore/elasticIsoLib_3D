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
	__constant__ int dev_nTimeInterpFilter; // Time interpolation filter length
	__constant__ int dev_hTimeInterpFilter; // Time interpolation filter half-length
	__constant__ double dev_timeInterpFilter[2*(SUB_MAX+1)]; // Time interpolation filter stored in constant memory

	__constant__ int dev_nts; // Number of time steps at the coarse time sampling on Device
	__constant__ int dev_ntw; // Number of time steps at the fine time sampling on Device
	__constant__ int dev_sub; // Subsampling in time
	__constant__ double dev_dts_inv; // 1/dts for computing time derivative on device
	__constant__ double dev_dtw; // dtw

	__constant__ double dev_alphaCos; // Decay coefficient
	__constant__ int dev_minPad; // Minimum padding length
	__constant__ double dev_cosDampingCoeff[PAD_MAX]; // Padding array


	// Global memory variables
	long long **dev_sourcesPositionRegCenterGrid, **dev_sourcesPositionRegXGrid, **dev_sourcesPositionRegYGrid, **dev_sourcesPositionRegZGrid, **dev_sourcesPositionRegXZGrid, **dev_sourcesPositionRegXYGrid, **dev_sourcesPositionRegYZGrid; // Array containing the positions of the sources on the regular grid
	long long **dev_receiversPositionRegCenterGrid, **dev_receiversPositionRegXGrid, **dev_receiversPositionRegYGrid, **dev_receiversPositionRegZGrid, **dev_receiversPositionRegXZGrid, **dev_receiversPositionRegXYGrid, **dev_receiversPositionRegYZGrid; // Array containing the positions of the receivers on the regular grid
	double **dev_p0_vx, **dev_p0_vy, **dev_p0_vz, **dev_p0_sigmaxx, **dev_p0_sigmayy, **dev_p0_sigmazz, **dev_p0_sigmaxz, **dev_p0_sigmaxy, **dev_p0_sigmayz; // Temporary slices for stepping
	double **dev_p1_vx, **dev_p1_vy, **dev_p1_vz, **dev_p1_sigmaxx, **dev_p1_sigmayy, **dev_p1_sigmazz, **dev_p1_sigmaxz, **dev_p1_sigmaxy, **dev_p1_sigmayz; // Temporary slices for stepping
	double **dev_temp1; // Temporary slices for stepping

	double **dev_modelRegDts_vx, **dev_modelRegDts_vy, **dev_modelRegDts_vz, **dev_modelRegDts_sigmaxx, **dev_modelRegDts_sigmayy, **dev_modelRegDts_sigmazz, **dev_modelRegDts_sigmaxz, **dev_modelRegDts_sigmaxy, **dev_modelRegDts_sigmayz; // Model for nonlinear propagation (wavelet)
	double **dev_dataRegDts_vx, **dev_dataRegDts_vy, **dev_dataRegDts_vz, **dev_dataRegDts_sigmaxx, **dev_dataRegDts_sigmayy, **dev_dataRegDts_sigmazz, **dev_dataRegDts_sigmaxz, **dev_dataRegDts_sigmaxy, **dev_dataRegDts_sigmayz; // Data on device at coarse time-sampling (converted to regular grid)

	double **dev_rhoxDtw, **dev_rhoyDtw, **dev_rhozDtw, **dev_lamb2MuDtw, **dev_lambDtw, **dev_muxzDtw, **dev_muxyDtw, **dev_muyzDtw; // Precomputed scaled properties

	/************************************* HOST DECLARATION *********************************/
	double host_dx;
	double host_dy;
	double host_dz;

	int host_nts;
	double host_dts;
	int host_ntw;
	int host_sub;
	int host_minPad;



#endif

#endif
