#include "varElaDeclare_3D.h"
#include <stdio.h>

/******************************************************************************/
/*********************************** Injection ********************************/
/******************************************************************************/

/*															SOURCE INJECTION      												*/
// NORMAL STRESSES
/* Interpolate and inject source on center grid */
__global__ void ker_inject_source_centerGrid_3D(double *dev_signalIn_sigmaxx, double *dev_signalIn_sigmayy, double *dev_signalIn_sigmazz, double *dev_timeSlice_sigmaxx, double *dev_timeSlice_sigmayy, double *dev_timeSlice_sigmazz, long long its, long long it2, long long *dev_sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegCenterGrid) {
			dev_timeSlice_sigmaxx[dev_sourcesPositionRegCenterGrid[iThread]] += dev_signalIn_sigmaxx[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmaxx[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
			dev_timeSlice_sigmayy[dev_sourcesPositionRegCenterGrid[iThread]] += dev_signalIn_sigmayy[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmayy[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
			dev_timeSlice_sigmazz[dev_sourcesPositionRegCenterGrid[iThread]] += dev_signalIn_sigmazz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmazz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}

// PARTICLE VELOCITIES
/* Interpolate and inject source on x shifted grid */
__global__ void ker_inject_source_xGrid_3D(double *dev_signalIn_vx, double *dev_timeSlice_vx, long long its, long long it2, long long *dev_sourcesPositionRegXGrid, long long nSourcesRegXGrid){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegXGrid) {
				dev_timeSlice_vx[dev_sourcesPositionRegXGrid[iThread]] += dev_signalIn_vx[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_vx[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
/* Interpolate and inject source on y shifted grid */
__global__ void ker_inject_source_yGrid_3D(double *dev_signalIn_vy, double *dev_timeSlice_vy, long long its, long long it2, long long *dev_sourcesPositionRegYGrid, long long nSourcesRegYGrid){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegYGrid) {
				dev_timeSlice_vy[dev_sourcesPositionRegYGrid[iThread]] += dev_signalIn_vy[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_vy[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
/* Interpolate and inject source on z shifted grid */
__global__ void ker_inject_source_zGrid_3D(double *dev_signalIn_vz, double *dev_timeSlice_vz, long long its, long long it2, long long *dev_sourcesPositionRegZGrid, long long nSourcesRegZGrid){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegZGrid) {
				dev_timeSlice_vz[dev_sourcesPositionRegZGrid[iThread]] += dev_signalIn_vz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_vz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}

// SHEAR STRESSES
/* Interpolate and inject source on xz shifted grid */
__global__ void ker_inject_source_xzGrid_3D(double *dev_signalIn_sigmaxz, double *dev_timeSlice_sigmaxz, long long its, long long it2, long long *dev_sourcesPositionRegXZGrid, long long nSourcesRegXZGrid){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegXZGrid) {
				dev_timeSlice_sigmaxz[dev_sourcesPositionRegXZGrid[iThread]] += dev_signalIn_sigmaxz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmaxz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
/* Interpolate and inject source on xy shifted grid */
__global__ void ker_inject_source_xyGrid_3D(double *dev_signalIn_sigmaxy, double *dev_timeSlice_sigmaxy, long long its, long long it2, long long *dev_sourcesPositionRegXYGrid, long long nSourcesRegXYGrid){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegXYGrid) {
				dev_timeSlice_sigmaxy[dev_sourcesPositionRegXYGrid[iThread]] += dev_signalIn_sigmaxy[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmaxy[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
/* Interpolate and inject source on yz shifted grid */
__global__ void ker_inject_source_yzGrid_3D(double *dev_signalIn_sigmayz, double *dev_timeSlice_sigmayz, long long its, long long it2, long long *dev_sourcesPositionRegYZGrid, long long nSourcesRegYZGrid){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegYZGrid) {
				dev_timeSlice_sigmayz[dev_sourcesPositionRegYZGrid[iThread]] += dev_signalIn_sigmayz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmayz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}


/*															DATA INJECTION      													*/
// NORMAL STRESSES
/* Interpolate and inject data on center grid */
__global__ void ker_inject_data_centerGrid_3D(double *dev_signalIn_sigmaxx, double *dev_signalIn_sigmayy, double *dev_signalIn_sigmazz, double *dev_timeSlice_sigmaxx, double *dev_timeSlice_sigmayy, double *dev_timeSlice_sigmazz, long long its, long long it2, long long *dev_receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegCenterGrid) {
			dev_timeSlice_sigmaxx[dev_receiversPositionRegCenterGrid[iThread]] += dev_signalIn_sigmaxx[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmaxx[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
			dev_timeSlice_sigmayy[dev_receiversPositionRegCenterGrid[iThread]] += dev_signalIn_sigmayy[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmayy[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
			dev_timeSlice_sigmazz[dev_receiversPositionRegCenterGrid[iThread]] += dev_signalIn_sigmazz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmazz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}

// PARTICLE VELOCITIES
/* Interpolate and inject data on x shifted grid */
__global__ void ker_inject_data_xGrid_3D(double *dev_signalIn_vx, double *dev_timeSlice_vx, long long its, long long it2, long long *dev_receiversPositionRegXGrid, long long nReceiversRegXGrid){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegXGrid) {
				dev_timeSlice_vx[dev_receiversPositionRegXGrid[iThread]] += dev_signalIn_vx[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_vx[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
/* Interpolate and inject data on y shifted grid */
__global__ void ker_inject_data_yGrid_3D(double *dev_signalIn_vy, double *dev_timeSlice_vy, long long its, long long it2, long long *dev_receiversPositionRegYGrid, long long nReceiversRegYGrid){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegYGrid) {
				dev_timeSlice_vy[dev_receiversPositionRegYGrid[iThread]] += dev_signalIn_vy[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_vy[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
/* Interpolate and inject data on z shifted grid */
__global__ void ker_inject_data_zGrid_3D(double *dev_signalIn_vz, double *dev_timeSlice_vz, long long its, long long it2, long long *dev_receiversPositionRegZGrid, long long nReceiversRegZGrid){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegZGrid) {
				dev_timeSlice_vz[dev_receiversPositionRegZGrid[iThread]] += dev_signalIn_vz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_vz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}

// SHEAR STRESSES
/* Interpolate and inject data on xz shifted grid */
__global__ void ker_inject_data_xzGrid_3D(double *dev_signalIn_sigmaxz, double *dev_timeSlice_sigmaxz, long long its, long long it2, long long *dev_receiversPositionRegXZGrid, long long nReceiversRegXZGrid){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegXZGrid) {
				dev_timeSlice_sigmaxz[dev_receiversPositionRegXZGrid[iThread]] += dev_signalIn_sigmaxz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmaxz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
/* Interpolate and inject data on xy shifted grid */
__global__ void ker_inject_data_xyGrid_3D(double *dev_signalIn_sigmaxy, double *dev_timeSlice_sigmaxy, long long its, long long it2, long long *dev_receiversPositionRegXYGrid, long long nReceiversRegXYGrid){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegXYGrid) {
				dev_timeSlice_sigmaxy[dev_receiversPositionRegXYGrid[iThread]] += dev_signalIn_sigmaxy[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmaxy[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
/* Interpolate and inject data on yz shifted grid */
__global__ void ker_inject_data_yzGrid_3D(double *dev_signalIn_sigmayz, double *dev_timeSlice_sigmayz, long long its, long long it2, long long *dev_receiversPositionRegYZGrid, long long nReceiversRegYZGrid){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegYZGrid) {
				dev_timeSlice_sigmayz[dev_receiversPositionRegYZGrid[iThread]] += dev_signalIn_sigmayz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmayz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}

/******************************************************************************/
/******************************* Extraction ***********************************/
/******************************************************************************/

/*															SOURCE EXTRACTION      												*/
// NORMAL STRESSES
/*extract and interpolate source thar are on center grid */
__global__ void ker_record_source_centerGrid_3D(double *dev_newTimeSlice_sigmaxx, double *dev_newTimeSlice_sigmazz, double *dev_signalOut_sigmaxx, double *dev_signalOut_sigmazz, long long its, long long it2, long long *dev_sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid) {
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegCenterGrid){
			dev_signalOut_sigmaxx[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxx[dev_sourcesPositionRegCenterGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmaxx[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxx[dev_sourcesPositionRegCenterGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
	    dev_signalOut_sigmazz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmazz[dev_sourcesPositionRegCenterGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmazz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmazz[dev_sourcesPositionRegCenterGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}

// PARTICLE VELOCITIES
/*extract and interpolate source tha are on x shifted grid */
__global__ void ker_record_source_xGrid_3D(double *dev_newTimeSlice_vx, double *dev_signalOut_vx, long long its, long long it2, long long *dev_sourcesPositionRegXGrid, long long nSourcesRegXGrid) {
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegXGrid){
			dev_signalOut_vx[dev_nts*iThread+its]   += dev_newTimeSlice_vx[dev_sourcesPositionRegXGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_vx[dev_nts*iThread+its+1] += dev_newTimeSlice_vx[dev_sourcesPositionRegXGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}
/*extract and interpolate source tha are on z shifted grid */
__global__ void ker_record_source_zGrid_3D(double *dev_newTimeSlice_vz, double *dev_signalOut_vz, long long its, long long it2, long long *dev_sourcesPositionRegZGrid, long long nSourcesRegZGrid) {
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegZGrid){
			dev_signalOut_vz[dev_nts*iThread+its]   += dev_newTimeSlice_vz[dev_sourcesPositionRegZGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_vz[dev_nts*iThread+its+1] += dev_newTimeSlice_vz[dev_sourcesPositionRegZGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}

// SHEAR STRESSES
/*extract and interpolate source tha are on xz shifted grid */
__global__ void ker_record_source_xzGrid_3D(double *dev_newTimeSlice_sigmaxz, double *dev_signalOut_sigmaxz, long long its, long long it2, long long *dev_sourcesPositionRegXZGrid, long long nSourcesRegXZGrid) {
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegXZGrid){
			dev_signalOut_sigmaxz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxz[dev_sourcesPositionRegXZGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmaxz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxz[dev_sourcesPositionRegXZGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}
/*extract and interpolate source tha are on xy shifted grid */
__global__ void ker_record_source_xyGrid_3D(double *dev_newTimeSlice_sigmaxy, double *dev_signalOut_sigmaxy, long long its, long long it2, long long *dev_sourcesPositionRegXYGrid, long long nSourcesRegXYGrid) {
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegXYGrid){
			dev_signalOut_sigmaxy[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxy[dev_sourcesPositionRegXYGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmaxy[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxy[dev_sourcesPositionRegXYGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}
/*extract and interpolate source tha are on yz shifted grid */
__global__ void ker_record_source_yzGrid_3D(double *dev_newTimeSlice_sigmayz, double *dev_signalOut_sigmayz, long long its, long long it2, long long *dev_sourcesPositionRegYZGrid, long long nSourcesRegYZGrid) {
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegYZGrid){
			dev_signalOut_sigmayz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmayz[dev_sourcesPositionRegYZGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmayz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmayz[dev_sourcesPositionRegYZGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}


/*															 DATA EXTRACTION      												*/
// NORMAL STRESSES
/* Extract and interpolate data on center grid */
__global__ void ker_record_interp_data_centerGrid_3D(double *dev_newTimeSlice_sigmaxx, double *dev_newTimeSlice_sigmazz, double *dev_signalOut_sigmaxx, double *dev_signalOut_sigmazz, long long its, long long it2, long long *dev_receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid) {

    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
    if (iThread < nReceiversRegCenterGrid) {
	    dev_signalOut_sigmaxx[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxx[dev_receiversPositionRegCenterGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmaxx[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxx[dev_receiversPositionRegCenterGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
	    dev_signalOut_sigmazz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmazz[dev_receiversPositionRegCenterGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmazz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmazz[dev_receiversPositionRegCenterGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    }
}

// PARTICLE VELOCITIES
/* Extract and interpolate data on x shifted grid */
__global__ void ker_record_interp_data_xGrid_3D(double *dev_newTimeSlice_vx, double *dev_signalOut_vx, long long its, long long it2, long long *dev_receiversPositionRegXGrid, long long nReceiversRegXGrid) {

    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
    if (iThread < nReceiversRegXGrid) {
	    dev_signalOut_vx[dev_nts*iThread+its]   += dev_newTimeSlice_vx[dev_receiversPositionRegXGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_vx[dev_nts*iThread+its+1] += dev_newTimeSlice_vx[dev_receiversPositionRegXGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    }
}
/* Extract and interpolate data on z shifted grid */
__global__ void ker_record_interp_data_zGrid_3D(double *dev_newTimeSlice_vz, double *dev_signalOut_vz, long long its, long long it2, long long *dev_receiversPositionRegZGrid, long long nReceiversRegZGrid) {

    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
    if (iThread < nReceiversRegZGrid) {
	    dev_signalOut_vz[dev_nts*iThread+its]   += dev_newTimeSlice_vz[dev_receiversPositionRegZGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_vz[dev_nts*iThread+its+1] += dev_newTimeSlice_vz[dev_receiversPositionRegZGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    }
}


// SHEAR STRESSES
/* Extract and interpolate data on xz shifted grid */
__global__ void ker_record_interp_data_xzGrid_3D(double *dev_newTimeSlice_sigmaxz, double *dev_signalOut_sigmaxz, long long its, long long it2, long long *dev_receiversPositionRegXZGrid, long long nReceiversRegXZGrid) {

    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
    if (iThread < nReceiversRegXZGrid) {
	    dev_signalOut_sigmaxz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxz[dev_receiversPositionRegXZGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmaxz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxz[dev_receiversPositionRegXZGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    }
}
/* Extract and interpolate data on xy shifted grid */
__global__ void ker_record_interp_data_xyGrid_3D(double *dev_newTimeSlice_sigmaxy, double *dev_signalOut_sigmaxy, long long its, long long it2, long long *dev_receiversPositionRegXYGrid, long long nReceiversRegXYGrid) {

    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
    if (iThread < nReceiversRegXYGrid) {
	    dev_signalOut_sigmaxy[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxy[dev_receiversPositionRegXYGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmaxy[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxy[dev_receiversPositionRegXYGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    }
}
/* Extract and interpolate data on yz shifted grid */
__global__ void ker_record_interp_data_yzGrid_3D(double *dev_newTimeSlice_sigmayz, double *dev_signalOut_sigmayz, long long its, long long it2, long long *dev_receiversPositionRegYZGrid, long long nReceiversRegYZGrid) {

    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
    if (iThread < nReceiversRegYZGrid) {
	    dev_signalOut_sigmayz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmayz[dev_receiversPositionRegYZGrid[iThread]] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmayz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmayz[dev_receiversPositionRegYZGrid[iThread]] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    }
}


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
			shared_c_vx[threadIdx.y][izLocal] = dev_c_vx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vy[threadIdx.y][izLocal] = dev_c_vy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vz[threadIdx.y][izLocal] = dev_c_vz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxx[threadIdx.y][izLocal] = dev_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayy[threadIdx.y][izLocal] = dev_c_sigmayy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmazz[threadIdx.y][izLocal] = dev_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxz[threadIdx.y][izLocal] = dev_c_sigmaxz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxy[threadIdx.y][izLocal] = dev_c_sigmaxy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayz[threadIdx.y][izLocal] = dev_c_sigmayz[iGlobal-dev_nz*FAT]; // Left side

			shared_c_vx[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_vx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vy[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_vy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vz[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_vz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxx[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayy[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmayy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmazz[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxz[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmaxz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxy[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmaxy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayz[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmayz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
		}

		// Load the halos in the z-direction
		if (threadIdx.x < FAT) {
			shared_c_vx[ixLocal][threadIdx.x] = dev_c_vx[iGlobal-FAT]; // Up
			shared_c_vy[ixLocal][threadIdx.x] = dev_c_vy[iGlobal-FAT]; // Up
			shared_c_vz[ixLocal][threadIdx.x] = dev_c_vz[iGlobal-FAT]; // Up
			shared_c_sigmaxx[ixLocal][threadIdx.x] = dev_c_sigmaxx[iGlobal-FAT]; // Up
			shared_c_sigmayy[ixLocal][threadIdx.x] = dev_c_sigmayy[iGlobal-FAT]; // Up
			shared_c_sigmazz[ixLocal][threadIdx.x] = dev_c_sigmazz[iGlobal-FAT]; // Up
			shared_c_sigmaxz[ixLocal][threadIdx.x] = dev_c_sigmaxz[iGlobal-FAT]; // Up
			shared_c_sigmaxy[ixLocal][threadIdx.x] = dev_c_sigmaxy[iGlobal-FAT]; // Up
			shared_c_sigmayz[ixLocal][threadIdx.x] = dev_c_sigmayz[iGlobal-FAT]; // Up

			shared_c_vx[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_vx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vy[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_vy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vz[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_vz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxx[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmaxx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayy[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmayy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmazz[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmazz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxz[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmaxz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxy[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmaxy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayz[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmayz[iGlobal+BLOCK_SIZE_Z]; // Down
		}

		// Wait until all threads of this block have loaded the slice y-slice into shared memory
		__syncthreads(); // Synchronise all threads within each block
		//
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
													dev_yCoeff[0]*(dev_c_sigmayz_y[3]-shared_c_sigmayz[ixLocal][izLocal]) +
											    dev_yCoeff[1]*(dev_c_sigmayz_y[4]-dev_c_sigmayz_y[2])+
											    dev_yCoeff[2]*(dev_c_sigmayz_y[5]-dev_c_sigmayz_y[1])+
											    dev_yCoeff[3]*(dev_c_sigmayz_y[6]-dev_c_sigmayz_y[0])+
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
													     dev_yCoeff[3]*(dev_c_vx_y[6]-dev_c_vx_y[0])+
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
													     dev_zCoeff[3]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-4][izLocal])+
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
													     dev_yCoeff[3]*(dev_c_vz_y[6]-dev_c_vz_y[0])+
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
			shared_c_sigmaxx[threadIdx.y][izLocal] = dev_c_sigmaxx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayy[threadIdx.y][izLocal] = dev_c_sigmayy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmazz[threadIdx.y][izLocal] = dev_c_sigmazz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxz[threadIdx.y][izLocal] = dev_c_sigmaxz[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmaxy[threadIdx.y][izLocal] = dev_c_sigmaxy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_sigmayz[threadIdx.y][izLocal] = dev_c_sigmayz[iGlobal-dev_nz*FAT]; // Left side

			shared_c_sigmaxx[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmaxx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayy[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmayy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmazz[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmazz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxz[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmaxz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmaxy[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmaxy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_sigmayz[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_sigmayz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
		}

		// Load the halos in the z-direction
		if (threadIdx.x < FAT) {
			shared_c_sigmaxx[ixLocal][threadIdx.x] = dev_c_sigmaxx[iGlobal-FAT]; // Up
			shared_c_sigmayy[ixLocal][threadIdx.x] = dev_c_sigmayy[iGlobal-FAT]; // Up
			shared_c_sigmazz[ixLocal][threadIdx.x] = dev_c_sigmazz[iGlobal-FAT]; // Up
			shared_c_sigmaxz[ixLocal][threadIdx.x] = dev_c_sigmaxz[iGlobal-FAT]; // Up
			shared_c_sigmaxy[ixLocal][threadIdx.x] = dev_c_sigmaxy[iGlobal-FAT]; // Up
			shared_c_sigmayz[ixLocal][threadIdx.x] = dev_c_sigmayz[iGlobal-FAT]; // Up

			shared_c_sigmaxx[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmaxx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayy[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmayy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmazz[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmazz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxz[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmaxz[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmaxy[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmaxy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_sigmayz[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_sigmayz[iGlobal+BLOCK_SIZE_Z]; // Down
		}

		// Wait until all threads of this block have loaded the slice y-slice into shared memory
		__syncthreads(); // Synchronise all threads within each block

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
													dev_yCoeff[0]*(dev_c_sigmayz_y[3]-shared_c_sigmayz[ixLocal][izLocal]) +
											    dev_yCoeff[1]*(dev_c_sigmayz_y[4]-dev_c_sigmayz_y[2])+
											    dev_yCoeff[2]*(dev_c_sigmayz_y[5]-dev_c_sigmayz_y[1])+
											    dev_yCoeff[3]*(dev_c_sigmayz_y[6]-dev_c_sigmayz_y[0])+
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
			shared_c_vx[threadIdx.y][izLocal] = dev_c_vx[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vy[threadIdx.y][izLocal] = dev_c_vy[iGlobal-dev_nz*FAT]; // Left side
			shared_c_vz[threadIdx.y][izLocal] = dev_c_vz[iGlobal-dev_nz*FAT]; // Left side

			shared_c_vx[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_vx[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vy[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_vy[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
			shared_c_vz[ixLocal+BLOCK_SIZE_X][izLocal] = dev_c_vz[iGlobal+dev_nz*BLOCK_SIZE_X]; // Right side
		}

		// Load the halos in the z-direction
		if (threadIdx.x < FAT) {
			shared_c_vx[ixLocal][threadIdx.x] = dev_c_vx[iGlobal-FAT]; // Up
			shared_c_vy[ixLocal][threadIdx.x] = dev_c_vy[iGlobal-FAT]; // Up
			shared_c_vz[ixLocal][threadIdx.x] = dev_c_vz[iGlobal-FAT]; // Up

			shared_c_vx[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_vx[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vy[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_vy[iGlobal+BLOCK_SIZE_Z]; // Down
			shared_c_vz[ixLocal][izLocal+BLOCK_SIZE_Z] = dev_c_vz[iGlobal+BLOCK_SIZE_Z]; // Down
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
													     dev_yCoeff[3]*(dev_c_vx_y[6]-dev_c_vx_y[0])+
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
													     dev_zCoeff[3]*(shared_c_vx[ixLocal+3][izLocal]-shared_c_vx[ixLocal-4][izLocal])+
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
													     dev_yCoeff[3]*(dev_c_vz_y[6]-dev_c_vz_y[0])+
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

/******************************************************************************/
/************************************** Damping *******************************/
/******************************************************************************/
__global__ void dampCosineEdge_3D(double *dev_p1_vx, double *dev_p2_vx, double *dev_p1_vy, double *dev_p2_vy, double *dev_p1_vz, double *dev_p2_vz, double *dev_p1_sigmaxx, double *dev_p2_sigmaxx, double *dev_p1_sigmayy, double *dev_p2_sigmayy, double *dev_p1_sigmazz, double *dev_p2_sigmazz, double *dev_p1_sigmaxz, double *dev_p2_sigmaxz, double *dev_p1_sigmaxy, double *dev_p2_sigmaxy, double *dev_p1_sigmayz, double *dev_p2_sigmayz) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate on the x-axis
  long long yStride = dev_nz * dev_nx;

	for (long long iyGlobal=FAT; iyGlobal<dev_ny-FAT; iyGlobal++){

		// Compute distance to the closest edge of model (not including the fat)
		// For example, the first non fat element will have a distance of 0
		long long distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, dev_nz-izGlobal-1-FAT, dev_nx-ixGlobal-1-FAT);
		distToEdge = min2(distToEdge,min2(iyGlobal-FAT,dev_ny-iyGlobal-1-FAT));

		if (distToEdge < dev_minPad){

			// Compute global index
			long long iGlobal = iyGlobal * yStride + dev_nz * ixGlobal + izGlobal;

			// Compute damping coefficient
			double damp = dev_cosDampingCoeff[distToEdge];

			// Apply damping
			// Velocities
			dev_p1_vx[iGlobal] *= damp;
			dev_p2_vx[iGlobal] *= damp;
			dev_p1_vy[iGlobal] *= damp;
			dev_p2_vy[iGlobal] *= damp;
			dev_p1_vz[iGlobal] *= damp;
			dev_p2_vz[iGlobal] *= damp;
			// Stresses
			dev_p1_sigmaxx[iGlobal] *= damp;
			dev_p2_sigmaxx[iGlobal] *= damp;
			dev_p1_sigmayy[iGlobal] *= damp;
			dev_p2_sigmayy[iGlobal] *= damp;
			dev_p1_sigmazz[iGlobal] *= damp;
			dev_p2_sigmazz[iGlobal] *= damp;
			dev_p1_sigmaxz[iGlobal] *= damp;
			dev_p2_sigmaxz[iGlobal] *= damp;
			dev_p1_sigmaxy[iGlobal] *= damp;
			dev_p2_sigmaxy[iGlobal] *= damp;
			dev_p1_sigmayz[iGlobal] *= damp;
			dev_p2_sigmayz[iGlobal] *= damp;
		}
	}
}
