#include "varElaDeclare_3D.h"
#include <stdio.h>

/******************************************************************************/
/******************************** Injection ***********************************/
/******************************************************************************/
/*															SOURCE INJECTION      												*/
// NORMAL STRESSES
/* Interpolate and inject source on center grid */
__global__ void ker_inject_source_centerGriddomDec_3D(float *dev_signalIn_sigmaxx, float *dev_signalIn_sigmayy, float *dev_signalIn_sigmazz, float *dev_timeSlice_sigmaxx, float *dev_timeSlice_sigmayy, float *dev_timeSlice_sigmazz, int its, int it2, long long *dev_sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long shift, long long min_idx, long long max_idx){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegCenterGrid && min_idx <= dev_sourcesPositionRegCenterGrid[iThread] && dev_sourcesPositionRegCenterGrid[iThread] < max_idx) {
			long long idx = dev_sourcesPositionRegCenterGrid[iThread]-shift;
			dev_timeSlice_sigmaxx[idx] += dev_signalIn_sigmaxx[dev_nts*iThread+its] * dev_timeInterpFilter[it2] + dev_signalIn_sigmaxx[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
			dev_timeSlice_sigmayy[idx] += dev_signalIn_sigmayy[dev_nts*iThread+its] * dev_timeInterpFilter[it2] + dev_signalIn_sigmayy[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
			dev_timeSlice_sigmazz[idx] += dev_signalIn_sigmazz[dev_nts*iThread+its] * dev_timeInterpFilter[it2] + dev_signalIn_sigmazz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}
//STAGGERED GRIDs
__global__ void ker_inject_source_stagGriddomDec_3D(float *dev_signalIn, float *dev_timeSlice, int its, int it2, long long *dev_sourcesPositionRegGrid, long long nSourcesRegGrid, long long shift, long long min_idx, long long max_idx){

    //thread per source device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		// if (iThread == 0 && its == 0) {printf("dev_sourcesPositionRegGrid[iThread] = %d\n", dev_sourcesPositionRegGrid[iThread]);}
		if (iThread < nSourcesRegGrid && min_idx <= dev_sourcesPositionRegGrid[iThread] && dev_sourcesPositionRegGrid[iThread] < max_idx) {
			long long idx = dev_sourcesPositionRegGrid[iThread]-shift;
			dev_timeSlice[idx] += dev_signalIn[dev_nts*iThread+its] * dev_timeInterpFilter[it2] + dev_signalIn[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}

/*															DATA INJECTION      													*/
// NORMAL STRESSES
/* Interpolate and inject data on center grid */
__global__ void ker_inject_data_centerGriddomDec_3D(float *dev_signalIn_sigmaxx, float *dev_signalIn_sigmayy, float *dev_signalIn_sigmazz, float *dev_timeSlice_sigmaxx, float *dev_timeSlice_sigmayy, float *dev_timeSlice_sigmazz, int its, int it2, long long *dev_receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long shift, long long min_idx, long long max_idx){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegCenterGrid && min_idx <= dev_receiversPositionRegCenterGrid[iThread] && dev_receiversPositionRegCenterGrid[iThread] < max_idx) {
			long long idx = dev_receiversPositionRegCenterGrid[iThread]-shift;
			dev_timeSlice_sigmaxx[idx] += dev_signalIn_sigmaxx[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmaxx[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
			dev_timeSlice_sigmayy[idx] += dev_signalIn_sigmayy[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmayy[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
			dev_timeSlice_sigmazz[idx] += dev_signalIn_sigmazz[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn_sigmazz[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}
//STAGGERED GRIDs
__global__ void ker_inject_data_stagGriddomDec_3D(float *dev_signalIn, float *dev_timeSlice, int its, int it2, long long *dev_receiversPositionRegGrid, long long nReceiversRegGrid, long long shift, long long min_idx, long long max_idx){

    //thread per receiver device
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nReceiversRegGrid && min_idx <= dev_receiversPositionRegGrid[iThread] && dev_receiversPositionRegGrid[iThread] < max_idx) {
			long long idx = dev_receiversPositionRegGrid[iThread]-shift;
			dev_timeSlice[idx] += dev_signalIn[dev_nts*iThread+its] * dev_timeInterpFilter[it2+1] + dev_signalIn[dev_nts*iThread+its+1] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2+1];
		}
}


/******************************************************************************/
/******************************* Extraction ***********************************/
/******************************************************************************/

/*															SOURCE EXTRACTION      												*/
// NORMAL STRESSES
/*extract and interpolate source thar are on center grid */
__global__ void ker_record_source_centerGriddomDec_3D(float *dev_newTimeSlice_sigmaxx, float *dev_newTimeSlice_sigmayy, float *dev_newTimeSlice_sigmazz, float *dev_signalOut_sigmaxx, float *dev_signalOut_sigmayy, float *dev_signalOut_sigmazz, int its, int it2, long long *dev_sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long shift, long long min_idx, long long max_idx) {
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegCenterGrid && min_idx <= dev_sourcesPositionRegCenterGrid[iThread] && dev_sourcesPositionRegCenterGrid[iThread] < max_idx){
			long long idx = dev_sourcesPositionRegCenterGrid[iThread] - shift;
			dev_signalOut_sigmaxx[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxx[idx] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmaxx[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxx[idx] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
			dev_signalOut_sigmayy[dev_nts*iThread+its]   += dev_newTimeSlice_sigmayy[idx] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmayy[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmayy[idx] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
	    dev_signalOut_sigmazz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmazz[idx] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmazz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmazz[idx] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}
//STAGGERED GRIDs
__global__ void ker_record_source_stagGriddomDec_3D(float *dev_newTimeSlice, float *dev_signalOut, int its, int it2, long long *dev_sourcesPositionRegGrid, long long nSourcesRegGrid, long long shift, long long min_idx, long long max_idx) {
    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
		if (iThread < nSourcesRegGrid && min_idx <= dev_sourcesPositionRegGrid[iThread] && dev_sourcesPositionRegGrid[iThread] < max_idx){
			long long idx = dev_sourcesPositionRegGrid[iThread] - shift;
			dev_signalOut[dev_nts*iThread+its]   += dev_newTimeSlice[idx] * dev_timeInterpFilter[it2];
	    dev_signalOut[dev_nts*iThread+its+1] += dev_newTimeSlice[idx] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
		}
}

/*															 DATA EXTRACTION      												*/
// NORMAL STRESSES
/* Extract and interpolate data on center grid */
__global__ void ker_record_interp_data_centerGriddomDec_3D(float *dev_newTimeSlice_sigmaxx, float *dev_newTimeSlice_sigmayy, float *dev_newTimeSlice_sigmazz, float *dev_signalOut_sigmaxx, float *dev_signalOut_sigmayy, float *dev_signalOut_sigmazz, int its, int it2, long long *dev_receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long shift, long long min_idx, long long max_idx) {

    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
    if (iThread < nReceiversRegCenterGrid && min_idx <= dev_receiversPositionRegCenterGrid[iThread] && dev_receiversPositionRegCenterGrid[iThread] < max_idx) {
			long long idx = dev_receiversPositionRegCenterGrid[iThread] - shift;
	    dev_signalOut_sigmaxx[dev_nts*iThread+its]   += dev_newTimeSlice_sigmaxx[idx] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmaxx[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmaxx[idx] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
			dev_signalOut_sigmayy[dev_nts*iThread+its]   += dev_newTimeSlice_sigmayy[idx] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmayy[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmayy[idx] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
	    dev_signalOut_sigmazz[dev_nts*iThread+its]   += dev_newTimeSlice_sigmazz[idx] * dev_timeInterpFilter[it2];
	    dev_signalOut_sigmazz[dev_nts*iThread+its+1] += dev_newTimeSlice_sigmazz[idx] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    }
}
//STAGGERED GRIDs
__global__ void ker_record_interp_data_stagGriddomDec_3D(float *dev_newTimeSlice, float *dev_signalOut, int its, int it2, long long *dev_receiversPositionRegGrid, long long nReceiversRegGrid, long long shift, long long min_idx, long long max_idx) {

    long long iThread = blockIdx.x * blockDim.x + threadIdx.x;
    if (iThread < nReceiversRegGrid && min_idx <= dev_receiversPositionRegGrid[iThread] && dev_receiversPositionRegGrid[iThread] < max_idx) {
			long long idx = dev_receiversPositionRegGrid[iThread] - shift;
	    dev_signalOut[dev_nts*iThread+its]   += dev_newTimeSlice[idx] * dev_timeInterpFilter[it2];
	    dev_signalOut[dev_nts*iThread+its+1] += dev_newTimeSlice[idx] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2];
    }
}


/******************************************************************************/
/******************************* Forward stepper ******************************/
/******************************************************************************/
//Calling kernel from kernelsElaGpu_3D

/******************************************************************************/
/******************************* Adjoint stepper ******************************/
/******************************************************************************/
//Calling kernel from kernelsElaGpu_3D

/******************************************************************************/
/************************************** Damping *******************************/
/******************************************************************************/
__global__ void dampCosineEdgedomDec_3D(float *dev_p1_vx, float *dev_p2_vx, float *dev_p1_vy, float *dev_p2_vy, float *dev_p1_vz, float *dev_p2_vz, float *dev_p1_sigmaxx, float *dev_p2_sigmaxx, float *dev_p1_sigmayy, float *dev_p2_sigmayy, float *dev_p1_sigmazz, float *dev_p2_sigmazz, float *dev_p1_sigmaxz, float *dev_p2_sigmaxz, float *dev_p1_sigmaxy, float *dev_p2_sigmaxy, float *dev_p1_sigmayz, float *dev_p2_sigmayz, int nx, int ny, int nz, int dampCond) {

	// dampCond = 0 y-leftmost domains (not dampening right side)
	// dampCond = 1 central domains (not dampening sides)
	// dampCond = 2 y-rightmost domains (not dampening left side)

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE_Z + threadIdx.x; // Global z-coordinate on the z-axis
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE_X + threadIdx.y; // Global x-coordinate on the x-axis
  long long yStride = nz * nx;

	long long iyStart, iyEnd;
	if (dampCond == 0) {
		iyStart=FAT;
		iyEnd=ny; //Dampening in the FAT layer
	} else if (dampCond == 2) {
		iyStart=0;//Dampening in the FAT layer
		iyEnd=ny-FAT;
	} else {
		iyStart=0;//Dampening in the FAT layer
		iyEnd=ny; //Dampening in the FAT layer
	}

	for (long long iyGlobal=iyStart; iyGlobal<iyEnd; iyGlobal++){

		// Compute distance to the closest edge of model (not including the fat)
		// For example, the first non fat element will have a distance of 0
		long long distToEdge;
		if (dampCond == 0) {
			//Don't dampen right side
			distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, nz-izGlobal-1-FAT, nx-ixGlobal-1-FAT);
			distToEdge = min2(distToEdge,iyGlobal-FAT);
		} else if (dampCond == 1) {
			//Don't dampen sides
			distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, nz-izGlobal-1-FAT, nx-ixGlobal-1-FAT);
		} else if (dampCond == 2) {
			//Don't dampen left side
			distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, nz-izGlobal-1-FAT, nx-ixGlobal-1-FAT);
			distToEdge = min2(distToEdge,ny-iyGlobal-1-FAT);
		}

		if (distToEdge < dev_minPad){

			// Compute global index
			long long iGlobal = iyGlobal * yStride + nz * ixGlobal + izGlobal;

			// Compute damping coefficient
			float damp = dev_cosDampingCoeff[distToEdge];

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

/****************************************************************************************/
/******************************** Wavefield Interpolation *******************************/
/****************************************************************************************/
__global__ void interpWavefieldVxVyVzdomDec_3D(float *dev_wavefieldVx_left, float *dev_wavefieldVx_right, float *dev_wavefieldVy_left, float *dev_wavefieldVy_right, float *dev_wavefieldVz_left, float *dev_wavefieldVz_right, float *dev_timeSliceVx, float *dev_timeSliceVy, float *dev_timeSliceVz, int nx, int yStart, int yEnd, int nz, int it2) {

    long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
    long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate

		// Number of elements in one y-slice
		long long yStride = nz * nx;

		// Global index of the first element at which we are going to compute the Laplacian
		long long iGlobal = yStart * yStride + nz * ixGlobal + izGlobal;

		// Loop over y
		for (long long iy=yStart; iy<yEnd; iy++){
			// Interpolating Vx and Vz
	    dev_wavefieldVx_left[iGlobal] += dev_timeSliceVx[iGlobal] * dev_timeInterpFilter[it2]; // its
	    dev_wavefieldVx_right[iGlobal] += dev_timeSliceVx[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2]; // its+1
			dev_wavefieldVy_left[iGlobal] += dev_timeSliceVy[iGlobal] * dev_timeInterpFilter[it2]; // its
	    dev_wavefieldVy_right[iGlobal] += dev_timeSliceVy[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2]; // its+1
			dev_wavefieldVz_left[iGlobal] += dev_timeSliceVz[iGlobal] * dev_timeInterpFilter[it2]; // its
	    dev_wavefieldVz_right[iGlobal] += dev_timeSliceVz[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+it2]; // its+1

			// Move forward one grid point in the y-direction
			iGlobal += yStride;
		}

}
/* Interpolate and inject secondary source at fine time-sampling */
__global__ void injectSecondarySourcedomDec_3D(float *dev_ssLeft, float *dev_ssRight, float *dev_p0, int nx, int yStart, int yEnd, int nz, int indexFilter){
	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate

	// Number of elements in one y-slice
	long long yStride = nz * nx;

	// Global index of the first element at which we are going to compute the Laplacian
	long long iGlobal = yStart * yStride + nz * ixGlobal + izGlobal;

	// Loop over y
	for (long long iy=yStart; iy<yEnd; iy++){
	  dev_p0[iGlobal] += dev_ssLeft[iGlobal] * dev_timeInterpFilter[indexFilter] + dev_ssRight[iGlobal] * dev_timeInterpFilter[dev_hTimeInterpFilter+indexFilter];

		// Move forward one grid point in the y-direction
		iGlobal += yStride;
	}
}
