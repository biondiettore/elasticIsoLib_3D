#include "varElaDeclare_3D.h"
#include <stdio.h>

/******************************************************************************/
/******************************** Injection ***********************************/
/******************************************************************************/
/*															SOURCE INJECTION      												*/
// NORMAL STRESSES
/* Interpolate and inject source on center grid */
__global__ void ker_inject_source_centerGriddomDec_3D(double *dev_signalIn_sigmaxx, double *dev_signalIn_sigmayy, double *dev_signalIn_sigmazz, double *dev_timeSlice_sigmaxx, double *dev_timeSlice_sigmayy, double *dev_timeSlice_sigmazz, int its, int it2, long long *dev_sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long shift, long long min_idx, long long max_idx){

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
__global__ void ker_inject_source_stagGriddomDec_3D(double *dev_signalIn, double *dev_timeSlice, int its, int it2, long long *dev_sourcesPositionRegGrid, long long nSourcesRegGrid, long long shift, long long min_idx, long long max_idx){

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
__global__ void ker_inject_data_centerGriddomDec_3D(double *dev_signalIn_sigmaxx, double *dev_signalIn_sigmayy, double *dev_signalIn_sigmazz, double *dev_timeSlice_sigmaxx, double *dev_timeSlice_sigmayy, double *dev_timeSlice_sigmazz, int its, int it2, long long *dev_receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long shift, long long min_idx, long long max_idx){

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
__global__ void ker_inject_data_stagGriddomDec_3D(double *dev_signalIn, double *dev_timeSlice, int its, int it2, long long *dev_receiversPositionRegGrid, long long nReceiversRegGrid, long long shift, long long min_idx, long long max_idx){

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
__global__ void ker_record_source_centerGriddomDec_3D(double *dev_newTimeSlice_sigmaxx, double *dev_newTimeSlice_sigmayy, double *dev_newTimeSlice_sigmazz, double *dev_signalOut_sigmaxx, double *dev_signalOut_sigmayy, double *dev_signalOut_sigmazz, int its, int it2, long long *dev_sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long shift, long long min_idx, long long max_idx) {
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
__global__ void ker_record_source_stagGriddomDec_3D(double *dev_newTimeSlice, double *dev_signalOut, int its, int it2, long long *dev_sourcesPositionRegGrid, long long nSourcesRegGrid, long long shift, long long min_idx, long long max_idx) {
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
__global__ void ker_record_interp_data_centerGriddomDec_3D(double *dev_newTimeSlice_sigmaxx, double *dev_newTimeSlice_sigmayy, double *dev_newTimeSlice_sigmazz, double *dev_signalOut_sigmaxx, double *dev_signalOut_sigmayy, double *dev_signalOut_sigmazz, int its, int it2, long long *dev_receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long shift, long long min_idx, long long max_idx) {

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
__global__ void ker_record_interp_data_stagGriddomDec_3D(double *dev_newTimeSlice, double *dev_signalOut, int its, int it2, long long *dev_receiversPositionRegGrid, long long nReceiversRegGrid, long long shift, long long min_idx, long long max_idx) {

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
__global__ void dampCosineEdgedomDec_3D(double *dev_p1_vx, double *dev_p2_vx, double *dev_p1_vy, double *dev_p2_vy, double *dev_p1_vz, double *dev_p2_vz, double *dev_p1_sigmaxx, double *dev_p2_sigmaxx, double *dev_p1_sigmayy, double *dev_p2_sigmayy, double *dev_p1_sigmazz, double *dev_p2_sigmazz, double *dev_p1_sigmaxz, double *dev_p2_sigmaxz, double *dev_p1_sigmaxy, double *dev_p2_sigmaxy, double *dev_p1_sigmayz, double *dev_p2_sigmayz, int nx, int ny, int nz, int dampCond) {

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
