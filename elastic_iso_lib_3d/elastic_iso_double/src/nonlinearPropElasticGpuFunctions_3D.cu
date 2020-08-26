#include <cstring>
#include <iostream>
#include "nonlinearPropElasticGpuFunctions_3D.h"
#include "varElaDeclare_3D.h"
#include "kernelsElaGpu_3D.cu"
#include "kernelsElaGpudomDec_3D.cu"
#include "cudaErrors_3D.cu"
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <assert.h>

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
// GPU info
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo){

	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);

	if (info == 1){

		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "---------------------------- INFO FOR GPU# " << deviceNumberInfo << " ----------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;

		// Number of devices
		std::cout << "Number of requested GPUs: " << gpuList.size() << std::endl;
		std::cout << "Number of available GPUs: " << nDevice << std::endl;
		std::cout << "Id of requested GPUs: ";
		for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
			if (iGpu<gpuList.size()-1){std::cout << gpuList[iGpu] << ", ";}
 			else{ std::cout << gpuList[iGpu] << std::endl;}
		}

		// Driver version
		std::cout << "Cuda driver version: " << cudaDriverGetVersion(&driver) << std::endl;

		// Get properties
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop,deviceNumberInfo);

		// Display
		std::cout << "Name: " << dprop.name << std::endl;
		std::cout << "Total global memory: " << dprop.totalGlobalMem/(1024*1024*1024) << " [GB] " << std::endl;
		std::cout << "Shared memory per block: " << dprop.sharedMemPerBlock/1024 << " [kB]" << std::endl;
		std::cout << "Number of register per block: " << dprop.regsPerBlock << std::endl;
		std::cout << "Warp size: " << dprop.warpSize << " [threads]" << std::endl;
		std::cout << "Maximum pitch allowed for memory copies in bytes: " << dprop.memPitch/(1024*1024*1024) << " [GB]" << std::endl;
		std::cout << "Maximum threads per block: " << dprop.maxThreadsPerBlock << std::endl;
		std::cout << "Maximum block dimensions: " << "(" << dprop.maxThreadsDim[0] << ", " << dprop.maxThreadsDim[1] << ", " << dprop.maxThreadsDim[2] << ")" << std::endl;
		std::cout << "Maximum grid dimensions: " << "(" << dprop.maxGridSize[0] << ", " << dprop.maxGridSize[1] << ", " << dprop.maxGridSize[2] << ")" << std::endl;
		std::cout << "Total constant memory: " << dprop.totalConstMem/1024 << " [kB]" << std::endl;
		std::cout << "Number of streaming multiprocessors on device: " << dprop.multiProcessorCount << std::endl;
		if (dprop.deviceOverlap == 1) {std::cout << "Device can simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.deviceOverlap != 1) {std::cout << "Device cannot simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.canMapHostMemory == 1) { std::cout << "Device can map host memory" << std::endl; }
		if (dprop.canMapHostMemory != 1) { std::cout << "Device cannot map host memory" << std::endl; }
		if (dprop.concurrentKernels == 1) {std::cout << "Device can support concurrent kernel" << std::endl;}
		if (dprop.concurrentKernels != 1) {std::cout << "Device cannot support concurrent kernel execution" << std::endl;}

		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	// Check that the number of requested GPU is less or equal to the total number of available GPUs
	if (gpuList.size()>nDevice) {
		std::cout << "**** ERROR [getGpuInfo_3D]: Number of requested GPU greater than available GPUs ****" << std::endl;
		return false;
	}

	// Check that the GPU numbers in the list are between 0 and nGpu-1
	for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
		if (gpuList[iGpu]<0 || gpuList[iGpu]>nDevice-1){
			std::cout << "**** ERROR [getGpuInfo_3D]: One of the element of the GPU Id list is not a valid GPU Id number ****" << std::endl;
			return false;
		}
	}

	return true;
}
// GPU P2P setup
// Setup_cuda checks there enough GPUs, and sets up the communication pipelines for future GPU-GPU transfers
void setGpuP2P(int nGpu, int info, std::vector<int> gpuList){

  for(int iGpu=0; iGpu<nGpu; iGpu++){
    cudaDeviceSynchronize();
    cudaSetDevice(gpuList[iGpu]);

    //Enable P2P memcopies between GPUs
    for(int jGpu=0; jGpu<nGpu; jGpu++){
      if(iGpu==jGpu) continue;
      int peer_access_available=0;
      cudaDeviceCanAccessPeer( &peer_access_available, gpuList[iGpu], gpuList[jGpu]);
      if(peer_access_available){
        cudaDeviceEnablePeerAccess(gpuList[jGpu],0);
      } else {
				if (info == 1) {
					std::cout << "GPUid: " << gpuList[iGpu] << "cannot access GPUid:" << gpuList[jGpu] << std::endl;
				}
			}
    }

  }

}


// Setting common parameters
void initNonlinearElasticGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

		// Set GPU
		cudaSetDevice(iGpuId);

		// Host variables
		host_nts = nts;
		host_sub = sub;
		host_ntw = (nts - 1) * sub + 1;
		host_minPad = minPad;

		/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
		// Only one GPU will perform the following
		if (iGpuId == iGpuAlloc) {

				// Time slices for FD stepping for each wavefield
				dev_p0_vx       = new double*[nGpu];
				dev_p0_vy       = new double*[nGpu];
				dev_p0_vz       = new double*[nGpu];
				dev_p0_sigmaxx  = new double*[nGpu];
				dev_p0_sigmayy  = new double*[nGpu];
				dev_p0_sigmazz  = new double*[nGpu];
				dev_p0_sigmaxz  = new double*[nGpu];
				dev_p0_sigmaxy  = new double*[nGpu];
				dev_p0_sigmayz  = new double*[nGpu];

				dev_p1_vx       = new double*[nGpu];
				dev_p1_vy       = new double*[nGpu];
				dev_p1_vz       = new double*[nGpu];
				dev_p1_sigmaxx  = new double*[nGpu];
				dev_p1_sigmayy  = new double*[nGpu];
				dev_p1_sigmazz  = new double*[nGpu];
				dev_p1_sigmaxz  = new double*[nGpu];
				dev_p1_sigmaxy  = new double*[nGpu];
				dev_p1_sigmayz  = new double*[nGpu];

				dev_temp1    = new double*[nGpu];

				// model (Source)
				dev_modelRegDts_vx = new double*[nGpu];
				dev_modelRegDts_vy = new double*[nGpu];
				dev_modelRegDts_vz = new double*[nGpu];
				dev_modelRegDts_sigmaxx = new double*[nGpu];
				dev_modelRegDts_sigmayy = new double*[nGpu];
				dev_modelRegDts_sigmazz = new double*[nGpu];
				dev_modelRegDts_sigmaxz = new double*[nGpu];
				dev_modelRegDts_sigmaxy = new double*[nGpu];
				dev_modelRegDts_sigmayz = new double*[nGpu];
				// data
				dev_dataRegDts_vx = new double*[nGpu];
				dev_dataRegDts_vy = new double*[nGpu];
				dev_dataRegDts_vz = new double*[nGpu];
				dev_dataRegDts_sigmaxx = new double*[nGpu];
				dev_dataRegDts_sigmayy = new double*[nGpu];
				dev_dataRegDts_sigmazz = new double*[nGpu];
				dev_dataRegDts_sigmaxz = new double*[nGpu];
				dev_dataRegDts_sigmaxy = new double*[nGpu];
				dev_dataRegDts_sigmayz = new double*[nGpu];

				// Source positions
				dev_sourcesPositionRegCenterGrid = new long long*[nGpu];
				dev_sourcesPositionRegXGrid = new long long*[nGpu];
				dev_sourcesPositionRegYGrid = new long long*[nGpu];
				dev_sourcesPositionRegZGrid = new long long*[nGpu];
				dev_sourcesPositionRegXZGrid = new long long*[nGpu];
				dev_sourcesPositionRegXYGrid = new long long*[nGpu];
				dev_sourcesPositionRegYZGrid = new long long*[nGpu];
				// Receiver positions
				dev_receiversPositionRegCenterGrid = new long long*[nGpu];
				dev_receiversPositionRegXGrid = new long long*[nGpu];
				dev_receiversPositionRegYGrid = new long long*[nGpu];
				dev_receiversPositionRegZGrid = new long long*[nGpu];
				dev_receiversPositionRegXZGrid = new long long*[nGpu];
				dev_receiversPositionRegXYGrid = new long long*[nGpu];
				dev_receiversPositionRegYZGrid = new long long*[nGpu];

				// Scaled velocity
				dev_rhoxDtw = new double*[nGpu]; // Precomputed scaling dtw / rho_x
				dev_rhoyDtw = new double*[nGpu]; // Precomputed scaling dtw / rho_y
				dev_rhozDtw = new double*[nGpu]; // Precomputed scaling dtw / rho_z
				dev_lamb2MuDtw = new double*[nGpu]; // Precomputed scaling (lambda + 2*mu) * dtw
				dev_lambDtw = new double*[nGpu]; // Precomputed scaling lambda * dtw
				dev_muxzDtw = new double*[nGpu]; // Precomputed scaling mu_xz * dtw
				dev_muxyDtw = new double*[nGpu]; // Precomputed scaling mu_xy * dtw
				dev_muyzDtw = new double*[nGpu]; // Precomputed scaling mu_yz * dtw

				// Streams for saving the wavefield and time slices
				// compStream = new cudaStream_t[nGpu];
				// transferStream = new cudaStream_t[nGpu];
				// pin_wavefieldSlice = new double*[nGpu];
				// dev_wavefieldDts_left = new double*[nGpu];
				// dev_wavefieldDts_right = new double*[nGpu];
				// dev_pStream = new double*[nGpu];

		}

		/**************************** COMPUTE DERIVATIVE COEFFICIENTS ************************/
		double zCoeff[COEFF_SIZE];
		double yCoeff[COEFF_SIZE];
		double xCoeff[COEFF_SIZE];

		zCoeff[0] = 1.196289062541883 / dz;
		zCoeff[1] = -0.079752604188901 / dz;
		zCoeff[2] = 0.009570312506634 / dz;
		zCoeff[3] = -6.975446437140719e-04 / dz;

		xCoeff[0] = 1.196289062541883 / dx;
		xCoeff[1] = -0.079752604188901 / dx;
		xCoeff[2] = 0.009570312506634 / dx;
		xCoeff[3] = -6.975446437140719e-04 / dx;

		yCoeff[0] = 1.196289062541883 / dx;
		yCoeff[1] = -0.079752604188901 / dx;
		yCoeff[2] = 0.009570312506634 / dx;
		yCoeff[3] = -6.975446437140719e-04 / dx;


		/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
		// Time interpolation filter length / half length
		int hInterpFilter = host_sub + 1;
		int nInterpFilter = 2 * hInterpFilter;

		// Check the subsampling coefficient is smaller than the maximum allowed
		if (sub>=SUB_MAX){
			std::cout << "**** ERROR [nonlinearPropElasticGpu_3D]: Subsampling parameter for time interpolation is too high ****" << std::endl;
			throw std::runtime_error("");
		}

		// Allocate and fill time interpolation filter
		double interpFilter[nInterpFilter];
		for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
			interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
			interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
			interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
			interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
		}

		/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
		if (minPad>=PAD_MAX){
				std::cout << "**** ERROR [nonlinearPropElasticGpu_3D]: Padding value is too high ****" << std::endl;
				throw std::runtime_error("");
		}
		double cosDampingCoeff[minPad];

		// Cosine padding
		for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
				double arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
				arg = alphaCos + (1.0-alphaCos) * cos(arg);
				cosDampingCoeff[iFilter-FAT] = arg;
		}

		// Check that the block size is consistent between parfile and "varDeclare.h"
		if (blockSize != BLOCK_SIZE) {
				std::cout << "**** ERROR [nonlinearPropElasticGpu_3D]: Block size for time stepper is not consistent with parfile ****" << std::endl;
				throw std::runtime_error("");
		}

		/**************************** COPY TO CONSTANT MEMORY *******************************/
		// Finite-difference coefficients
		// Copy derivative coefficients to device
		cuda_call(cudaMemcpyToSymbol(dev_zCoeff, zCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_xCoeff, xCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_yCoeff, yCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice));

		// Time interpolation filter
		cuda_call(cudaMemcpyToSymbol(dev_nTimeInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
		cuda_call(cudaMemcpyToSymbol(dev_hTimeInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
		cuda_call(cudaMemcpyToSymbol(dev_timeInterpFilter, interpFilter, nInterpFilter*sizeof(double), 0, cudaMemcpyHostToDevice)); // Filter

		// Cosine damping parameters
		cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(double), 0, cudaMemcpyHostToDevice)); // Array for damping
		cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(double), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
		cuda_call(cudaMemcpyToSymbol(dev_minPad, &minPad, sizeof(int), 0, cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

		// FD parameters
		cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
		cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}

// Allocate model paramaters and propagation slices
void allocateNonlinearElasticGpu_3D(double *rhoxDtw, double *rhoyDtw, double *rhozDtw, double *lamb2MuDtw, double *lambDtw, double *muxzDtw, double *muxyDtw, double *muyzDtw, int nx, int ny, int nz, int iGpu, int iGpuId){

		// Get GPU number
		cudaSetDevice(iGpuId);

		unsigned long long nModel = nz;
		nModel *= nx * ny;

		// Allocate scaled elastic parameters to device
		cuda_call(cudaMalloc((void**) &dev_rhoxDtw[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_rhoyDtw[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_rhozDtw[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_lamb2MuDtw[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_lambDtw[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_muxzDtw[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_muxyDtw[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_muyzDtw[iGpu], nModel*sizeof(double)));

		// Copy scaled elastic parameters to device
		cuda_call(cudaMemcpy(dev_rhoxDtw[iGpu], rhoxDtw, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_rhoyDtw[iGpu], rhoyDtw, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_rhozDtw[iGpu], rhozDtw, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_lamb2MuDtw[iGpu], lamb2MuDtw, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_lambDtw[iGpu], lambDtw, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_muxzDtw[iGpu], muxzDtw, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_muxyDtw[iGpu], muxyDtw, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_muyzDtw[iGpu], muyzDtw, nModel*sizeof(double), cudaMemcpyHostToDevice));

		// Allocate wavefield time slices on device (for the stepper)
		cuda_call(cudaMalloc((void**) &dev_p0_vx[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_vy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_vz[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmaxx[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmayy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmazz[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmaxz[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmaxy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p0_sigmayz[iGpu], nModel*sizeof(double)));

		cuda_call(cudaMalloc((void**) &dev_p1_vx[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_vy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_vz[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmaxx[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmayy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmazz[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmaxz[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmaxy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_p1_sigmayz[iGpu], nModel*sizeof(double)));

}
void deallocateNonlinearElasticGpu_3D(int iGpu, int iGpuId){
		cudaSetDevice(iGpuId); // Set device number on GPU cluster

		// Deallocate scaled elastic params
		cuda_call(cudaFree(dev_rhoxDtw[iGpu]));
		cuda_call(cudaFree(dev_rhoyDtw[iGpu]));
		cuda_call(cudaFree(dev_rhozDtw[iGpu]));
		cuda_call(cudaFree(dev_lamb2MuDtw[iGpu]));
		cuda_call(cudaFree(dev_lambDtw[iGpu]));
		cuda_call(cudaFree(dev_muxzDtw[iGpu]));
		cuda_call(cudaFree(dev_muxyDtw[iGpu]));
		cuda_call(cudaFree(dev_muyzDtw[iGpu]));

		// Deallocate wavefields
		cuda_call(cudaFree(dev_p0_vx[iGpu]));
		cuda_call(cudaFree(dev_p0_vy[iGpu]));
		cuda_call(cudaFree(dev_p0_vz[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmayy[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmaxy[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmayz[iGpu]));

		cuda_call(cudaFree(dev_p1_vx[iGpu]));
		cuda_call(cudaFree(dev_p1_vy[iGpu]));
		cuda_call(cudaFree(dev_p1_vz[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmayy[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmaxy[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmayz[iGpu]));
}

/****************************************************************************************/
/**************************** Functions for setting propagation *************************/
/****************************************************************************************/
void srcAllocateAndCopyToGpu_3D(long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, int iGpu){
		// Sources geometry
		// Central
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegCenterGrid[iGpu], sourcesPositionRegCenterGrid, nSourcesRegCenterGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// X-Staggered
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegXGrid[iGpu], sourcesPositionRegXGrid, nSourcesRegXGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// Y-Staggered
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegYGrid[iGpu], nSourcesRegYGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegYGrid[iGpu], sourcesPositionRegYGrid, nSourcesRegYGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// Z-Staggered
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegZGrid[iGpu], sourcesPositionRegZGrid, nSourcesRegZGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// XZ-Staggered
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegXZGrid[iGpu], sourcesPositionRegXZGrid, nSourcesRegXZGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// XY-Staggered
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegXYGrid[iGpu], nSourcesRegXYGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegXYGrid[iGpu], sourcesPositionRegXYGrid, nSourcesRegXYGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// YZ-Staggered
		cuda_call(cudaMalloc((void**) &dev_sourcesPositionRegYZGrid[iGpu], nSourcesRegYZGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_sourcesPositionRegYZGrid[iGpu], sourcesPositionRegYZGrid, nSourcesRegYZGrid*sizeof(long long), cudaMemcpyHostToDevice));
}
void recAllocateAndCopyToGpu_3D(long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int iGpu){
		// Receivers geometry
		// Central
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegCenterGrid[iGpu], nReceiversRegCenterGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegCenterGrid[iGpu], receiversPositionRegCenterGrid, nReceiversRegCenterGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// X-Staggered
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegXGrid[iGpu], nReceiversRegXGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegXGrid[iGpu], receiversPositionRegXGrid, nReceiversRegXGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// Y-Staggered
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegYGrid[iGpu], nReceiversRegYGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegYGrid[iGpu], receiversPositionRegYGrid, nReceiversRegYGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// Z-Staggered
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegZGrid[iGpu], nReceiversRegZGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegZGrid[iGpu], receiversPositionRegZGrid, nReceiversRegZGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// XZ-Staggered
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegXZGrid[iGpu], nReceiversRegXZGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegXZGrid[iGpu], receiversPositionRegZGrid, nReceiversRegXZGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// YX-Staggered
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegXYGrid[iGpu], nReceiversRegXYGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegXYGrid[iGpu], receiversPositionRegXYGrid, nReceiversRegXYGrid*sizeof(long long), cudaMemcpyHostToDevice));
		// YZ-Staggered
		cuda_call(cudaMalloc((void**) &dev_receiversPositionRegYZGrid[iGpu], nReceiversRegYZGrid*sizeof(long long)));
		cuda_call(cudaMemcpy(dev_receiversPositionRegYZGrid[iGpu], receiversPositionRegYZGrid, nReceiversRegYZGrid*sizeof(long long), cudaMemcpyHostToDevice));
}
void srcRecAllocateAndCopyToGpu_3D(long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int iGpu){

		srcAllocateAndCopyToGpu_3D(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, iGpu);
		recAllocateAndCopyToGpu_3D(receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, iGpu);
}

// allocate source-model signals on device
void modelAllocateGpu_3D(long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int iGpu){
		// fx
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_vx[iGpu], nSourcesRegXGrid*host_nts*sizeof(double)));
		// fy
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_vy[iGpu], nSourcesRegYGrid*host_nts*sizeof(double)));
		// fz
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_vz[iGpu], nSourcesRegZGrid*host_nts*sizeof(double)));
		// mxx
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_sigmaxx[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// myy
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_sigmayy[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// mzz
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_sigmazz[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// mxz
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_sigmaxz[iGpu], nSourcesRegXZGrid*host_nts*sizeof(double)));
		// mxy
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_sigmaxy[iGpu], nSourcesRegXYGrid*host_nts*sizeof(double)));
		// myz
		cuda_call(cudaMalloc((void**) &dev_modelRegDts_sigmayz[iGpu], nSourcesRegYZGrid*host_nts*sizeof(double)));
}
//copy source-model signals from host to device
void modelCopyToGpu_3D(double *modelRegDts_vx, double *modelRegDts_vy, double *modelRegDts_vz, double *modelRegDts_sigmaxx, double *modelRegDts_sigmayy, double *modelRegDts_sigmazz, double *modelRegDts_sigmaxz, double *modelRegDts_sigmaxy, double *modelRegDts_sigmayz, long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int iGpu){
		// fx
		cuda_call(cudaMemcpy(dev_modelRegDts_vx[iGpu], modelRegDts_vx, nSourcesRegXGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// fy
		cuda_call(cudaMemcpy(dev_modelRegDts_vy[iGpu], modelRegDts_vy, nSourcesRegYGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// fz
		cuda_call(cudaMemcpy(dev_modelRegDts_vz[iGpu], modelRegDts_vz, nSourcesRegZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// mxx
		cuda_call(cudaMemcpy(dev_modelRegDts_sigmaxx[iGpu], modelRegDts_sigmaxx, nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// myy
		cuda_call(cudaMemcpy(dev_modelRegDts_sigmayy[iGpu], modelRegDts_sigmayy, nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// mzz
		cuda_call(cudaMemcpy(dev_modelRegDts_sigmazz[iGpu], modelRegDts_sigmazz, nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// mxz
		cuda_call(cudaMemcpy(dev_modelRegDts_sigmaxz[iGpu], modelRegDts_sigmaxz, nSourcesRegXZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// mxy
		cuda_call(cudaMemcpy(dev_modelRegDts_sigmaxy[iGpu], modelRegDts_sigmaxy, nSourcesRegXYGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// myz
		cuda_call(cudaMemcpy(dev_modelRegDts_sigmayz[iGpu], modelRegDts_sigmayz, nSourcesRegYZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
}
//initialize source-model values to
void modelInitializeOnGpu_3D(long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int iGpu){
		// fx
		cuda_call(cudaMemset(dev_modelRegDts_vx[iGpu], 0, nSourcesRegXGrid*host_nts*sizeof(double)));
		// fy
		cuda_call(cudaMemset(dev_modelRegDts_vy[iGpu], 0, nSourcesRegYGrid*host_nts*sizeof(double)));
		// fz
		cuda_call(cudaMemset(dev_modelRegDts_vz[iGpu], 0, nSourcesRegZGrid*host_nts*sizeof(double)));
		// mxx
		cuda_call(cudaMemset(dev_modelRegDts_sigmaxx[iGpu], 0, nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// myy
		cuda_call(cudaMemset(dev_modelRegDts_sigmayy[iGpu], 0, nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// mzz
		cuda_call(cudaMemset(dev_modelRegDts_sigmazz[iGpu], 0, nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// mxz
		cuda_call(cudaMemset(dev_modelRegDts_sigmaxz[iGpu], 0, nSourcesRegXZGrid*host_nts*sizeof(double)));
		// mxy
		cuda_call(cudaMemset(dev_modelRegDts_sigmaxy[iGpu], 0, nSourcesRegXYGrid*host_nts*sizeof(double)));
		// myz
		cuda_call(cudaMemset(dev_modelRegDts_sigmayz[iGpu], 0, nSourcesRegYZGrid*host_nts*sizeof(double)));
}

//allocate data on device
void dataAllocateGpu_3D(long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int iGpu){
		// vx
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_vx[iGpu], nReceiversRegXGrid*host_nts*sizeof(double)));
		// vy
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_vy[iGpu], nReceiversRegYGrid*host_nts*sizeof(double)));
		// vz
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_vz[iGpu], nReceiversRegZGrid*host_nts*sizeof(double)));
		// sigmaxx
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmaxx[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double)));
		// sigmayy
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmayy[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double)));
		// sigmazz
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmazz[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double)));
		// sigmaxz
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmaxz[iGpu], nReceiversRegXZGrid*host_nts*sizeof(double)));
		// sigmaxy
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmaxy[iGpu], nReceiversRegXYGrid*host_nts*sizeof(double)));
		// sigmayz
		cuda_call(cudaMalloc((void**) &dev_dataRegDts_sigmayz[iGpu], nReceiversRegYZGrid*host_nts*sizeof(double)));
}
void dataCopyToGpu_3D(double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int iGpu){
		// vx
		cuda_call(cudaMemcpy(dev_dataRegDts_vx[iGpu], dataRegDts_vx, nReceiversRegXGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// vy
		cuda_call(cudaMemcpy(dev_dataRegDts_vy[iGpu], dataRegDts_vy, nReceiversRegYGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// vz
		cuda_call(cudaMemcpy(dev_dataRegDts_vz[iGpu], dataRegDts_vz, nReceiversRegZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// sigmaxx
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmaxx[iGpu], dataRegDts_sigmaxx, nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// sigmayy
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmayy[iGpu], dataRegDts_sigmayy, nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// sigmazz
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmazz[iGpu], dataRegDts_sigmazz, nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// sigmaxz
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmaxz[iGpu], dataRegDts_sigmaxz, nReceiversRegXZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// sigmaxy
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmaxy[iGpu], dataRegDts_sigmaxy, nReceiversRegXYGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// sigmayz
		cuda_call(cudaMemcpy(dev_dataRegDts_sigmayz[iGpu], dataRegDts_sigmayz, nReceiversRegYZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
}
void dataInitializeOnGpu_3D(long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int iGpu){
		// vx
		cuda_call(cudaMemset(dev_dataRegDts_vx[iGpu], 0, nReceiversRegXGrid*host_nts*sizeof(double)));
		// vy
		cuda_call(cudaMemset(dev_dataRegDts_vy[iGpu], 0, nReceiversRegYGrid*host_nts*sizeof(double)));
		// vz
		cuda_call(cudaMemset(dev_dataRegDts_vz[iGpu], 0, nReceiversRegZGrid*host_nts*sizeof(double)));
		// sigmaxx
		cuda_call(cudaMemset(dev_dataRegDts_sigmaxx[iGpu], 0, nReceiversRegCenterGrid*host_nts*sizeof(double)));
		// sigmayy
		cuda_call(cudaMemset(dev_dataRegDts_sigmayy[iGpu], 0, nReceiversRegCenterGrid*host_nts*sizeof(double)));
		// sigmazz
		cuda_call(cudaMemset(dev_dataRegDts_sigmazz[iGpu], 0, nReceiversRegCenterGrid*host_nts*sizeof(double)));
		// sigmaxz
		cuda_call(cudaMemset(dev_dataRegDts_sigmaxz[iGpu], 0, nReceiversRegXZGrid*host_nts*sizeof(double)));
		// sigmaxy
		cuda_call(cudaMemset(dev_dataRegDts_sigmaxy[iGpu], 0, nReceiversRegXYGrid*host_nts*sizeof(double)));
		// sigmayz
		cuda_call(cudaMemset(dev_dataRegDts_sigmayz[iGpu], 0, nReceiversRegYZGrid*host_nts*sizeof(double)));
}

void wavefieldInitializeOnGpu_3D(long long nModel, int iGpu){
		// nModel = nx * ny * nz

		// Current time slices
		cuda_call(cudaMemset(dev_p0_vx[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_vy[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_vz[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmaxx[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmayy[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmazz[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmaxz[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmaxy[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p0_sigmayz[iGpu], 0, nModel*sizeof(double)));

		// New/Old time slices
		cuda_call(cudaMemset(dev_p1_vx[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_vy[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_vz[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmaxx[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmayy[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmazz[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmaxz[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmaxy[iGpu], 0, nModel*sizeof(double)));
		cuda_call(cudaMemset(dev_p1_sigmayz[iGpu], 0, nModel*sizeof(double)));
}

//setup: a) src and receiver positions allocation and copying to device
//       b) allocate and copy model (arrays for sources for each wavefield) to device
//       c) allocate and initialize (0) data (receiver-recording arrays) on device
//       d) allocate and initialize (0) wavefield time slices on gpu
void setupFwdGpu_3D(double *modelRegDts_vx, double *modelRegDts_vy, double *modelRegDts_vz, double *modelRegDts_sigmaxx, double *modelRegDts_sigmayy, double *modelRegDts_sigmazz, double *modelRegDts_sigmaxz, double *modelRegDts_sigmaxy, double *modelRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, int iGpu, int iGpuId){
		// Set device number on GPU cluster
		cudaSetDevice(iGpuId);

		//allocate and copy src and rec geometry to gpu
		srcRecAllocateAndCopyToGpu_3D(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, iGpu);

		// Model - wavelets for each wavefield component. Allocate and copy to gpu
		modelAllocateGpu_3D(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);
		modelCopyToGpu_3D(modelRegDts_vx, modelRegDts_vy, modelRegDts_vz, modelRegDts_sigmaxx, modelRegDts_sigmayy, modelRegDts_sigmazz, modelRegDts_sigmaxz, modelRegDts_sigmaxy, modelRegDts_sigmayz, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);

		// Data - data recordings for each wavefield component. Allocate and initialize on gpu
		dataAllocateGpu_3D(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);
		dataInitializeOnGpu_3D(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);

		//initialize wavefield slices to zero
		long long nModel = nz;
		nModel *= nx * ny;
		wavefieldInitializeOnGpu_3D(nModel, iGpu);

}

//setup: a) src and receiver positions allocation and copying to device
//       b) allocate and initialize (0) model (arrays for sources for each wavefield) to device
//       c) allocate and copy data (receiver-recording arrays) to device
//       d) allocate and initialize (0) wavefield time slices to gpu
void setupAdjGpu_3D(double *modelRegDts_vx, double *modelRegDts_vy, double *modelRegDts_vz, double *modelRegDts_sigmaxx, double *modelRegDts_sigmayy, double *modelRegDts_sigmazz, double *modelRegDts_sigmaxz, double *modelRegDts_sigmaxy, double *modelRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, int iGpu, int iGpuId){
		// Set device number on GPU cluster
		cudaSetDevice(iGpuId);

		//allocate and copy src and rec geometry to gpu
		srcRecAllocateAndCopyToGpu_3D(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, iGpu);

		// Model - wavelets for each wavefield component. Allocate and initilaize to 0 on gpu
		modelAllocateGpu_3D(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);
		modelInitializeOnGpu_3D(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);

		// Data - data recordings for each wavefield component. Allocate and initialize on gpu
		dataAllocateGpu_3D(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);
		dataCopyToGpu_3D(dataRegDts_vx, dataRegDts_vy, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmayy, dataRegDts_sigmazz, dataRegDts_sigmaxz, dataRegDts_sigmaxy, dataRegDts_sigmayz, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);

		//initialize wavefield slices to zero
		long long nModel = nz;
		nModel *= nx *ny;
		wavefieldInitializeOnGpu_3D(nModel, iGpu);
}

void switchPointers_3D(int iGpu){
		// Vx
		dev_temp1[iGpu] = dev_p0_vx[iGpu];
		dev_p0_vx[iGpu] = dev_p1_vx[iGpu];
		dev_p1_vx[iGpu] = dev_temp1[iGpu];
		// Vy
		dev_temp1[iGpu] = dev_p0_vy[iGpu];
		dev_p0_vy[iGpu] = dev_p1_vy[iGpu];
		dev_p1_vy[iGpu] = dev_temp1[iGpu];
		// Vz
		dev_temp1[iGpu] = dev_p0_vz[iGpu];
		dev_p0_vz[iGpu] = dev_p1_vz[iGpu];
		dev_p1_vz[iGpu] = dev_temp1[iGpu];
		// Sigmaxx
		dev_temp1[iGpu] = dev_p0_sigmaxx[iGpu];
		dev_p0_sigmaxx[iGpu] = dev_p1_sigmaxx[iGpu];
		dev_p1_sigmaxx[iGpu] = dev_temp1[iGpu];
		// Sigmayy
		dev_temp1[iGpu] = dev_p0_sigmayy[iGpu];
		dev_p0_sigmayy[iGpu] = dev_p1_sigmayy[iGpu];
		dev_p1_sigmayy[iGpu] = dev_temp1[iGpu];
		// Sigmazz
		dev_temp1[iGpu] = dev_p0_sigmazz[iGpu];
		dev_p0_sigmazz[iGpu] = dev_p1_sigmazz[iGpu];
		dev_p1_sigmazz[iGpu] = dev_temp1[iGpu];
		// Sigmaxz
		dev_temp1[iGpu] = dev_p0_sigmaxz[iGpu];
		dev_p0_sigmaxz[iGpu] = dev_p1_sigmaxz[iGpu];
		dev_p1_sigmaxz[iGpu] = dev_temp1[iGpu];
		// Sigmaxy
		dev_temp1[iGpu] = dev_p0_sigmaxy[iGpu];
		dev_p0_sigmaxy[iGpu] = dev_p1_sigmaxy[iGpu];
		dev_p1_sigmaxy[iGpu] = dev_temp1[iGpu];
		// Sigmayz
		dev_temp1[iGpu] = dev_p0_sigmayz[iGpu];
		dev_p0_sigmayz[iGpu] = dev_p1_sigmayz[iGpu];
		dev_p1_sigmayz[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] = NULL;
}

/************************************************/
// 				 Kernel launching FWD functions
/************************************************/
void launchFwdStepKernels_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu){
		kernel_exec(stepFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_vx[iGpu], dev_p1_vy[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmayy[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p1_sigmaxy[iGpu], dev_p1_sigmayz[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_rhoxDtw[iGpu], dev_rhoyDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu], dev_muxyDtw[iGpu], dev_muyzDtw[iGpu], nx, ny, nz));
}
void launchFwdStepKernelsdomDec_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, long long shift, int iGpu, cudaStream_t stream){
		kernel_stream_exec(stepFwdGpu_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_p0_vx[iGpu]+shift, dev_p0_vy[iGpu]+shift, dev_p0_vz[iGpu]+shift, dev_p0_sigmaxx[iGpu]+shift, dev_p0_sigmayy[iGpu]+shift, dev_p0_sigmazz[iGpu]+shift, dev_p0_sigmaxz[iGpu]+shift, dev_p0_sigmaxy[iGpu]+shift, dev_p0_sigmayz[iGpu]+shift, dev_p1_vx[iGpu]+shift, dev_p1_vy[iGpu]+shift, dev_p1_vz[iGpu]+shift, dev_p1_sigmaxx[iGpu]+shift, dev_p1_sigmayy[iGpu]+shift, dev_p1_sigmazz[iGpu]+shift, dev_p1_sigmaxz[iGpu]+shift, dev_p1_sigmaxy[iGpu]+shift, dev_p1_sigmayz[iGpu]+shift, dev_p0_vx[iGpu]+shift, dev_p0_vy[iGpu]+shift, dev_p0_vz[iGpu]+shift, dev_p0_sigmaxx[iGpu]+shift, dev_p0_sigmayy[iGpu]+shift, dev_p0_sigmazz[iGpu]+shift, dev_p0_sigmaxz[iGpu]+shift, dev_p0_sigmaxy[iGpu]+shift, dev_p0_sigmayz[iGpu]+shift, dev_rhoxDtw[iGpu]+shift, dev_rhoyDtw[iGpu]+shift, dev_rhozDtw[iGpu]+shift, dev_lamb2MuDtw[iGpu]+shift, dev_lambDtw[iGpu]+shift, dev_muxzDtw[iGpu]+shift, dev_muxyDtw[iGpu]+shift, dev_muyzDtw[iGpu]+shift, nx, ny, nz));
}

void launchFwdInjectSourceKernels_3D(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouYGrid, int nblockSouZGrid, int nblockSouXZGrid, int nblockSouXYGrid, int nblockSouYZGrid, long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int its, int it2, int iGpu){
		// Mxx, Myy, Mzz
		kernel_exec(ker_inject_source_centerGrid_3D<<<nblockSouCenterGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_sigmaxx[iGpu], dev_modelRegDts_sigmayy[iGpu], dev_modelRegDts_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], its, it2-1, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid));
		// fx
		kernel_exec(ker_inject_source_stagGrid_3D<<<nblockSouXGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_vx[iGpu], dev_p0_vx[iGpu], its, it2-1, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid));
		// fy
		kernel_exec(ker_inject_source_stagGrid_3D<<<nblockSouYGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_vy[iGpu], dev_p0_vy[iGpu], its, it2-1, dev_sourcesPositionRegYGrid[iGpu], nSourcesRegYGrid));
		// fz
		kernel_exec(ker_inject_source_stagGrid_3D<<<nblockSouZGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_vz[iGpu], dev_p0_vz[iGpu], its, it2-1, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid));
		// Mxz
		kernel_exec(ker_inject_source_stagGrid_3D<<<nblockSouXZGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2-1, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid));
		// Mxy
		kernel_exec(ker_inject_source_stagGrid_3D<<<nblockSouXYGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_sigmaxy[iGpu], dev_p0_sigmaxy[iGpu], its, it2-1, dev_sourcesPositionRegXYGrid[iGpu], nSourcesRegXYGrid));
		// Myz
		kernel_exec(ker_inject_source_stagGrid_3D<<<nblockSouYZGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_sigmayz[iGpu], dev_p0_sigmayz[iGpu], its, it2-1, dev_sourcesPositionRegYZGrid[iGpu], nSourcesRegYZGrid));

}
void launchFwdInjectSourceKernelsdomDec_3D(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouYGrid, int nblockSouZGrid, int nblockSouXZGrid, int nblockSouXYGrid, int nblockSouYZGrid, long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int its, int it2, int iGpu, long long shift, long long min_idx, long long max_idx, cudaStream_t stream){
		// Mxx, Myy, Mzz
		kernel_stream_exec(ker_inject_source_centerGriddomDec_3D<<<nblockSouCenterGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_modelRegDts_sigmaxx[iGpu], dev_modelRegDts_sigmayy[iGpu], dev_modelRegDts_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], its, it2-1, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid, shift, min_idx, max_idx));
		// fx
		kernel_stream_exec(ker_inject_source_stagGriddomDec_3D<<<nblockSouXGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_modelRegDts_vx[iGpu], dev_p0_vx[iGpu], its, it2-1, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid, shift, min_idx, max_idx));
		// fy
		kernel_stream_exec(ker_inject_source_stagGriddomDec_3D<<<nblockSouYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_modelRegDts_vy[iGpu], dev_p0_vy[iGpu], its, it2-1, dev_sourcesPositionRegYGrid[iGpu], nSourcesRegYGrid, shift, min_idx, max_idx));
		// fz
		kernel_stream_exec(ker_inject_source_stagGriddomDec_3D<<<nblockSouZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_modelRegDts_vz[iGpu], dev_p0_vz[iGpu], its, it2-1, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid, shift, min_idx, max_idx));
		// Mxz
		kernel_stream_exec(ker_inject_source_stagGriddomDec_3D<<<nblockSouXZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_modelRegDts_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2-1, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid, shift, min_idx, max_idx));
		// Mxy
		kernel_stream_exec(ker_inject_source_stagGriddomDec_3D<<<nblockSouXYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_modelRegDts_sigmaxy[iGpu], dev_p0_sigmaxy[iGpu], its, it2-1, dev_sourcesPositionRegXYGrid[iGpu], nSourcesRegXYGrid, shift, min_idx, max_idx));
		// Myz
		kernel_stream_exec(ker_inject_source_stagGriddomDec_3D<<<nblockSouYZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_modelRegDts_sigmayz[iGpu], dev_p0_sigmayz[iGpu], its, it2-1, dev_sourcesPositionRegYZGrid[iGpu], nSourcesRegYZGrid, shift, min_idx, max_idx));

}

void launchDampCosineEdgeKernels_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu){
		kernel_exec(dampCosineEdge_3D<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vy[iGpu], dev_p1_vy[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p1_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p1_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_sigmayz[iGpu], nx, ny, nz));
}
void launchDampCosineEdgeKernelsdomDec_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu, int dampCond, cudaStream_t stream){
		kernel_stream_exec(dampCosineEdgedomDec_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vy[iGpu], dev_p1_vy[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p1_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p1_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_sigmayz[iGpu], nx, ny, nz, dampCond));
}

void launchFwdRecordInterpDataKernels_3D(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataYGrid, int nblockDataZGrid, int nblockDataXZGrid, int nblockDataXYGrid, int nblockDataYZGrid, long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int its, int it2, int iGpu){

		// Sxx, Syy, Szz
		kernel_exec(ker_record_interp_data_centerGrid_3D<<<nblockDataCenterGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmayy[iGpu], dev_dataRegDts_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu], nReceiversRegCenterGrid));
		// Vx
		kernel_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataXGrid, BLOCK_SIZE_DATA>>>(dev_p0_vx[iGpu], dev_dataRegDts_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu], nReceiversRegXGrid));
		// Vy
		kernel_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataYGrid, BLOCK_SIZE_DATA>>>(dev_p0_vy[iGpu], dev_dataRegDts_vy[iGpu], its, it2, dev_receiversPositionRegYGrid[iGpu], nReceiversRegYGrid));
		// Vz
		kernel_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataZGrid, BLOCK_SIZE_DATA>>>(dev_p0_vz[iGpu], dev_dataRegDts_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu], nReceiversRegZGrid));
		// Sxz
		kernel_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataXZGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxz[iGpu], dev_dataRegDts_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu], nReceiversRegXZGrid));
		// Sxy
		kernel_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataXYGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxy[iGpu], dev_dataRegDts_sigmaxy[iGpu], its, it2, dev_receiversPositionRegXYGrid[iGpu], nReceiversRegXYGrid));
		// Syz
		kernel_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataYZGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmayz[iGpu], dev_dataRegDts_sigmayz[iGpu], its, it2, dev_receiversPositionRegYZGrid[iGpu], nReceiversRegYZGrid));
}
void launchFwdRecordInterpDataKernelsdomDec_3D(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataYGrid, int nblockDataZGrid, int nblockDataXZGrid, int nblockDataXYGrid, int nblockDataYZGrid, long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int its, int it2, int iGpu, long long shift, long long min_idx, long long max_idx, cudaStream_t stream){

		// Sxx, Syy, Szz
		kernel_stream_exec(ker_record_interp_data_centerGriddomDec_3D<<<nblockDataCenterGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmayy[iGpu], dev_dataRegDts_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu], nReceiversRegCenterGrid, shift, min_idx, max_idx));
		// Vx
		kernel_stream_exec(ker_record_interp_data_stagGriddomDec_3D<<<nblockDataXGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_vx[iGpu], dev_dataRegDts_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu], nReceiversRegXGrid, shift, min_idx, max_idx));
		// Vy
		kernel_stream_exec(ker_record_interp_data_stagGriddomDec_3D<<<nblockDataYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_vy[iGpu], dev_dataRegDts_vy[iGpu], its, it2, dev_receiversPositionRegYGrid[iGpu], nReceiversRegYGrid, shift, min_idx, max_idx));
		// Vz
		kernel_stream_exec(ker_record_interp_data_stagGriddomDec_3D<<<nblockDataZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_vz[iGpu], dev_dataRegDts_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu], nReceiversRegZGrid, shift, min_idx, max_idx));
		// Sxz
		kernel_stream_exec(ker_record_interp_data_stagGriddomDec_3D<<<nblockDataXZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_sigmaxz[iGpu], dev_dataRegDts_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu], nReceiversRegXZGrid, shift, min_idx, max_idx));
		// Sxy
		kernel_stream_exec(ker_record_interp_data_stagGriddomDec_3D<<<nblockDataXYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_sigmaxy[iGpu], dev_dataRegDts_sigmaxy[iGpu], its, it2, dev_receiversPositionRegXYGrid[iGpu], nReceiversRegXYGrid, shift, min_idx, max_idx));
		// Syz
		kernel_stream_exec(ker_record_interp_data_stagGriddomDec_3D<<<nblockDataYZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_sigmayz[iGpu], dev_dataRegDts_sigmayz[iGpu], its, it2, dev_receiversPositionRegYZGrid[iGpu], nReceiversRegYZGrid, shift, min_idx, max_idx));
}


/************************************************/
// 				 Kernel launching ADJ functions
/************************************************/
void launchAdjStepKernels_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu){
		kernel_exec(stepAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_vx[iGpu], dev_p1_vy[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmayy[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p1_sigmaxy[iGpu], dev_p1_sigmayz[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_rhoxDtw[iGpu], dev_rhoyDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu], dev_muxyDtw[iGpu], dev_muyzDtw[iGpu], nx, ny, nz));
}

void launchAdjInterpInjectDataKernels_3D(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataYGrid, int nblockDataZGrid, int nblockDataXZGrid, int nblockDataXYGrid, int nblockDataYZGrid, long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int its, int it2, int iGpu){
	// Sxx, Syy, Szz
	kernel_exec(ker_inject_data_centerGrid_3D<<<nblockDataCenterGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmayy[iGpu], dev_dataRegDts_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu], nReceiversRegCenterGrid));
	// Vx
	kernel_exec(ker_inject_data_stagGrid_3D<<<nblockDataXGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_vx[iGpu], dev_p0_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu], nReceiversRegXGrid));
	// Vy
	kernel_exec(ker_inject_data_stagGrid_3D<<<nblockDataYGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_vy[iGpu], dev_p0_vy[iGpu], its, it2, dev_receiversPositionRegYGrid[iGpu], nReceiversRegYGrid));
	// Vz
	kernel_exec(ker_inject_data_stagGrid_3D<<<nblockDataZGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_vz[iGpu], dev_p0_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu], nReceiversRegZGrid));
	// Sxz
	kernel_exec(ker_inject_data_stagGrid_3D<<<nblockDataXZGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu], nReceiversRegXZGrid));
	// Sxy
	kernel_exec(ker_inject_data_stagGrid_3D<<<nblockDataXYGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_sigmaxy[iGpu], dev_p0_sigmaxy[iGpu], its, it2, dev_receiversPositionRegXYGrid[iGpu], nReceiversRegXYGrid));
	// Syz
	kernel_exec(ker_inject_data_stagGrid_3D<<<nblockDataYZGrid, BLOCK_SIZE_DATA>>>(dev_dataRegDts_sigmayz[iGpu], dev_p0_sigmayz[iGpu], its, it2, dev_receiversPositionRegYZGrid[iGpu], nReceiversRegYZGrid));

}

void launchAdjExtractSource_3D(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouYGrid, int nblockSouZGrid, int nblockSouXZGrid, int nblockSouXYGrid, int nblockSouYZGrid, long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int its, int it2, int iGpu){
	// Mxx, Myy, Mzz
	kernel_exec(ker_record_source_centerGrid_3D<<<nblockSouCenterGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_modelRegDts_sigmaxx[iGpu], dev_modelRegDts_sigmayy[iGpu], dev_modelRegDts_sigmazz[iGpu], its, it2, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid));
	// fx
	kernel_exec(ker_record_source_stagGrid_3D<<<nblockSouXGrid, BLOCK_SIZE_DATA>>>(dev_p0_vx[iGpu], dev_modelRegDts_vx[iGpu], its, it2, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid));
	// fy
	kernel_exec(ker_record_source_stagGrid_3D<<<nblockSouYGrid, BLOCK_SIZE_DATA>>>(dev_p0_vy[iGpu], dev_modelRegDts_vy[iGpu], its, it2, dev_sourcesPositionRegYGrid[iGpu], nSourcesRegYGrid));
	// fz
	kernel_exec(ker_record_source_stagGrid_3D<<<nblockSouZGrid, BLOCK_SIZE_DATA>>>(dev_p0_vz[iGpu], dev_modelRegDts_vz[iGpu], its, it2, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid));
	// Mxz
	kernel_exec(ker_record_source_stagGrid_3D<<<nblockSouXZGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxz[iGpu], dev_modelRegDts_sigmaxz[iGpu], its, it2, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid));
	// Mxy
	kernel_exec(ker_record_source_stagGrid_3D<<<nblockSouXYGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxy[iGpu], dev_modelRegDts_sigmaxy[iGpu], its, it2, dev_sourcesPositionRegXYGrid[iGpu], nSourcesRegXYGrid));
	// Myz
	kernel_exec(ker_record_source_stagGrid_3D<<<nblockSouYZGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmayz[iGpu], dev_modelRegDts_sigmayz[iGpu], its, it2, dev_sourcesPositionRegYZGrid[iGpu], nSourcesRegYZGrid));

}

/************************************************/
// 			Domain decompositions utily functions
/************************************************/
void copyHalos(long long offset_rec, long long offset_snd, int iGpu, int jGpu, int dstDevice, int srcDevice, long long yStride, cudaStream_t stream){

	// Vx
	cuda_call(cudaMemcpyPeerAsync(dev_p0_vx[iGpu]+offset_rec, dstDevice, dev_p0_vx[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));
	// Vy
	cuda_call(cudaMemcpyPeerAsync(dev_p0_vy[iGpu]+offset_rec, dstDevice, dev_p0_vy[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));
	// Vz
	cuda_call(cudaMemcpyPeerAsync(dev_p0_vz[iGpu]+offset_rec, dstDevice, dev_p0_vz[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));
	// Sigmaxx
	cuda_call(cudaMemcpyPeerAsync(dev_p0_sigmaxx[iGpu]+offset_rec, dstDevice, dev_p0_sigmaxx[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));
	// Sigmayy
	cuda_call(cudaMemcpyPeerAsync(dev_p0_sigmayy[iGpu]+offset_rec, dstDevice, dev_p0_sigmayy[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));
	// Sigmazz
	cuda_call(cudaMemcpyPeerAsync(dev_p0_sigmazz[iGpu]+offset_rec, dstDevice, dev_p0_sigmazz[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));
	// Sigmaxz
	cuda_call(cudaMemcpyPeerAsync(dev_p0_sigmaxz[iGpu]+offset_rec, dstDevice, dev_p0_sigmaxz[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));
	// Sigmaxy
	cuda_call(cudaMemcpyPeerAsync(dev_p0_sigmaxy[iGpu]+offset_rec, dstDevice, dev_p0_sigmaxy[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));
	// Sigmayz
	cuda_call(cudaMemcpyPeerAsync(dev_p0_sigmayz[iGpu]+offset_rec, dstDevice, dev_p0_sigmayz[jGpu]+offset_snd, srcDevice, yStride*FAT*sizeof(double), stream));

}

/************************************************/
// 				 Interface functions
/************************************************/


/****************************************************************************************/
/******************************* Nonlinear forward propagation **************************/
/****************************************************************************************/
void propElasticFwdGpu_3D(double *modelRegDts_vx, double *modelRegDts_vy, double *modelRegDts_vz, double *modelRegDts_sigmaxx, double *modelRegDts_sigmayy, double *modelRegDts_sigmazz, double *modelRegDts_sigmaxz, double *modelRegDts_sigmaxy, double *modelRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, int iGpu, int iGpuId){


		//setup: a) src and receiver positions allocation and copying to device
		//       b) allocate and copy model (arrays for sources for each wavefield) to device
		//       c) allocate and initialize(0) data (recevier recordings arrays) to device
		//       d) allocate and copy wavefield time slices to gpu
		setupFwdGpu_3D(modelRegDts_vx, modelRegDts_vy, modelRegDts_vz, modelRegDts_sigmaxx, modelRegDts_sigmayy, modelRegDts_sigmazz, modelRegDts_sigmaxz, modelRegDts_sigmaxy, modelRegDts_sigmayz, dataRegDts_vx, dataRegDts_vy, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmayy, dataRegDts_sigmazz, dataRegDts_sigmaxz, dataRegDts_sigmaxy, dataRegDts_sigmayz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, nx, ny, nz, iGpu, iGpuId);

		//Finite-difference grid and blocks
		int nblockx = (nz-2*FAT) / BLOCK_SIZE;
		int nblocky = (nx-2*FAT) / BLOCK_SIZE;
		dim3 dimGrid(nblockx, nblocky);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

		// Extraction grid size
		int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataYGrid = (nReceiversRegYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataXYGrid = (nReceiversRegXYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockDataYZGrid = (nReceiversRegYZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		// Injection grid size
		int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouYGrid = (nSourcesRegYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouXYGrid = (nSourcesRegXYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
		int nblockSouYZGrid = (nSourcesRegYZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

		// Start propagation
		for (int its = 0; its < host_nts-1; its++){
				// std::cout << "its = " << its << std::endl;
				for (int it2 = 1; it2 < host_sub+1; it2++){
						// Compute fine time-step index

							// Step forward
							launchFwdStepKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu);
							// Inject source
							launchFwdInjectSourceKernels_3D(nblockSouCenterGrid, nblockSouXGrid, nblockSouYGrid, nblockSouZGrid,nblockSouXZGrid, nblockSouXYGrid, nblockSouYZGrid, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, its, it2, iGpu);
							// Damp wavefields
							launchDampCosineEdgeKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu);
							// Extract and interpolate data
							launchFwdRecordInterpDataKernels_3D(nblockDataCenterGrid, nblockDataXGrid, nblockDataYGrid, nblockDataZGrid, nblockDataXZGrid, nblockDataXYGrid, nblockDataYZGrid, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, its, it2, iGpu);
							// Switch pointers
							switchPointers_3D(iGpu);

				}
		}

		// Copy data back to host
		cuda_call(cudaMemcpy(dataRegDts_vx, dev_dataRegDts_vx[iGpu], nReceiversRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_vy, dev_dataRegDts_vy[iGpu], nReceiversRegYGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_vz, dev_dataRegDts_vz[iGpu], nReceiversRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxx, dev_dataRegDts_sigmaxx[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmayy, dev_dataRegDts_sigmayy[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmazz, dev_dataRegDts_sigmazz[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxz, dev_dataRegDts_sigmaxz[iGpu], nReceiversRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxy, dev_dataRegDts_sigmaxy[iGpu], nReceiversRegXYGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmayz, dev_dataRegDts_sigmayz[iGpu], nReceiversRegYZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

		// Deallocate all slices
		cuda_call(cudaFree(dev_modelRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_vy[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmayy[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmaxz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmaxy[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmayz[iGpu]));

		cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vy[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmayy[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxy[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmayz[iGpu]));

		cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegYGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXYGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegYZGrid[iGpu]));

		cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegYGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXYGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegYZGrid[iGpu]));

}
void propElasticFwdGpudomDec_3D(double *modelRegDts_vx, double *modelRegDts_vy, double *modelRegDts_vz, double *modelRegDts_sigmaxx, double *modelRegDts_sigmayy, double *modelRegDts_sigmazz, double *modelRegDts_sigmaxz, double *modelRegDts_sigmaxy, double *modelRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, std::vector<int> ny_domDec, std::vector<int> gpuList){

	// Number of GPUs employed
	int nGpu = gpuList.size();
	// Other parameters
	long long yStride = nx * nz;
	long long minIdxInj[nGpu], maxIdxInj[nGpu]; //Minimum and maximum indices for injection
	long long minIdxExt[nGpu], maxIdxExt[nGpu]; //Minimum and maximum indices for extraction
	long long dampCond[nGpu]; //Absorbing boundary condition, shift for body propagation
	int nyBody[nGpu]; //Number of samples+2*FAT for body propagation

	//Injection-extraction variables
	minIdxInj[0] = 0;
	maxIdxInj[0] = yStride*ny_domDec[0];
	minIdxExt[0] = 0;
	maxIdxExt[0] = maxIdxInj[0]-yStride*FAT;
	//Propagation variables
	nyBody[0] = ny_domDec[0]-FAT;
	dampCond[0] = 0;
	for(int iGpu=1; iGpu<nGpu; iGpu++){
		//Injection-extraction variables
		minIdxInj[iGpu] = maxIdxExt[iGpu-1]-yStride*FAT;
		maxIdxInj[iGpu] = minIdxInj[iGpu]+yStride*ny_domDec[iGpu];
		minIdxExt[iGpu] = maxIdxExt[iGpu-1];
		maxIdxExt[iGpu] = maxIdxInj[iGpu]-yStride*FAT;
		//Propagation variables
		if (iGpu == nGpu-1){
			dampCond[iGpu] = 2;
			nyBody[iGpu] = ny_domDec[iGpu]-FAT;
			maxIdxExt[iGpu] = maxIdxInj[iGpu];
		} else{
			dampCond[iGpu] = 1;
			nyBody[iGpu] = ny_domDec[iGpu]-2*FAT;
		}
	}

	//Define separate streams for overlapping communication
  cudaStream_t haloStream[nGpu], bodyStream[nGpu];

	//Allocating temporary data arrays
	double* dataRegDts_vxTmp = new double[nReceiversRegXGrid*host_nts];
	double* dataRegDts_vyTmp = new double[nReceiversRegYGrid*host_nts];
	double* dataRegDts_vzTmp = new double[nReceiversRegZGrid*host_nts];
	double* dataRegDts_sigmaxxTmp = new double[nReceiversRegCenterGrid*host_nts];
	double* dataRegDts_sigmayyTmp = new double[nReceiversRegCenterGrid*host_nts];
	double* dataRegDts_sigmazzTmp = new double[nReceiversRegCenterGrid*host_nts];
	double* dataRegDts_sigmaxzTmp = new double[nReceiversRegXZGrid*host_nts];
	double* dataRegDts_sigmaxyTmp = new double[nReceiversRegXYGrid*host_nts];
	double* dataRegDts_sigmayzTmp = new double[nReceiversRegYZGrid*host_nts];

  for(int iGpu=0; iGpu<nGpu; iGpu++){
    cuda_call(cudaSetDevice(gpuList[iGpu]));
    cuda_call(cudaStreamCreate(&haloStream[iGpu]));
    cuda_call(cudaStreamCreate(&bodyStream[iGpu]));

		//allocate and copy src and rec geometry to gpus (allocate and copy the entire shot/receiver gather to all cards)
		srcRecAllocateAndCopyToGpu_3D(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, iGpu);

		// Model - wavelets for each wavefield component. Allocate and copy to gpu
		modelAllocateGpu_3D(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);
		modelCopyToGpu_3D(modelRegDts_vx, modelRegDts_vy, modelRegDts_vz, modelRegDts_sigmaxx, modelRegDts_sigmayy, modelRegDts_sigmazz, modelRegDts_sigmaxz, modelRegDts_sigmaxy, modelRegDts_sigmayz, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);

		// Data - data recordings for each wavefield component. Allocate and initialize on gpu
		dataAllocateGpu_3D(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);
		dataInitializeOnGpu_3D(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);

		//initialize wavefield slices to zero
		long long nModel = nz;
		nModel *= nx * ny_domDec[iGpu];
		wavefieldInitializeOnGpu_3D(nModel, iGpu);

  }

	//Finite-difference grid and blocks
	int nblockx = (nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (nx-2*FAT) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	// Extraction grid size
	int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataYGrid = (nReceiversRegYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataXYGrid = (nReceiversRegXYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataYZGrid = (nReceiversRegYZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	// Injection grid size
	int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouYGrid = (nSourcesRegYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouXYGrid = (nSourcesRegXYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouYZGrid = (nSourcesRegYZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation
	for (int its = 0; its < host_nts-1; its++){
			for (int it2 = 1; it2 < host_sub+1; it2++){

						/*************************************/
						// Step forward
						for(int iGpu=0; iGpu<nGpu; iGpu++){
					    cudaSetDevice(gpuList[iGpu]);

							//Computing Halos
							if (iGpu == 0){
								//Left-hand domain
								launchFwdStepKernelsdomDec_3D(dimGrid, dimBlock, nx, 3*FAT, nz, yStride*(ny_domDec[iGpu]-3*FAT), iGpu, haloStream[iGpu]);
							} else if(iGpu == nGpu-1) {
								//Right-hand domain
								launchFwdStepKernelsdomDec_3D(dimGrid, dimBlock, nx, 3*FAT, nz, 0, iGpu, haloStream[iGpu]);
							} else {
								//Central domains (left and right halos)
								launchFwdStepKernelsdomDec_3D(dimGrid, dimBlock, nx, 3*FAT, nz, 0, iGpu, haloStream[iGpu]);
								launchFwdStepKernelsdomDec_3D(dimGrid, dimBlock, nx, 3*FAT, nz, yStride*(ny_domDec[iGpu]-3*FAT), iGpu, haloStream[iGpu]);
							}

							// cudaStreamQuery(haloStream[iGpu]);

							//Computing Internal parts
							if (iGpu == 0) {
								launchFwdStepKernelsdomDec_3D(dimGrid, dimBlock, nx, nyBody[iGpu], nz, 0, iGpu, bodyStream[iGpu]);
							} else {
								launchFwdStepKernelsdomDec_3D(dimGrid, dimBlock, nx, nyBody[iGpu], nz, yStride*FAT, iGpu, bodyStream[iGpu]);
							}
						}
						/*************************************/

						/*************************************/
						//Exchange halos
						for(int iGpu=0; iGpu<nGpu-1; iGpu++){
							//Sending from iGpu to iGpu+1
					   	cudaSetDevice(gpuList[iGpu]);
							copyHalos(0, yStride*(ny_domDec[iGpu]-2*FAT), iGpu+1, iGpu, gpuList[iGpu+1], gpuList[iGpu], yStride, haloStream[iGpu]);
						}
						//Synchronize to avoid stalling
				    for(int iGpu=0; iGpu<nGpu; iGpu++){
				      cudaSetDevice(gpuList[iGpu]);
				      cuda_call(cudaStreamSynchronize(haloStream[iGpu]));
				    }
						for(int iGpu=1; iGpu<nGpu; iGpu++){
							//Sending from iGpu to iGpu-1
					    cudaSetDevice(gpuList[iGpu]);
							copyHalos(yStride*(ny_domDec[iGpu-1]-FAT), FAT*yStride, iGpu-1, iGpu, gpuList[iGpu-1], gpuList[iGpu], yStride, haloStream[iGpu]);
						}
						//Synchronize to avoid issue with injection kernels
				    for(int iGpu=0; iGpu<nGpu; iGpu++){
				      cudaSetDevice(gpuList[iGpu]);
				      cuda_call(cudaStreamSynchronize(haloStream[iGpu]));
				    }
						/*************************************/

						/*************************************/
						// Inject source everywhere in the domain (repeated injection in the halos)
						for(int iGpu=0; iGpu<nGpu; iGpu++){
					    cudaSetDevice(gpuList[iGpu]);
							launchFwdInjectSourceKernelsdomDec_3D(nblockSouCenterGrid, nblockSouXGrid, nblockSouYGrid, nblockSouZGrid,nblockSouXZGrid, nblockSouXYGrid, nblockSouYZGrid, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, its, it2, iGpu, minIdxInj[iGpu], minIdxInj[iGpu], maxIdxInj[iGpu], bodyStream[iGpu]);
						}
						/*************************************/

						/*************************************/
						// Damp wavefields
						for(int iGpu=0; iGpu<nGpu; iGpu++){
					    cudaSetDevice(gpuList[iGpu]);
							launchDampCosineEdgeKernelsdomDec_3D(dimGrid, dimBlock, nx, ny_domDec[iGpu], nz, iGpu, dampCond[iGpu], bodyStream[iGpu]);
						}
						/*************************************/

						/*************************************/
						// Extract and interpolate data (excluding the halos in this case)
						for(int iGpu=0; iGpu<nGpu; iGpu++){
					    cudaSetDevice(gpuList[iGpu]);
							launchFwdRecordInterpDataKernelsdomDec_3D(nblockDataCenterGrid, nblockDataXGrid, nblockDataYGrid, nblockDataZGrid, nblockDataXZGrid, nblockDataXYGrid, nblockDataYZGrid, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, its, it2, iGpu, minIdxInj[iGpu], minIdxExt[iGpu], maxIdxExt[iGpu], bodyStream[iGpu]);
						}
						/*************************************/

						/*************************************/
						// Switch pointers
						for(int iGpu=0; iGpu<nGpu; iGpu++){
							cudaSetDevice(gpuList[iGpu]);
							cuda_call(cudaDeviceSynchronize());
							switchPointers_3D(iGpu);
						}
						/*************************************/

			}
	}

	// Copy data back to host
	for(int iGpu=0; iGpu<nGpu; iGpu++){

		cudaSetDevice(gpuList[iGpu]);
		cuda_call(cudaMemcpy(dataRegDts_vxTmp, dev_dataRegDts_vx[iGpu], nReceiversRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_vyTmp, dev_dataRegDts_vy[iGpu], nReceiversRegYGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_vzTmp, dev_dataRegDts_vz[iGpu], nReceiversRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxxTmp, dev_dataRegDts_sigmaxx[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmayyTmp, dev_dataRegDts_sigmayy[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmazzTmp, dev_dataRegDts_sigmazz[iGpu], nReceiversRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxzTmp, dev_dataRegDts_sigmaxz[iGpu], nReceiversRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmaxyTmp, dev_dataRegDts_sigmaxy[iGpu], nReceiversRegXYGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		cuda_call(cudaMemcpy(dataRegDts_sigmayzTmp, dev_dataRegDts_sigmayz[iGpu], nReceiversRegYZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

		//Adding Vx to output data
		#pragma omp parallel for collapse(2)
		for (long long idev = 0; idev < nReceiversRegXGrid; idev++){
			for (int its = 0; its < host_nts; its++){
				long long idx = idev*host_nts + its;
				dataRegDts_vx[idx] += dataRegDts_vxTmp[idx];
			}
		}
		//Adding Vy to output data
		#pragma omp parallel for collapse(2)
		for (long long idev = 0; idev < nReceiversRegYGrid; idev++){
			for (int its = 0; its < host_nts; its++){
				long long idx = idev*host_nts + its;
				dataRegDts_vy[idx] += dataRegDts_vyTmp[idx];
			}
		}
		//Adding Vz to output data
		#pragma omp parallel for collapse(2)
		for (long long idev = 0; idev < nReceiversRegZGrid; idev++){
			for (int its = 0; its < host_nts; its++){
				long long idx = idev*host_nts + its;
				dataRegDts_vz[idx] += dataRegDts_vzTmp[idx];
			}
		}
		//Adding Sigmaxx, Sigmayy, Sigmazz to output data
		#pragma omp parallel for collapse(2)
		for (long long idev = 0; idev < nReceiversRegCenterGrid; idev++){
			for (int its = 0; its < host_nts; its++){
				long long idx = idev*host_nts + its;
				dataRegDts_sigmaxx[idx] += dataRegDts_sigmaxxTmp[idx];
				dataRegDts_sigmayy[idx] += dataRegDts_sigmayyTmp[idx];
				dataRegDts_sigmazz[idx] += dataRegDts_sigmazzTmp[idx];
			}
		}
		//Adding Sigmaxz to output data
		#pragma omp parallel for collapse(2)
		for (long long idev = 0; idev < nReceiversRegXZGrid; idev++){
			for (int its = 0; its < host_nts; its++){
				long long idx = idev*host_nts + its;
				dataRegDts_sigmaxz[idx] += dataRegDts_sigmaxzTmp[idx];
			}
		}
		//Adding Sigmaxy to output data
		#pragma omp parallel for collapse(2)
		for (long long idev = 0; idev < nReceiversRegXYGrid; idev++){
			for (int its = 0; its < host_nts; its++){
				long long idx = idev*host_nts + its;
				dataRegDts_sigmaxy[idx] += dataRegDts_sigmaxyTmp[idx];
			}
		}
		//Adding Sigmayz to output data
		#pragma omp parallel for collapse(2)
		for (long long idev = 0; idev < nReceiversRegYZGrid; idev++){
			for (int its = 0; its < host_nts; its++){
				long long idx = idev*host_nts + its;
				dataRegDts_sigmayz[idx] += dataRegDts_sigmayzTmp[idx];
			}
		}

	}

	//Deleting temporary data arrays
	delete[] dataRegDts_vxTmp;
	delete[] dataRegDts_vyTmp;
	delete[] dataRegDts_vzTmp;
	delete[] dataRegDts_sigmaxxTmp;
	delete[] dataRegDts_sigmayyTmp;
	delete[] dataRegDts_sigmazzTmp;
	delete[] dataRegDts_sigmaxzTmp;
	delete[] dataRegDts_sigmaxyTmp;
	delete[] dataRegDts_sigmayzTmp;


	// Deallocate all slices
	for(int iGpu=0; iGpu<nGpu; iGpu++){
		cuda_call(cudaFree(dev_modelRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_vy[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmayy[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmaxz[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmaxy[iGpu]));
		cuda_call(cudaFree(dev_modelRegDts_sigmayz[iGpu]));

		cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vy[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmayy[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmaxy[iGpu]));
		cuda_call(cudaFree(dev_dataRegDts_sigmayz[iGpu]));

		cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegYGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegXYGrid[iGpu]));
		cuda_call(cudaFree(dev_sourcesPositionRegYZGrid[iGpu]));

		cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegYGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegXYGrid[iGpu]));
		cuda_call(cudaFree(dev_receiversPositionRegYZGrid[iGpu]));

		// Destroying CUDA streams
		cuda_call(cudaStreamDestroy(haloStream[iGpu]));
		cuda_call(cudaStreamDestroy(bodyStream[iGpu]));

	}


}


void propElasticAdjGpu_3D(double *modelRegDts_vx, double *modelRegDts_vy, double *modelRegDts_vz, double *modelRegDts_sigmaxx, double *modelRegDts_sigmayy, double *modelRegDts_sigmazz, double *modelRegDts_sigmaxz, double *modelRegDts_sigmaxy, double *modelRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, int iGpu, int iGpuId) {

	//setup: a) src and receiver positions allocation and copying to device
	//       b) allocate and initialize (0) model (arrays for sources for each wavefield) to device
	//       c) allocate and copy data (recevier recordings arrays) to device
	//       d) allocate and copy wavefield time slices to gpu
	setupAdjGpu_3D(modelRegDts_vx, modelRegDts_vy, modelRegDts_vz, modelRegDts_sigmaxx, modelRegDts_sigmayy, modelRegDts_sigmazz, modelRegDts_sigmaxz, modelRegDts_sigmaxy, modelRegDts_sigmayz, dataRegDts_vx, dataRegDts_vy, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmayy, dataRegDts_sigmazz, dataRegDts_sigmaxz, dataRegDts_sigmaxy, dataRegDts_sigmayz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, nx, ny, nz, iGpu, iGpuId);

	//Finite-difference grid and blocks
	int nblockx = (nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (nx-2*FAT) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Grid and block dimensions for data injection
	// Injection grid size (data)
	int nblockDataCenterGrid = (nReceiversRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataXGrid = (nReceiversRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataYGrid = (nReceiversRegYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataZGrid = (nReceiversRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataXZGrid = (nReceiversRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataXYGrid = (nReceiversRegXYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockDataYZGrid = (nReceiversRegYZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	// Extraction grid size (source)
	int nblockSouCenterGrid = (nSourcesRegCenterGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouXGrid = (nSourcesRegXGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouYGrid = (nSourcesRegYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouZGrid = (nSourcesRegZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouXZGrid = (nSourcesRegXZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouXYGrid = (nSourcesRegXYGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	int nblockSouYZGrid = (nSourcesRegYZGrid+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	for (int its = host_nts-2; its > -1; its--){
			for (int it2 = host_sub-1; it2 > -1; it2--){
					// Compute fine time-step index

					// Step adjoint
					launchAdjStepKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu);
					// Inject and interpolate data
					launchAdjInterpInjectDataKernels_3D(nblockDataCenterGrid, nblockDataXGrid, nblockDataYGrid, nblockDataZGrid, nblockDataXZGrid, nblockDataXYGrid, nblockDataYZGrid, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, its, it2, iGpu);
					// Damp wavefields
					launchDampCosineEdgeKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu);
					// Extract and interpolate source
					launchAdjExtractSource_3D(nblockSouCenterGrid, nblockSouXGrid, nblockSouYGrid, nblockSouZGrid, nblockSouXZGrid, nblockSouXYGrid, nblockSouYZGrid, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, its, it2, iGpu);
					// Switch pointers
					switchPointers_3D(iGpu);

			}
	}

	// Copy model back to host
	cuda_call(cudaMemcpy(modelRegDts_vx, dev_modelRegDts_vx[iGpu], nSourcesRegXGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(modelRegDts_vy, dev_modelRegDts_vy[iGpu], nSourcesRegYGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(modelRegDts_vz, dev_modelRegDts_vz[iGpu], nSourcesRegZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(modelRegDts_sigmaxx, dev_modelRegDts_sigmaxx[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(modelRegDts_sigmayy, dev_modelRegDts_sigmayy[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(modelRegDts_sigmazz, dev_modelRegDts_sigmazz[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(modelRegDts_sigmaxz, dev_modelRegDts_sigmaxz[iGpu], nSourcesRegXZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(modelRegDts_sigmaxy, dev_modelRegDts_sigmaxy[iGpu], nSourcesRegXYGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(modelRegDts_sigmayz, dev_modelRegDts_sigmayz[iGpu], nSourcesRegYZGrid*host_nts*sizeof(double), cudaMemcpyDeviceToHost));


	// Deallocate all slices
	cuda_call(cudaFree(dev_modelRegDts_vx[iGpu]));
	cuda_call(cudaFree(dev_modelRegDts_vy[iGpu]));
	cuda_call(cudaFree(dev_modelRegDts_vz[iGpu]));
	cuda_call(cudaFree(dev_modelRegDts_sigmaxx[iGpu]));
	cuda_call(cudaFree(dev_modelRegDts_sigmayy[iGpu]));
	cuda_call(cudaFree(dev_modelRegDts_sigmazz[iGpu]));
	cuda_call(cudaFree(dev_modelRegDts_sigmaxz[iGpu]));
	cuda_call(cudaFree(dev_modelRegDts_sigmaxy[iGpu]));
	cuda_call(cudaFree(dev_modelRegDts_sigmayz[iGpu]));

	cuda_call(cudaFree(dev_dataRegDts_vx[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts_vy[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts_vz[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts_sigmaxx[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts_sigmayy[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts_sigmazz[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts_sigmaxz[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts_sigmaxy[iGpu]));
	cuda_call(cudaFree(dev_dataRegDts_sigmayz[iGpu]));

	cuda_call(cudaFree(dev_sourcesPositionRegCenterGrid[iGpu]));
	cuda_call(cudaFree(dev_sourcesPositionRegXGrid[iGpu]));
	cuda_call(cudaFree(dev_sourcesPositionRegYGrid[iGpu]));
	cuda_call(cudaFree(dev_sourcesPositionRegZGrid[iGpu]));
	cuda_call(cudaFree(dev_sourcesPositionRegXZGrid[iGpu]));
	cuda_call(cudaFree(dev_sourcesPositionRegXYGrid[iGpu]));
	cuda_call(cudaFree(dev_sourcesPositionRegYZGrid[iGpu]));

	cuda_call(cudaFree(dev_receiversPositionRegCenterGrid[iGpu]));
	cuda_call(cudaFree(dev_receiversPositionRegXGrid[iGpu]));
	cuda_call(cudaFree(dev_receiversPositionRegYGrid[iGpu]));
	cuda_call(cudaFree(dev_receiversPositionRegZGrid[iGpu]));
	cuda_call(cudaFree(dev_receiversPositionRegXZGrid[iGpu]));
	cuda_call(cudaFree(dev_receiversPositionRegXYGrid[iGpu]));
	cuda_call(cudaFree(dev_receiversPositionRegYZGrid[iGpu]));
}





//
