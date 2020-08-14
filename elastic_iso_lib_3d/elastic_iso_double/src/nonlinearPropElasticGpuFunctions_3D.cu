#include <cstring>
#include <iostream>
#include "nonlinearPropElasticGpuFunctions_3D.h"
#include "varElaDeclare_3D.h"
#include "kernelsElaGpu_3D.cu"
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
// Setting common parameters
void initNonlinearElasticGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

		// Set GPU
		cudaSetDevice(iGpuId);

		// Host variables
		host_nz = nz;
		host_nx = nx;
		host_ny = ny;
		host_nModel = nz * nx * ny;
		host_yStride = nz * nx;
		host_nts = nts;
		host_sub = sub;
		host_ntw = (nts - 1) * sub + 1;

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
		std::cout << "[allocateNonlinearElasticGpu_3D] nModel = " << nModel << std::endl;
		std::cout << "Max rhoxDtw = " << *std::max_element(rhoxDtw,rhoxDtw+nModel) << std::endl;
		std::cout << "Min rhoxDtw = " << *std::min_element(rhoxDtw,rhoxDtw+nModel) << std::endl;
		std::cout << "Max rhoyDtw = " << *std::max_element(rhoyDtw,rhoyDtw+nModel) << std::endl;
		std::cout << "Min rhoyDtw = " << *std::min_element(rhoyDtw,rhoyDtw+nModel) << std::endl;
		std::cout << "Max rhozDtw = " << *std::max_element(rhozDtw,rhozDtw+nModel) << std::endl;
		std::cout << "Min rhozDtw = " << *std::min_element(rhozDtw,rhozDtw+nModel) << std::endl;
		std::cout << "Max lamb2MuDtw = " << *std::max_element(lamb2MuDtw,lamb2MuDtw+nModel) << std::endl;
		std::cout << "Min lamb2MuDtw = " << *std::min_element(lamb2MuDtw,lamb2MuDtw+nModel) << std::endl;
		std::cout << "Max lambDtw = " << *std::max_element(lambDtw,lambDtw+nModel) << std::endl;
		std::cout << "Min lambDtw = " << *std::min_element(lambDtw,lambDtw+nModel) << std::endl;
		std::cout << "Max muxzDtw = " << *std::max_element(muxzDtw,muxzDtw+nModel) << std::endl;
		std::cout << "Min muxzDtw = " << *std::min_element(muxzDtw,muxzDtw+nModel) << std::endl;
		std::cout << "Max muxyDtw = " << *std::max_element(muxyDtw,muxyDtw+nModel) << std::endl;
		std::cout << "Min muxyDtw = " << *std::min_element(muxyDtw,muxyDtw+nModel) << std::endl;
		std::cout << "Max muyzDtw = " << *std::max_element(muyzDtw,muyzDtw+nModel) << std::endl;
		std::cout << "Min muyzDtw = " << *std::min_element(muyzDtw,muyzDtw+nModel) << std::endl;

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
	kernel_exec(stepFwdGpu_3DGlobal<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_vx[iGpu], dev_p1_vy[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmayy[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p1_sigmaxy[iGpu], dev_p1_sigmayz[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_rhoxDtw[iGpu], dev_rhoyDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu], dev_muxyDtw[iGpu], dev_muyzDtw[iGpu], nx, ny, nz));
		// kernel_exec(stepFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_vx[iGpu], dev_p1_vy[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmayy[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p1_sigmaxy[iGpu], dev_p1_sigmayz[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_rhoxDtw[iGpu], dev_rhoyDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu], dev_muxyDtw[iGpu], dev_muyzDtw[iGpu], nx, ny, nz));
		// kernel_exec(stepFwdGpu_vel_3D<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_vx[iGpu], dev_p1_vy[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmayy[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p1_sigmaxy[iGpu], dev_p1_sigmayz[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_rhoxDtw[iGpu], dev_rhoyDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu], dev_muxyDtw[iGpu], dev_muyzDtw[iGpu], nx, ny, nz));
		// kernel_exec(stepFwdGpu_stress_3D<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_vx[iGpu], dev_p1_vy[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmayy[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p1_sigmaxy[iGpu], dev_p1_sigmayz[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_rhoxDtw[iGpu], dev_rhoyDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu], dev_muxyDtw[iGpu], dev_muyzDtw[iGpu], nx, ny, nz));
}

void launchFwdInjectSourceKernels_3D(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouYGrid, int nblockSouZGrid, int nblockSouXZGrid, int nblockSouXYGrid, int nblockSouYZGrid, long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int its, int it2, int iGpu){
		// Mxx, Myy, Mzz
		kernel_exec(ker_inject_source_centerGrid_3D<<<nblockSouCenterGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_sigmaxx[iGpu], dev_modelRegDts_sigmayy[iGpu], dev_modelRegDts_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], its, it2-1, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid));
		// fx
		kernel_exec(ker_inject_source_xGrid_3D<<<nblockSouXGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_vx[iGpu], dev_p0_vx[iGpu], its, it2-1, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid));
		// fy
		kernel_exec(ker_inject_source_yGrid_3D<<<nblockSouYGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_vy[iGpu], dev_p0_vy[iGpu], its, it2-1, dev_sourcesPositionRegYGrid[iGpu], nSourcesRegYGrid));
		// fz
		kernel_exec(ker_inject_source_zGrid_3D<<<nblockSouZGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_vz[iGpu], dev_p0_vz[iGpu], its, it2-1, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid));
		// Mxz
		kernel_exec(ker_inject_source_xzGrid_3D<<<nblockSouXZGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2-1, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid));
		// Mxy
		kernel_exec(ker_inject_source_xyGrid_3D<<<nblockSouXYGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_sigmaxy[iGpu], dev_p0_sigmaxy[iGpu], its, it2-1, dev_sourcesPositionRegXYGrid[iGpu], nSourcesRegXYGrid));
		// Myz
		kernel_exec(ker_inject_source_yzGrid_3D<<<nblockSouYZGrid, BLOCK_SIZE_DATA>>>(dev_modelRegDts_sigmayz[iGpu], dev_p0_sigmayz[iGpu], its, it2-1, dev_sourcesPositionRegYZGrid[iGpu], nSourcesRegYZGrid));

}

void launchDampCosineEdgeKernels_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu){
		kernel_exec(dampCosineEdge_3D<<<dimGrid, dimBlock>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vy[iGpu], dev_p1_vy[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p1_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p1_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_sigmayz[iGpu], nx, ny, nz));
}

void launchFwdRecordInterpDataKernels_3D(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataYGrid, int nblockDataZGrid, int nblockDataXZGrid, int nblockDataXYGrid, int nblockDataYZGrid, long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int its, int it2, int iGpu){

		// Sxx, Syy, Szz
		kernel_exec(ker_record_interp_data_centerGrid_3D<<<nblockDataCenterGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmayy[iGpu], dev_dataRegDts_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu], nReceiversRegCenterGrid));
		// Vx
		kernel_exec(ker_record_interp_data_xGrid_3D<<<nblockDataXGrid, BLOCK_SIZE_DATA>>>(dev_p0_vx[iGpu], dev_dataRegDts_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu], nReceiversRegXGrid));
		// Vy
		kernel_exec(ker_record_interp_data_yGrid_3D<<<nblockDataYGrid, BLOCK_SIZE_DATA>>>(dev_p0_vy[iGpu], dev_dataRegDts_vy[iGpu], its, it2, dev_receiversPositionRegYGrid[iGpu], nReceiversRegYGrid));
		// Vz
		kernel_exec(ker_record_interp_data_zGrid_3D<<<nblockDataZGrid, BLOCK_SIZE_DATA>>>(dev_p0_vz[iGpu], dev_dataRegDts_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu], nReceiversRegZGrid));
		// Sxz
		kernel_exec(ker_record_interp_data_xzGrid_3D<<<nblockDataXZGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxz[iGpu], dev_dataRegDts_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu], nReceiversRegXZGrid));
		// Sxy
		kernel_exec(ker_record_interp_data_xyGrid_3D<<<nblockDataXYGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmaxy[iGpu], dev_dataRegDts_sigmaxy[iGpu], its, it2, dev_receiversPositionRegXYGrid[iGpu], nReceiversRegXYGrid));
		// Syz
		kernel_exec(ker_record_interp_data_yzGrid_3D<<<nblockDataYZGrid, BLOCK_SIZE_DATA>>>(dev_p0_sigmayz[iGpu], dev_dataRegDts_sigmayz[iGpu], its, it2, dev_receiversPositionRegYZGrid[iGpu], nReceiversRegYZGrid));
}


/************************************************/
// 				 Kernel launching ADJ functions
/************************************************/



/************************************************/
// 				 Interface functions
/************************************************/


/****************************************************************************************/
/******************************* Nonlinear forward propagation **************************/
/****************************************************************************************/
void propElasticFwdGpu_3D(double *modelRegDts_vx, double *modelRegDts_vy, double *modelRegDts_vz, double *modelRegDts_sigmaxx, double *modelRegDts_sigmayy, double *modelRegDts_sigmazz, double *modelRegDts_sigmaxz, double *modelRegDts_sigmaxy, double *modelRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, int iGpu, int iGpuId) {


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
		//
		// // Extraction grid size
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

		long long tmp_nModel = nz;
		tmp_nModel *= nx*ny;
		double * temp = new double[tmp_nModel];

		std::cout << "tmp_nModel = " << tmp_nModel << std::endl;
		std::cout << "nx = " << nx << std::endl;
		std::cout << "ny = " << ny << std::endl;
		std::cout << "nz = " << nz << std::endl;
		std::cout << "nblockx = " << nblockx << std::endl;
		std::cout << "nblocky = " << nblocky << std::endl;

		// Start propagation
		for (int its = 0; its < host_nts-1; its++){
				// std::cout << "its = " << its << std::endl;
				for (int it2 = 1; it2 < host_sub+1; it2++){
						// Compute fine time-step index
						int itw = its * host_sub + it2;

							// Step forward
							launchFwdStepKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu);
							// std::cout << "here2 itw = " << itw << std::endl;
							// cuda_call(cudaMemcpy(temp, dev_p0_vx[iGpu], tmp_nModel*sizeof(double), cudaMemcpyDeviceToHost));
							// std::cout << "Max vx = " << *std::max_element(temp,temp+tmp_nModel) << std::endl;
							// std::cout << "Min vx = " << *std::min_element(temp,temp+tmp_nModel) << std::endl;
							// cuda_call(cudaMemcpy(temp, dev_p0_vy[iGpu], tmp_nModel*sizeof(double), cudaMemcpyDeviceToHost));
							// std::cout << "Max vy = " << *std::max_element(temp,temp+tmp_nModel) << std::endl;
							// std::cout << "Min vy = " << *std::min_element(temp,temp+tmp_nModel) << std::endl;
							// cuda_call(cudaMemcpy(temp, dev_p0_vz[iGpu], tmp_nModel*sizeof(double), cudaMemcpyDeviceToHost));
							// std::cout << "Max vz = " << *std::max_element(temp,temp+tmp_nModel) << std::endl;
							// std::cout << "Min vz = " << *std::min_element(temp,temp+tmp_nModel) << std::endl;
							// cuda_call(cudaMemcpy(temp, dev_p0_sigmaxx[iGpu], tmp_nModel*sizeof(double), cudaMemcpyDeviceToHost));
							// std::cout << "Max sigmaxx = " << *std::max_element(temp,temp+tmp_nModel) << std::endl;
							// std::cout << "Min sigmaxx = " << *std::min_element(temp,temp+tmp_nModel) << std::endl;
							// cuda_call(cudaMemcpy(temp, dev_p0_sigmazz[iGpu], tmp_nModel*sizeof(double), cudaMemcpyDeviceToHost));
							// std::cout << "Max sigmazz = " << *std::max_element(temp,temp+tmp_nModel) << std::endl;
							// std::cout << "Min sigmazz = " << *std::min_element(temp,temp+tmp_nModel) << std::endl;
							// cuda_call(cudaMemcpy(temp, dev_p0_sigmaxz[iGpu], tmp_nModel*sizeof(double), cudaMemcpyDeviceToHost));
							// std::cout << "Max sigmaxz = " << *std::max_element(temp,temp+tmp_nModel) << std::endl;
							// std::cout << "Min sigmaxz = " << *std::min_element(temp,temp+tmp_nModel) << std::endl;
							// cuda_call(cudaMemcpy(temp, dev_p0_sigmaxy[iGpu], tmp_nModel*sizeof(double), cudaMemcpyDeviceToHost));
							// std::cout << "Max sigmaxy = " << *std::max_element(temp,temp+tmp_nModel) << std::endl;
							// std::cout << "Min sigmaxy = " << *std::min_element(temp,temp+tmp_nModel) << std::endl;
							// cuda_call(cudaMemcpy(temp, dev_p0_sigmayz[iGpu], tmp_nModel*sizeof(double), cudaMemcpyDeviceToHost));
							// std::cout << "Max sigmayz = " << *std::max_element(temp,temp+tmp_nModel) << std::endl;
							// std::cout << "Min sigmayz = " << *std::min_element(temp,temp+tmp_nModel) << std::endl;
							// Inject source
							launchFwdInjectSourceKernels_3D(nblockSouCenterGrid, nblockSouXGrid, nblockSouYGrid, nblockSouZGrid,nblockSouXZGrid, nblockSouXYGrid, nblockSouYZGrid, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, its, it2, iGpu);
							// std::cout << "here3" << std::endl;
							// Damp wavefields
							// launchDampCosineEdgeKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu);
							// std::cout << "here4" << std::endl;
							// Extract and interpolate data
							launchFwdRecordInterpDataKernels_3D(nblockDataCenterGrid, nblockDataXGrid, nblockDataYGrid, nblockDataZGrid, nblockDataXZGrid, nblockDataXYGrid, nblockDataYZGrid, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, its, it2, iGpu);
							// Switch pointers
							switchPointers_3D(iGpu);
							// std::cout << "here5" << std::endl;

				}
		}

		// std::cout << "here6" << std::endl;

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

		// std::cout << "here7" << std::endl;
		//
		// // Deallocate all slices
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

		// std::cout << "here8" << std::endl;

}







//
