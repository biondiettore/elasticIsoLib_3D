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
		cuda_call(cudaMemcpyToSymbol(dev_nz, &nz, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy model size to device
		cuda_call(cudaMemcpyToSymbol(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_ny, &ny, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
		cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}

void allocateNonlinearElasticGpu_3D(double *rhoxDtw, double *rhoyDtw, double *rhozDtw, double *lamb2MuDt, double *lambDtw, double *muxzDtw, double *muxyDtw, double *muyzDtw, int nx, int ny, int nz, int iGpu, int iGpuId){

		// Get GPU number
		cudaSetDevice(iGpuId);

		unsigned long long nModel = nz;
		nModel *= nx * nz;

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
		cuda_call(cudaMemcpy(dev_lamb2MuDtw[iGpu], lamb2MuDt, nModel*sizeof(double), cudaMemcpyHostToDevice));
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
