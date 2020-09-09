#include <cstring>
#include <iostream>
#include "BornElasticGpuFunctions_3D.h"
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
void initBornElasticGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

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

				// Source
				dev_sourceRegDts_vx = new double*[nGpu];
				dev_sourceRegDts_vy = new double*[nGpu];
				dev_sourceRegDts_vz = new double*[nGpu];
				dev_sourceRegDts_sigmaxx = new double*[nGpu];
				dev_sourceRegDts_sigmayy = new double*[nGpu];
				dev_sourceRegDts_sigmazz = new double*[nGpu];
				dev_sourceRegDts_sigmaxz = new double*[nGpu];
				dev_sourceRegDts_sigmaxy = new double*[nGpu];
				dev_sourceRegDts_sigmayz = new double*[nGpu];
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

				//Pointers specific to Born operator

				// Perturbation
				dev_drhox = new double*[nGpu];
				dev_drhoy = new double*[nGpu];
        dev_drhoz = new double*[nGpu];
        dev_dlame = new double*[nGpu];
        dev_dmu   = new double*[nGpu];
        dev_dmuxz = new double*[nGpu];
				dev_dmuxy = new double*[nGpu];
				dev_dmuyz = new double*[nGpu];

				// Wavefield pointers to pinned memory (DONE BY allocatePinBornElasticGpu_3D)
				// host_pinned_wavefield_vx = new double*[nGpu];
				// host_pinned_wavefield_vy = new double*[nGpu];
				// host_pinned_wavefield_vz = new double*[nGpu];

				// Wavefield slices
        dev_ssVxLeft  = new double*[nGpu];
        dev_ssVxRight = new double*[nGpu];
				dev_ssVyLeft  = new double*[nGpu];
        dev_ssVyRight = new double*[nGpu];
        dev_ssVzLeft  = new double*[nGpu];
        dev_ssVzRight = new double*[nGpu];
        dev_ssSigmaxxLeft  = new double*[nGpu];
        dev_ssSigmaxxRight = new double*[nGpu];
				dev_ssSigmayyLeft  = new double*[nGpu];
        dev_ssSigmayyRight = new double*[nGpu];
        dev_ssSigmazzLeft  = new double*[nGpu];
        dev_ssSigmazzRight = new double*[nGpu];
        dev_ssSigmaxzLeft  = new double*[nGpu];
        dev_ssSigmaxzRight = new double*[nGpu];
				dev_ssSigmaxyLeft  = new double*[nGpu];
        dev_ssSigmaxyRight = new double*[nGpu];
				dev_ssSigmayzLeft  = new double*[nGpu];
        dev_ssSigmayzRight = new double*[nGpu];

				// Source wavefield slices
				dev_wavefieldVx_left = new double*[nGpu];
				dev_wavefieldVx_cur = new double*[nGpu];
				dev_wavefieldVx_right = new double*[nGpu];
				dev_wavefieldVy_left = new double*[nGpu];
				dev_wavefieldVy_cur = new double*[nGpu];
				dev_wavefieldVy_right = new double*[nGpu];
				dev_wavefieldVz_left = new double*[nGpu];
				dev_wavefieldVz_cur = new double*[nGpu];
				dev_wavefieldVz_right = new double*[nGpu];

				dev_pStream_Vx = new double*[nGpu];
				dev_pStream_Vy = new double*[nGpu];
				dev_pStream_Vz = new double*[nGpu];

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
		double inv_dts = 1.0/(2.0 * dts);
  	cuda_call(cudaMemcpyToSymbol(dev_dts_inv, &inv_dts, sizeof(double), 0, cudaMemcpyHostToDevice)); // Inverse of the time-source sampling

}

void allocatePinBornElasticGpu_3D(int nx, int ny, int nz, int nts, int nGpu, int iGpuId, int iGpu, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	unsigned long long nModel = nz;
	nModel *= nx * ny * nts;

	/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {
		// Wavefield pointers to pinned memory (DONE BY A SEPARATE FUNCTION)
		host_pinned_wavefield_vx = new double*[nGpu];
		host_pinned_wavefield_vy = new double*[nGpu];
		host_pinned_wavefield_vz = new double*[nGpu];
	}

	cuda_call(cudaHostAlloc((void**) &host_pinned_wavefield_vx[iGpu], nModel*sizeof(double), cudaHostAllocDefault)); //Pinned memory on the Host
	cuda_call(cudaHostAlloc((void**) &host_pinned_wavefield_vy[iGpu], nModel*sizeof(double), cudaHostAllocDefault)); //Pinned memory on the Host
	cuda_call(cudaHostAlloc((void**) &host_pinned_wavefield_vz[iGpu], nModel*sizeof(double), cudaHostAllocDefault)); //Pinned memory on the Host

}

void deallocatePinBornElasticGpu_3D(int iGpu, int iGpuId){
	cudaSetDevice(iGpuId); // Set device number on GPU cluster
	//Deallocating pin memory
	cuda_call(cudaFreeHost(host_pinned_wavefield_vx[iGpu]));
	cuda_call(cudaFreeHost(host_pinned_wavefield_vy[iGpu]));
	cuda_call(cudaFreeHost(host_pinned_wavefield_vz[iGpu]));
}

// Allocate model paramaters and propagation slices
void allocateBornElasticGpu_3D(double *rhoxDtw, double *rhoyDtw, double *rhozDtw, double *lamb2MuDtw, double *lambDtw, double *muxzDtw, double *muxyDtw, double *muyzDtw, int nx, int ny, int nz, int iGpu, int iGpuId){

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

		// Model perturbations
		cuda_call(cudaMalloc((void**) &dev_drhox[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_drhoy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_drhoz[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_dlame[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_dmu[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_dmuxz[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_dmuxy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_dmuyz[iGpu], nModel*sizeof(double)));

		// Wavefield slices
		cuda_call(cudaMalloc((void**) &dev_ssVxLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssVxRight[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssVyLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssVyRight[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssVzLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssVzRight[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmaxxLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmaxxRight[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmayyLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmayyRight[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmazzLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmazzRight[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmaxzLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmaxzRight[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmaxyLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmaxyRight[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmayzLeft[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_ssSigmayzRight[iGpu], nModel*sizeof(double)));

		cuda_call(cudaMalloc((void**) &dev_wavefieldVx_left[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_wavefieldVx_cur[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_wavefieldVx_right[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_wavefieldVy_left[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_wavefieldVy_cur[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_wavefieldVy_right[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_wavefieldVz_left[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_wavefieldVz_cur[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_wavefieldVz_right[iGpu], nModel*sizeof(double)));

		cuda_call(cudaMalloc((void**) &dev_pStream_Vx[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_pStream_Vy[iGpu], nModel*sizeof(double)));
		cuda_call(cudaMalloc((void**) &dev_pStream_Vz[iGpu], nModel*sizeof(double)));

}


void deallocateBornElasticGpu_3D(int iGpu, int iGpuId){
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
		cuda_call(cudaFree(dev_p0_sigmaxz[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmaxy[iGpu]));
		cuda_call(cudaFree(dev_p0_sigmayz[iGpu]));

		cuda_call(cudaFree(dev_p1_vx[iGpu]));
		cuda_call(cudaFree(dev_p1_vy[iGpu]));
		cuda_call(cudaFree(dev_p1_vz[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmaxx[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmayy[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmazz[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmaxz[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmaxy[iGpu]));
		cuda_call(cudaFree(dev_p1_sigmayz[iGpu]));

		cuda_call(cudaFree(dev_drhox[iGpu]));
		cuda_call(cudaFree(dev_drhoy[iGpu]));
		cuda_call(cudaFree(dev_drhoz[iGpu]));
		cuda_call(cudaFree(dev_dlame[iGpu]));
		cuda_call(cudaFree(dev_dmu[iGpu]));
		cuda_call(cudaFree(dev_dmuxz[iGpu]));
		cuda_call(cudaFree(dev_dmuxy[iGpu]));
		cuda_call(cudaFree(dev_dmuyz[iGpu]));

		cuda_call(cudaFree(dev_ssVxLeft[iGpu]));
		cuda_call(cudaFree(dev_ssVxRight[iGpu]));
		cuda_call(cudaFree(dev_ssVyLeft[iGpu]));
		cuda_call(cudaFree(dev_ssVyRight[iGpu]));
		cuda_call(cudaFree(dev_ssVzLeft[iGpu]));
		cuda_call(cudaFree(dev_ssVzRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxxLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxxRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmayyLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmayyRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmazzLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmazzRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxzLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxzRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxyLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmaxyRight[iGpu]));
		cuda_call(cudaFree(dev_ssSigmayzLeft[iGpu]));
		cuda_call(cudaFree(dev_ssSigmayzRight[iGpu]));

		// Source wavefield slices
		cuda_call(cudaFree(dev_wavefieldVx_left[iGpu]));
		cuda_call(cudaFree(dev_wavefieldVx_cur[iGpu]));
		cuda_call(cudaFree(dev_wavefieldVx_right[iGpu]));
		cuda_call(cudaFree(dev_wavefieldVy_left[iGpu]));
		cuda_call(cudaFree(dev_wavefieldVy_cur[iGpu]));
		cuda_call(cudaFree(dev_wavefieldVy_right[iGpu]));
		cuda_call(cudaFree(dev_wavefieldVz_left[iGpu]));
		cuda_call(cudaFree(dev_wavefieldVz_cur[iGpu]));
		cuda_call(cudaFree(dev_wavefieldVz_right[iGpu]));

		cuda_call(cudaFree(dev_pStream_Vx[iGpu]));
		cuda_call(cudaFree(dev_pStream_Vy[iGpu]));
		cuda_call(cudaFree(dev_pStream_Vz[iGpu]));

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
void sourceAllocateGpu_3D(long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int iGpu){
		// fx
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_vx[iGpu], nSourcesRegXGrid*host_nts*sizeof(double)));
		// fy
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_vy[iGpu], nSourcesRegYGrid*host_nts*sizeof(double)));
		// fz
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_vz[iGpu], nSourcesRegZGrid*host_nts*sizeof(double)));
		// mxx
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_sigmaxx[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// myy
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_sigmayy[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// mzz
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_sigmazz[iGpu], nSourcesRegCenterGrid*host_nts*sizeof(double)));
		// mxz
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_sigmaxz[iGpu], nSourcesRegXZGrid*host_nts*sizeof(double)));
		// mxy
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_sigmaxy[iGpu], nSourcesRegXYGrid*host_nts*sizeof(double)));
		// myz
		cuda_call(cudaMalloc((void**) &dev_sourceRegDts_sigmayz[iGpu], nSourcesRegYZGrid*host_nts*sizeof(double)));
}
//copy source-model signals from host to device
void sourceCopyToGpu_3D(double *sourceRegDts_vx, double *sourceRegDts_vy, double *sourceRegDts_vz, double *sourceRegDts_sigmaxx, double *sourceRegDts_sigmayy, double *sourceRegDts_sigmazz, double *sourceRegDts_sigmaxz, double *sourceRegDts_sigmaxy, double *sourceRegDts_sigmayz, long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int iGpu){
		// fx
		cuda_call(cudaMemcpy(dev_sourceRegDts_vx[iGpu], sourceRegDts_vx, nSourcesRegXGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// fy
		cuda_call(cudaMemcpy(dev_sourceRegDts_vy[iGpu], sourceRegDts_vy, nSourcesRegYGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// fz
		cuda_call(cudaMemcpy(dev_sourceRegDts_vz[iGpu], sourceRegDts_vz, nSourcesRegZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// mxx
		cuda_call(cudaMemcpy(dev_sourceRegDts_sigmaxx[iGpu], sourceRegDts_sigmaxx, nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// myy
		cuda_call(cudaMemcpy(dev_sourceRegDts_sigmayy[iGpu], sourceRegDts_sigmayy, nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// mzz
		cuda_call(cudaMemcpy(dev_sourceRegDts_sigmazz[iGpu], sourceRegDts_sigmazz, nSourcesRegCenterGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// mxz
		cuda_call(cudaMemcpy(dev_sourceRegDts_sigmaxz[iGpu], sourceRegDts_sigmaxz, nSourcesRegXZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// mxy
		cuda_call(cudaMemcpy(dev_sourceRegDts_sigmaxy[iGpu], sourceRegDts_sigmaxy, nSourcesRegXYGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
		// myz
		cuda_call(cudaMemcpy(dev_sourceRegDts_sigmayz[iGpu], sourceRegDts_sigmayz, nSourcesRegYZGrid*host_nts*sizeof(double), cudaMemcpyHostToDevice));
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

void pinnedWavefieldInitializeGpu_3D(long long nModel, int iGpu){
	cuda_call(cudaMemset(host_pinned_wavefield_vx[iGpu], 0, nModel*host_nts*sizeof(double))); // Initialize pinned memory
	cuda_call(cudaMemset(host_pinned_wavefield_vy[iGpu], 0, nModel*host_nts*sizeof(double))); // Initialize pinned memory
	cuda_call(cudaMemset(host_pinned_wavefield_vz[iGpu], 0, nModel*host_nts*sizeof(double))); // Initialize pinned memory
}

void modelCopyToGpu_3D(double *drhox, double *drhoy, double *drhoz, double *dlame, double *dmu, double *dmuxz, double *dmuxy, double *dmuyz, long long nModel, int iGpu){
		cuda_call(cudaMemcpy(dev_drhox[iGpu], drhox, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_drhoy[iGpu], drhoy, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_drhoz[iGpu], drhoz, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dlame[iGpu], dlame, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dmu[iGpu], dmu, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dmuxz[iGpu], dmuxz, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dmuxy[iGpu], dmuxy, nModel*sizeof(double), cudaMemcpyHostToDevice));
		cuda_call(cudaMemcpy(dev_dmuyz[iGpu], dmuyz, nModel*sizeof(double), cudaMemcpyHostToDevice));
}

void modelSetOnGpu_3D(long long nModel, int iGpu){
  cuda_call(cudaMemset(dev_drhox[iGpu], 0, nModel*sizeof(double)));
  cuda_call(cudaMemset(dev_drhoy[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_drhoz[iGpu], 0, nModel*sizeof(double)));
  cuda_call(cudaMemset(dev_dlame[iGpu], 0, nModel*sizeof(double)));
  cuda_call(cudaMemset(dev_dmu[iGpu], 0, nModel*sizeof(double)));
  cuda_call(cudaMemset(dev_dmuxz[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_dmuxy[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_dmuyz[iGpu], 0, nModel*sizeof(double)));
}

void wavefieldSliceInitializeGpu_3D(long long nModel, int iGpu){
	cuda_call(cudaMemset(dev_wavefieldVx_left[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_wavefieldVx_right[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_wavefieldVy_left[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_wavefieldVy_right[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_wavefieldVz_left[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_wavefieldVz_right[iGpu], 0, nModel*sizeof(double)));
}

void SecondarySourceInitializeOnGpu_3D(long long nModel, int iGpu){
	//Born specific slices
	cuda_call(cudaMemset(dev_ssVxLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssVxRight[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssVyLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssVzRight[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssVzLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssVzRight[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmaxxLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmaxxRight[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmayyLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmayyRight[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmazzLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmazzRight[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmaxzLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmaxzRight[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmaxyLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmaxyRight[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmayzLeft[iGpu], 0, nModel*sizeof(double)));
	cuda_call(cudaMemset(dev_ssSigmayzRight[iGpu], 0, nModel*sizeof(double)));
}

//setup: a) src and receiver positions allocation and copying to device
//       b) allocate and copy source (arrays for sources for each wavefield) to device
//       c) allocate and initialize (0) data (receiver-recording arrays) on device
//       d) allocate and initialize (0) wavefield time slices on gpu
void setupFwdGpu_3D(double *drhox, double *drhoy, double *drhoz, double *dlame, double *dmu, double *dmuxz, double *dmuxy, double *dmuyz, double *sourceRegDts_vx, double *sourceRegDts_vy, double *sourceRegDts_vz, double *sourceRegDts_sigmaxx, double *sourceRegDts_sigmayy, double *sourceRegDts_sigmazz, double *sourceRegDts_sigmaxz, double *sourceRegDts_sigmaxy, double *sourceRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, int iGpu, int iGpuId){
		// Set device number on GPU cluster
		cudaSetDevice(iGpuId);

		//allocate and copy src and rec geometry to gpu
		srcRecAllocateAndCopyToGpu_3D(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, iGpu);

		// source - wavelets for each wavefield component. Allocate and copy to gpu
		sourceAllocateGpu_3D(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);
		sourceCopyToGpu_3D(sourceRegDts_vx, sourceRegDts_vy, sourceRegDts_vz, sourceRegDts_sigmaxx, sourceRegDts_sigmayy, sourceRegDts_sigmazz, sourceRegDts_sigmaxz, sourceRegDts_sigmaxy, sourceRegDts_sigmayz, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);

		// Data - data recordings for each wavefield component. Allocate and initialize on gpu
		dataAllocateGpu_3D(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);
		dataInitializeOnGpu_3D(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);

		//initialize wavefield slices to zero
		long long nModel = nz;
		nModel *= nx * ny;
		wavefieldInitializeOnGpu_3D(nModel, iGpu);
		pinnedWavefieldInitializeGpu_3D(nModel, iGpu);
		modelCopyToGpu_3D(drhox, drhoy, drhoz, dlame, dmu, dmuxz, dmuxy, dmuyz, nModel, iGpu);

}

//setup: a) src and receiver positions allocation and copying to device
//       b) allocate and copy source (arrays for sources for each wavefield) to device
//       c) allocate and copy data (receiver-recording arrays) to device
//       d) allocate and initialize (0) wavefield time slices to gpu
void setupAdjGpu_3D(double *sourceRegDts_vx, double *sourceRegDts_vy, double *sourceRegDts_vz, double *sourceRegDts_sigmaxx, double *sourceRegDts_sigmayy, double *sourceRegDts_sigmazz, double *sourceRegDts_sigmaxz, double *sourceRegDts_sigmaxy, double *sourceRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, int iGpu, int iGpuId){
		// Set device number on GPU cluster
		cudaSetDevice(iGpuId);

		//allocate and copy src and rec geometry to gpu
		srcRecAllocateAndCopyToGpu_3D(sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, iGpu);

		// source - wavelets for each wavefield component. Allocate and copy to gpu
		sourceAllocateGpu_3D(nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);
		sourceCopyToGpu_3D(sourceRegDts_vx, sourceRegDts_vy, sourceRegDts_vz, sourceRegDts_sigmaxx, sourceRegDts_sigmayy, sourceRegDts_sigmazz, sourceRegDts_sigmaxz, sourceRegDts_sigmaxy, sourceRegDts_sigmayz, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, iGpu);

		// Data - data recordings for each wavefield component. Allocate and initialize on gpu
		dataAllocateGpu_3D(nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);
		dataCopyToGpu_3D(dataRegDts_vx, dataRegDts_vy, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmayy, dataRegDts_sigmazz, dataRegDts_sigmaxz, dataRegDts_sigmaxy, dataRegDts_sigmayz, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, iGpu);

		//initialize wavefield slices to zero
		long long nModel = nz;
		nModel *= nx * ny;
		wavefieldInitializeOnGpu_3D(nModel, iGpu);
		pinnedWavefieldInitializeGpu_3D(nModel, iGpu);
		modelSetOnGpu_3D(nModel, iGpu);
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

void switchPointersSecondarySource_3D(int iGpu){
		dev_temp1[iGpu] 	= dev_ssVxLeft[iGpu];
		dev_ssVxLeft[iGpu] 	= dev_ssVxRight[iGpu];
		dev_ssVxRight[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] 	= dev_ssVyLeft[iGpu];
		dev_ssVyLeft[iGpu] 	= dev_ssVyRight[iGpu];
		dev_ssVyRight[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] 	= dev_ssVzLeft[iGpu];
		dev_ssVzLeft[iGpu] 	= dev_ssVzRight[iGpu];
		dev_ssVzRight[iGpu] = dev_temp1[iGpu];

		dev_temp1[iGpu] 		= dev_ssSigmaxxLeft[iGpu];
		dev_ssSigmaxxLeft[iGpu] = dev_ssSigmaxxRight[iGpu];
		dev_ssSigmaxxRight[iGpu]= dev_temp1[iGpu];

		dev_temp1[iGpu] 		= dev_ssSigmayyLeft[iGpu];
		dev_ssSigmayyLeft[iGpu] = dev_ssSigmayyRight[iGpu];
		dev_ssSigmayyRight[iGpu]= dev_temp1[iGpu];

		dev_temp1[iGpu] 		= dev_ssSigmazzLeft[iGpu];
		dev_ssSigmazzLeft[iGpu] = dev_ssSigmazzRight[iGpu];
		dev_ssSigmazzRight[iGpu]= dev_temp1[iGpu];

		dev_temp1[iGpu] 		= dev_ssSigmaxzLeft[iGpu];
		dev_ssSigmaxzLeft[iGpu] = dev_ssSigmaxzRight[iGpu];
		dev_ssSigmaxzRight[iGpu]= dev_temp1[iGpu];

		dev_temp1[iGpu] 		= dev_ssSigmaxyLeft[iGpu];
		dev_ssSigmaxyLeft[iGpu] = dev_ssSigmaxyRight[iGpu];
		dev_ssSigmaxyRight[iGpu]= dev_temp1[iGpu];

		dev_temp1[iGpu] 		= dev_ssSigmayzLeft[iGpu];
		dev_ssSigmayzLeft[iGpu] = dev_ssSigmayzRight[iGpu];
		dev_ssSigmayzRight[iGpu]= dev_temp1[iGpu];

		dev_temp1[iGpu] = NULL;
}

void switchPointers_wavefield2Slices_3D(int iGpu){
	dev_temp1[iGpu] = dev_wavefieldVx_left[iGpu];
	dev_wavefieldVx_left[iGpu] = dev_wavefieldVx_right[iGpu];
	dev_wavefieldVx_right[iGpu] = dev_temp1[iGpu];
	dev_temp1[iGpu] = dev_wavefieldVy_left[iGpu];
	dev_wavefieldVy_left[iGpu] = dev_wavefieldVy_right[iGpu];
	dev_wavefieldVy_right[iGpu] = dev_temp1[iGpu];
	dev_temp1[iGpu] = dev_wavefieldVz_left[iGpu];
	dev_wavefieldVz_left[iGpu] = dev_wavefieldVz_right[iGpu];
	dev_wavefieldVz_right[iGpu] = dev_temp1[iGpu];
	dev_temp1[iGpu] = NULL;
}

void switchPointers_wavefield3Slices_3D(int iGpu){
	//Vx temporary slices
	dev_temp1[iGpu] = dev_wavefieldVx_left[iGpu];
	dev_wavefieldVx_left[iGpu] = dev_wavefieldVx_cur[iGpu];
	dev_wavefieldVx_cur[iGpu] = dev_wavefieldVx_right[iGpu];
	dev_wavefieldVx_right[iGpu] = dev_temp1[iGpu];
	//Vy temporary slices
	dev_temp1[iGpu] = dev_wavefieldVy_left[iGpu];
	dev_wavefieldVy_left[iGpu] = dev_wavefieldVy_cur[iGpu];
	dev_wavefieldVy_cur[iGpu] = dev_wavefieldVy_right[iGpu];
	dev_wavefieldVy_right[iGpu] = dev_temp1[iGpu];
	//Vz temporary slices
	dev_temp1[iGpu] = dev_wavefieldVz_left[iGpu];
	dev_wavefieldVz_left[iGpu] = dev_wavefieldVz_cur[iGpu];
	dev_wavefieldVz_cur[iGpu] = dev_wavefieldVz_right[iGpu];
	dev_wavefieldVz_right[iGpu] = dev_temp1[iGpu];
	dev_temp1[iGpu] = NULL;
}

void switchPointers_wavefield3Slices_adj_3D(int iGpu){
	//Vx temporary slices
	dev_temp1[iGpu] = dev_wavefieldVx_right[iGpu];
	dev_wavefieldVx_right[iGpu] = dev_wavefieldVx_cur[iGpu];
	dev_wavefieldVx_cur[iGpu] = dev_wavefieldVx_left[iGpu];
	dev_wavefieldVx_left[iGpu] = dev_temp1[iGpu];
	//Vy temporary slices
	dev_temp1[iGpu] = dev_wavefieldVy_right[iGpu];
	dev_wavefieldVy_right[iGpu] = dev_wavefieldVy_cur[iGpu];
	dev_wavefieldVy_cur[iGpu] = dev_wavefieldVy_left[iGpu];
	dev_wavefieldVy_left[iGpu] = dev_temp1[iGpu];
	//Vz temporary slices
	dev_temp1[iGpu] = dev_wavefieldVz_right[iGpu];
	dev_wavefieldVz_right[iGpu] = dev_wavefieldVz_cur[iGpu];
	dev_wavefieldVz_cur[iGpu] = dev_wavefieldVz_left[iGpu];
	dev_wavefieldVz_left[iGpu] = dev_temp1[iGpu];
	dev_temp1[iGpu] = NULL;
}

/************************************************/
// 				 Kernel launching FWD functions
/************************************************/
void launchFwdStepKernels_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu, cudaStream_t stream){
		kernel_stream_exec(stepFwdGpu_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_vx[iGpu], dev_p1_vy[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmayy[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p1_sigmaxy[iGpu], dev_p1_sigmayz[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_rhoxDtw[iGpu], dev_rhoyDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu], dev_muxyDtw[iGpu], dev_muyzDtw[iGpu], nx, ny, nz));
}

void launchFwdInjectSourceKernels_3D(int nblockSouCenterGrid, int nblockSouXGrid, int nblockSouYGrid, int nblockSouZGrid, int nblockSouXZGrid, int nblockSouXYGrid, int nblockSouYZGrid, long long nSourcesRegCenterGrid, long long nSourcesRegXGrid, long long nSourcesRegYGrid, long long nSourcesRegZGrid, long long nSourcesRegXZGrid, long long nSourcesRegXYGrid, long long nSourcesRegYZGrid, int its, int it2, int iGpu, cudaStream_t stream){
		// Mxx, Myy, Mzz
		kernel_stream_exec(ker_inject_source_centerGrid_3D<<<nblockSouCenterGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_sourceRegDts_sigmaxx[iGpu], dev_sourceRegDts_sigmayy[iGpu], dev_sourceRegDts_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], its, it2-1, dev_sourcesPositionRegCenterGrid[iGpu], nSourcesRegCenterGrid));
		// fx
		kernel_stream_exec(ker_inject_source_stagGrid_3D<<<nblockSouXGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_sourceRegDts_vx[iGpu], dev_p0_vx[iGpu], its, it2-1, dev_sourcesPositionRegXGrid[iGpu], nSourcesRegXGrid));
		// fy
		kernel_stream_exec(ker_inject_source_stagGrid_3D<<<nblockSouYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_sourceRegDts_vy[iGpu], dev_p0_vy[iGpu], its, it2-1, dev_sourcesPositionRegYGrid[iGpu], nSourcesRegYGrid));
		// fz
		kernel_stream_exec(ker_inject_source_stagGrid_3D<<<nblockSouZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_sourceRegDts_vz[iGpu], dev_p0_vz[iGpu], its, it2-1, dev_sourcesPositionRegZGrid[iGpu], nSourcesRegZGrid));
		// Mxz
		kernel_stream_exec(ker_inject_source_stagGrid_3D<<<nblockSouXZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_sourceRegDts_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2-1, dev_sourcesPositionRegXZGrid[iGpu], nSourcesRegXZGrid));
		// Mxy
		kernel_stream_exec(ker_inject_source_stagGrid_3D<<<nblockSouXYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_sourceRegDts_sigmaxy[iGpu], dev_p0_sigmaxy[iGpu], its, it2-1, dev_sourcesPositionRegXYGrid[iGpu], nSourcesRegXYGrid));
		// Myz
		kernel_stream_exec(ker_inject_source_stagGrid_3D<<<nblockSouYZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_sourceRegDts_sigmayz[iGpu], dev_p0_sigmayz[iGpu], its, it2-1, dev_sourcesPositionRegYZGrid[iGpu], nSourcesRegYZGrid));

}

void launchDampCosineEdgeKernels_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu, cudaStream_t stream){
		kernel_stream_exec(dampCosineEdge_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_p0_vx[iGpu], dev_p1_vx[iGpu], dev_p0_vy[iGpu], dev_p1_vy[iGpu], dev_p0_vz[iGpu],  dev_p1_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p1_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p1_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p1_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p1_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p1_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_sigmayz[iGpu], nx, ny, nz));
}

void launchFwdRecordInterpDataKernels_3D(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataYGrid, int nblockDataZGrid, int nblockDataXZGrid, int nblockDataXYGrid, int nblockDataYZGrid, long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int its, int it2, int iGpu, cudaStream_t stream){

		// Sxx, Syy, Szz
		kernel_stream_exec(ker_record_interp_data_centerGrid_3D<<<nblockDataCenterGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmayy[iGpu], dev_dataRegDts_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu], nReceiversRegCenterGrid));
		// Vx
		kernel_stream_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataXGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_vx[iGpu], dev_dataRegDts_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu], nReceiversRegXGrid));
		// Vy
		kernel_stream_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_vy[iGpu], dev_dataRegDts_vy[iGpu], its, it2, dev_receiversPositionRegYGrid[iGpu], nReceiversRegYGrid));
		// Vz
		kernel_stream_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_vz[iGpu], dev_dataRegDts_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu], nReceiversRegZGrid));
		// Sxz
		kernel_stream_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataXZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_sigmaxz[iGpu], dev_dataRegDts_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu], nReceiversRegXZGrid));
		// Sxy
		kernel_stream_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataXYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_sigmaxy[iGpu], dev_dataRegDts_sigmaxy[iGpu], its, it2, dev_receiversPositionRegXYGrid[iGpu], nReceiversRegXYGrid));
		// Syz
		kernel_stream_exec(ker_record_interp_data_stagGrid_3D<<<nblockDataYZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_p0_sigmayz[iGpu], dev_dataRegDts_sigmayz[iGpu], its, it2, dev_receiversPositionRegYZGrid[iGpu], nReceiversRegYZGrid));
}

void launchInjectSecondarySource_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu, int it2, cudaStream_t stream){
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssVxLeft[iGpu], dev_ssVxRight[iGpu], dev_p0_vx[iGpu], nx, ny, nz, it2-1));
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssVyLeft[iGpu], dev_ssVyRight[iGpu], dev_p0_vy[iGpu], nx, ny, nz, it2-1));
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssVzLeft[iGpu], dev_ssVzRight[iGpu], dev_p0_vz[iGpu], nx, ny, nz, it2-1));
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmaxxLeft[iGpu], dev_ssSigmaxxRight[iGpu], dev_p0_sigmaxx[iGpu], nx, ny, nz, it2-1));
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmayyLeft[iGpu], dev_ssSigmayyRight[iGpu], dev_p0_sigmayy[iGpu], nx, ny, nz, it2-1));
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmazzLeft[iGpu], dev_ssSigmazzRight[iGpu], dev_p0_sigmazz[iGpu], nx, ny, nz, it2-1));
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmaxzLeft[iGpu], dev_ssSigmaxzRight[iGpu], dev_p0_sigmaxz[iGpu], nx, ny, nz, it2-1));
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmaxyLeft[iGpu], dev_ssSigmaxyRight[iGpu], dev_p0_sigmaxy[iGpu], nx, ny, nz, it2-1));
	kernel_stream_exec(injectSecondarySource_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmayzLeft[iGpu], dev_ssSigmayzRight[iGpu], dev_p0_sigmayz[iGpu], nx, ny, nz, it2-1));
}

/************************************************/
// 				 Kernel launching ADJ functions
/************************************************/
void launchAdjStepKernels_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu, cudaStream_t stream){
		kernel_stream_exec(stepAdjGpu_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_p1_vx[iGpu], dev_p1_vy[iGpu], dev_p1_vz[iGpu], dev_p1_sigmaxx[iGpu], dev_p1_sigmayy[iGpu], dev_p1_sigmazz[iGpu], dev_p1_sigmaxz[iGpu], dev_p1_sigmaxy[iGpu], dev_p1_sigmayz[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], dev_p0_sigmaxz[iGpu], dev_p0_sigmaxy[iGpu], dev_p0_sigmayz[iGpu], dev_rhoxDtw[iGpu], dev_rhoyDtw[iGpu], dev_rhozDtw[iGpu], dev_lamb2MuDtw[iGpu], dev_lambDtw[iGpu], dev_muxzDtw[iGpu], dev_muxyDtw[iGpu], dev_muyzDtw[iGpu], nx, ny, nz));
}
void launchAdjInterpInjectDataKernels_3D(int nblockDataCenterGrid, int nblockDataXGrid, int nblockDataYGrid, int nblockDataZGrid, int nblockDataXZGrid, int nblockDataXYGrid, int nblockDataYZGrid, long long nReceiversRegCenterGrid, long long nReceiversRegXGrid, long long nReceiversRegYGrid, long long nReceiversRegZGrid, long long nReceiversRegXZGrid, long long nReceiversRegXYGrid, long long nReceiversRegYZGrid, int its, int it2, int iGpu, cudaStream_t stream){
	// Sxx, Syy, Szz
	kernel_stream_exec(ker_inject_data_centerGrid_3D<<<nblockDataCenterGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_dataRegDts_sigmaxx[iGpu], dev_dataRegDts_sigmayy[iGpu], dev_dataRegDts_sigmazz[iGpu], dev_p0_sigmaxx[iGpu], dev_p0_sigmayy[iGpu], dev_p0_sigmazz[iGpu], its, it2, dev_receiversPositionRegCenterGrid[iGpu], nReceiversRegCenterGrid));
	// Vx
	kernel_stream_exec(ker_inject_data_stagGrid_3D<<<nblockDataXGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_dataRegDts_vx[iGpu], dev_p0_vx[iGpu], its, it2, dev_receiversPositionRegXGrid[iGpu], nReceiversRegXGrid));
	// Vy
	kernel_stream_exec(ker_inject_data_stagGrid_3D<<<nblockDataYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_dataRegDts_vy[iGpu], dev_p0_vy[iGpu], its, it2, dev_receiversPositionRegYGrid[iGpu], nReceiversRegYGrid));
	// Vz
	kernel_stream_exec(ker_inject_data_stagGrid_3D<<<nblockDataZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_dataRegDts_vz[iGpu], dev_p0_vz[iGpu], its, it2, dev_receiversPositionRegZGrid[iGpu], nReceiversRegZGrid));
	// Sxz
	kernel_stream_exec(ker_inject_data_stagGrid_3D<<<nblockDataXZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_dataRegDts_sigmaxz[iGpu], dev_p0_sigmaxz[iGpu], its, it2, dev_receiversPositionRegXZGrid[iGpu], nReceiversRegXZGrid));
	// Sxy
	kernel_stream_exec(ker_inject_data_stagGrid_3D<<<nblockDataXYGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_dataRegDts_sigmaxy[iGpu], dev_p0_sigmaxy[iGpu], its, it2, dev_receiversPositionRegXYGrid[iGpu], nReceiversRegXYGrid));
	// Syz
	kernel_stream_exec(ker_inject_data_stagGrid_3D<<<nblockDataYZGrid, BLOCK_SIZE_DATA, 0, stream>>>(dev_dataRegDts_sigmayz[iGpu], dev_p0_sigmayz[iGpu], its, it2, dev_receiversPositionRegYZGrid[iGpu], nReceiversRegYZGrid));

}

void launchExtractInterpAdjointWavefield_3D(dim3 dimGrid, dim3 dimBlock, int nx, int ny, int nz, int iGpu, int it2, cudaStream_t stream){
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssVxLeft[iGpu], dev_ssVxRight[iGpu], dev_p0_vx[iGpu], nx, ny, nz, it2));
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssVyLeft[iGpu], dev_ssVyRight[iGpu], dev_p0_vy[iGpu], nx, ny, nz, it2));
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssVzLeft[iGpu], dev_ssVzRight[iGpu], dev_p0_vz[iGpu], nx, ny, nz, it2));
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmaxxLeft[iGpu], dev_ssSigmaxxLeft[iGpu], dev_p0_sigmaxx[iGpu], nx, ny, nz, it2));
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmayyLeft[iGpu], dev_ssSigmayyLeft[iGpu], dev_p0_sigmayy[iGpu], nx, ny, nz, it2));
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmazzLeft[iGpu], dev_ssSigmazzLeft[iGpu], dev_p0_sigmazz[iGpu], nx, ny, nz, it2));
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmaxzLeft[iGpu], dev_ssSigmaxzLeft[iGpu], dev_p0_sigmaxz[iGpu], nx, ny, nz, it2));
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmaxyLeft[iGpu], dev_ssSigmaxyLeft[iGpu], dev_p0_sigmaxy[iGpu], nx, ny, nz, it2));
	kernel_stream_exec(extractInterpAdjointWavefield_3D<<<dimGrid, dimBlock, 0, stream>>>(dev_ssSigmayzLeft[iGpu], dev_ssSigmayzLeft[iGpu], dev_p0_sigmayz[iGpu], nx, ny, nz, it2));
}

/************************************************/
// 			Domain decompositions utily functions
/************************************************/


/************************************************/
// 				 Interface functions
/************************************************/


/****************************************************************************************/
/********************************* Born forward operator ********************************/
/****************************************************************************************/
void BornElasticFwdGpu_3D(double *sourceRegDts_vx, double *sourceRegDts_vy, double *sourceRegDts_vz, double *sourceRegDts_sigmaxx, double *sourceRegDts_sigmayy, double *sourceRegDts_sigmazz, double *sourceRegDts_sigmaxz, double *sourceRegDts_sigmaxy, double *sourceRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, double *drhox, double *drhoy, double *drhoz, double *dlame,  double *dmu, double *dmuxz, double *dmuxy, double *dmuyz, int nx, int ny, int nz, int iGpu, int iGpuId){

	//setup: a) src and receiver positions allocation and copying to device
	//       b) allocate and copy source (arrays for sources for each wavefield) to device
	//       c) allocate and initialize(0) data (recevier recordings arrays) to device
	//       d) allocate and copy wavefield time slices to gpu
	setupFwdGpu_3D(drhox, drhoy, drhoz, dlame, dmu, dmuxz, dmuxy, dmuyz, sourceRegDts_vx, sourceRegDts_vy, sourceRegDts_vz, sourceRegDts_sigmaxx, sourceRegDts_sigmayy, sourceRegDts_sigmazz, sourceRegDts_sigmaxz, sourceRegDts_sigmaxy, sourceRegDts_sigmayz, dataRegDts_vx, dataRegDts_vy, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmayy, dataRegDts_sigmazz, dataRegDts_sigmaxz, dataRegDts_sigmaxy, dataRegDts_sigmayz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, nx, ny, nz, iGpu, iGpuId);

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

	cudaStream_t compStream, transferStream;
	// Create streams for saving the wavefield
	cudaStreamCreate(&compStream);
	cudaStreamCreate(&transferStream);

	long long nModel = nx;
	nModel *= ny * nz;

	wavefieldSliceInitializeGpu_3D(nModel, iGpu);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
			for (int it2 = 1; it2 < host_sub+1; it2++){

				// Step forward
				launchFwdStepKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, compStream);
				// Inject source
				launchFwdInjectSourceKernels_3D(nblockSouCenterGrid, nblockSouXGrid, nblockSouYGrid, nblockSouZGrid,nblockSouXZGrid, nblockSouXYGrid, nblockSouYZGrid, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, its, it2, iGpu, compStream);
				// Damp wavefields
				launchDampCosineEdgeKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, compStream);
				// Extract wavefield
				kernel_stream_exec(interpWavefieldVxVyVz_3D<<<dimGrid, dimBlock, 0, compStream>>>(dev_wavefieldVx_left[iGpu], dev_wavefieldVx_right[iGpu], dev_wavefieldVy_left[iGpu], dev_wavefieldVy_right[iGpu], dev_wavefieldVz_left[iGpu], dev_wavefieldVz_right[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], nx, ny, nz, it2));
				// Switch pointers
				switchPointers_3D(iGpu);

			}

			/* Note: At that point pLeft [its] is ready to be transfered back to host */

			// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
			cuda_call(cudaStreamSynchronize(transferStream));

			// Asynchronous copy of dev_wavefieldDts_left => dev_pStream [its] [compute]
			cuda_call(cudaMemcpyAsync(dev_pStream_Vx[iGpu], dev_wavefieldVx_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
			cuda_call(cudaMemcpyAsync(dev_pStream_Vy[iGpu], dev_wavefieldVy_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
			cuda_call(cudaMemcpyAsync(dev_pStream_Vz[iGpu], dev_wavefieldVz_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));

			// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
			cuda_call(cudaStreamSynchronize(compStream));

			// Asynchronous transfer of pStream => pin [its] [transfer]
			// Launch the transfer while we compute the next coarse time sample
			cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vx[iGpu]+its*nModel, dev_pStream_Vx[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
			cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vy[iGpu]+its*nModel, dev_pStream_Vy[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
			cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vz[iGpu]+its*nModel, dev_pStream_Vz[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));

			// Switch pointers
			switchPointers_wavefield2Slices_3D(iGpu);
			// Reinitialize dev_wavefieldDts_right to zero
			cuda_call(cudaMemsetAsync(dev_wavefieldVx_right[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_wavefieldVy_right[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_wavefieldVz_right[iGpu], 0, nModel*sizeof(double), compStream));

	}
	// Note: At that point, dev_wavefieldDts_left contains the value of the wavefield at nts-1 (last time sample) and the wavefield only has values up to nts-2
	cuda_call(cudaStreamSynchronize(compStream));
	cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vx[iGpu]+(host_nts-1)*nModel, dev_wavefieldVx_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
	cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vy[iGpu]+(host_nts-1)*nModel, dev_wavefieldVy_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
	cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vz[iGpu]+(host_nts-1)*nModel, dev_wavefieldVz_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
	cuda_call(cudaStreamSynchronize(transferStream));

	/************************** Scattered wavefield computation ****************************/

	// Initialize time slices on device
	wavefieldInitializeOnGpu_3D(nModel, iGpu);
	SecondarySourceInitializeOnGpu_3D(nModel, iGpu);

	// Copy model perturbations to device (done within setupBornFwdGpu function)
	//Note the perturbations have been already scaled by the wave-equation source scaling factor outside of this function

	// Copy wavefield time-slice its = 0: pinned -> dev_pSourceWavefield_old
	cuda_call(cudaMemcpyAsync(dev_wavefieldVx_left[iGpu], host_pinned_wavefield_vx[iGpu], nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_wavefieldVy_left[iGpu], host_pinned_wavefield_vy[iGpu], nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_wavefieldVz_left[iGpu], host_pinned_wavefield_vz[iGpu], nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));

	// Copy wavefield time-slice its = 1: pinned -> dev_pSourceWavefield_cur
	cuda_call(cudaMemcpyAsync(dev_wavefieldVx_cur[iGpu], host_pinned_wavefield_vx[iGpu]+nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_wavefieldVy_cur[iGpu], host_pinned_wavefield_vy[iGpu]+nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_wavefieldVz_cur[iGpu], host_pinned_wavefield_vz[iGpu]+nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaStreamSynchronize(transferStream));

	kernel_stream_exec(imagingElaFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream>>>(NULL, dev_wavefieldVx_left[iGpu], dev_wavefieldVx_cur[iGpu], NULL, dev_wavefieldVy_left[iGpu], dev_wavefieldVy_cur[iGpu], NULL, dev_wavefieldVz_left[iGpu], dev_wavefieldVz_cur[iGpu], dev_ssVxLeft[iGpu], dev_ssVyLeft[iGpu], dev_ssVzLeft[iGpu], dev_ssSigmaxxLeft[iGpu], dev_ssSigmayyLeft[iGpu], dev_ssSigmazzLeft[iGpu], dev_ssSigmaxzLeft[iGpu], dev_ssSigmaxyLeft[iGpu], dev_ssSigmayzLeft[iGpu], dev_drhox[iGpu], dev_drhoy[iGpu], dev_drhoz[iGpu], dev_dlame[iGpu], dev_dmu[iGpu], dev_dmuxz[iGpu], dev_dmuxy[iGpu], dev_dmuyz[iGpu], nx, ny, nz, 0));

	// Copy new slice from pinned for time its = 2 -> transfer to pStream
	cuda_call(cudaMemcpyAsync(dev_pStream_Vx[iGpu], host_pinned_wavefield_vx[iGpu]+2*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_pStream_Vy[iGpu], host_pinned_wavefield_vy[iGpu]+2*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_pStream_Vz[iGpu], host_pinned_wavefield_vz[iGpu]+2*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaStreamSynchronize(transferStream));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){
		if (its < host_nts-2){
			// Copy wavefield value at its+1 from pStream -> pSourceWavefield
			cuda_call(cudaMemcpyAsync(dev_wavefieldVx_right[iGpu], dev_pStream_Vx[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
			cuda_call(cudaMemcpyAsync(dev_wavefieldVy_right[iGpu], dev_pStream_Vy[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
			cuda_call(cudaMemcpyAsync(dev_wavefieldVz_right[iGpu], dev_pStream_Vz[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
		}

		if (its < host_nts-3){
			// Copy wavefield slice its+2 from pinned > dev_pStream
			cuda_call(cudaStreamSynchronize(compStream));
			cuda_call(cudaMemcpyAsync(dev_pStream_Vx[iGpu], host_pinned_wavefield_vx[iGpu]+(its+3)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
			cuda_call(cudaMemcpyAsync(dev_pStream_Vy[iGpu], host_pinned_wavefield_vy[iGpu]+(its+3)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
			cuda_call(cudaMemcpyAsync(dev_pStream_Vz[iGpu], host_pinned_wavefield_vz[iGpu]+(its+3)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
		}


		// Compute secondary source for first coarse time index (its+1) with compute stream
		if (its == host_nts-2){
			// Last step must be different since left => (its-3), cur => (its-2), right => (its-1)
			kernel_stream_exec(imagingElaFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream>>>(dev_wavefieldVx_cur[iGpu], dev_wavefieldVx_right[iGpu], NULL, dev_wavefieldVy_cur[iGpu], dev_wavefieldVy_right[iGpu], NULL, dev_wavefieldVz_cur[iGpu], dev_wavefieldVz_right[iGpu], NULL, dev_ssVxRight[iGpu], dev_ssVyRight[iGpu], dev_ssVzRight[iGpu], dev_ssSigmaxxRight[iGpu], dev_ssSigmayyRight[iGpu], dev_ssSigmazzRight[iGpu], dev_ssSigmaxzRight[iGpu], dev_ssSigmaxyRight[iGpu], dev_ssSigmayzRight[iGpu], dev_drhox[iGpu], dev_drhoy[iGpu], dev_drhoz[iGpu], dev_dlame[iGpu], dev_dmu[iGpu], dev_dmuxz[iGpu], dev_dmuxy[iGpu], dev_dmuyz[iGpu], nx, ny, nz, its+1));
		} else {
			kernel_stream_exec(imagingElaFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream>>>(dev_wavefieldVx_left[iGpu], dev_wavefieldVx_cur[iGpu], dev_wavefieldVx_right[iGpu], dev_wavefieldVy_left[iGpu], dev_wavefieldVy_cur[iGpu], dev_wavefieldVy_right[iGpu], dev_wavefieldVz_left[iGpu], dev_wavefieldVz_cur[iGpu], dev_wavefieldVz_right[iGpu], dev_ssVxRight[iGpu], dev_ssVyRight[iGpu], dev_ssVzRight[iGpu], dev_ssSigmaxxRight[iGpu], dev_ssSigmayyRight[iGpu], dev_ssSigmazzRight[iGpu], dev_ssSigmaxzRight[iGpu], dev_ssSigmaxyRight[iGpu], dev_ssSigmayzRight[iGpu], dev_drhox[iGpu], dev_drhoy[iGpu], dev_drhoz[iGpu], dev_dlame[iGpu], dev_dmu[iGpu], dev_dmuxz[iGpu], dev_dmuxy[iGpu], dev_dmuyz[iGpu], nx, ny, nz, its+1));
		}

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			launchFwdStepKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, compStream);
			// Inject secondary source sample itw-1 in each component
			launchInjectSecondarySource_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, it2, compStream);
			// Damp wavefields
			launchDampCosineEdgeKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, compStream);
			// Extract and interpolate data
			launchFwdRecordInterpDataKernels_3D(nblockDataCenterGrid, nblockDataXGrid, nblockDataYGrid, nblockDataZGrid, nblockDataXZGrid, nblockDataXYGrid, nblockDataYZGrid, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, its, it2, iGpu, compStream);
			// Switch pointers
			switchPointers_3D(iGpu);

		}
		// std::cout << "its = " << its << std::endl;
		// cuda_call(cudaMemcpy(tmpSlice, dev_p0_vx[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));
		// std::cout << "Min value pLeft = " << *std::min_element(tmpSlice,tmpSlice+nModel) << std::endl;
		// std::cout << "Max value pLeft = " << *std::max_element(tmpSlice,tmpSlice+nModel) << std::endl;

		// Switch pointers for secondary source and setting right slices to zero
		switchPointersSecondarySource_3D(iGpu);
		cuda_call(cudaMemsetAsync(dev_ssVxRight[iGpu], 0, nModel*sizeof(double), compStream));
		cuda_call(cudaMemsetAsync(dev_ssVyRight[iGpu], 0, nModel*sizeof(double), compStream));
		cuda_call(cudaMemsetAsync(dev_ssVzRight[iGpu], 0, nModel*sizeof(double), compStream));
		cuda_call(cudaMemsetAsync(dev_ssSigmaxxRight[iGpu], 0, nModel*sizeof(double), compStream));
		cuda_call(cudaMemsetAsync(dev_ssSigmayyRight[iGpu], 0, nModel*sizeof(double), compStream));
		cuda_call(cudaMemsetAsync(dev_ssSigmazzRight[iGpu], 0, nModel*sizeof(double), compStream));
		cuda_call(cudaMemsetAsync(dev_ssSigmaxzRight[iGpu], 0, nModel*sizeof(double), compStream));
		cuda_call(cudaMemsetAsync(dev_ssSigmaxyRight[iGpu], 0, nModel*sizeof(double), compStream));
		cuda_call(cudaMemsetAsync(dev_ssSigmayzRight[iGpu], 0, nModel*sizeof(double), compStream));

		if (its < host_nts-3){
			// Streams related pointers
			switchPointers_wavefield3Slices_3D(iGpu);
		}

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStream));

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
	cuda_call(cudaFree(dev_sourceRegDts_vx[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_vy[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_vz[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmaxx[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmayy[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmazz[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmaxz[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmaxy[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmayz[iGpu]));

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

	// Destroying streams
	cuda_call(cudaStreamDestroy(compStream));
	cuda_call(cudaStreamDestroy(transferStream));

}

/****************************************************************************************/
/********************************* Born adjoint operator ********************************/
/****************************************************************************************/
void BornElasticAdjGpu_3D(double *sourceRegDts_vx, double *sourceRegDts_vy, double *sourceRegDts_vz, double *sourceRegDts_sigmaxx, double *sourceRegDts_sigmayy, double *sourceRegDts_sigmazz, double *sourceRegDts_sigmaxz, double *sourceRegDts_sigmaxy, double *sourceRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, double *drhox, double *drhoy, double *drhoz, double *dlame,  double *dmu, double *dmuxz, double *dmuxy, double *dmuyz, int nx, int ny, int nz, int iGpu, int iGpuId){

	//setup: a) src and receiver positions allocation and copying to device
	//       b) allocate and copy source (arrays for sources for each wavefield) to device
	//       c) allocate and copy data (recevier recordings arrays) to device
	//       d) allocate and copy wavefield time slices to gpu
	setupAdjGpu_3D(sourceRegDts_vx, sourceRegDts_vy, sourceRegDts_vz, sourceRegDts_sigmaxx, sourceRegDts_sigmayy, sourceRegDts_sigmazz, sourceRegDts_sigmaxz, sourceRegDts_sigmaxy, sourceRegDts_sigmayz, dataRegDts_vx, dataRegDts_vy, dataRegDts_vz, dataRegDts_sigmaxx, dataRegDts_sigmayy, dataRegDts_sigmazz, dataRegDts_sigmaxz, dataRegDts_sigmaxy, dataRegDts_sigmayz, sourcesPositionRegCenterGrid, nSourcesRegCenterGrid, sourcesPositionRegXGrid, nSourcesRegXGrid, sourcesPositionRegYGrid, nSourcesRegYGrid, sourcesPositionRegZGrid, nSourcesRegZGrid, sourcesPositionRegXZGrid, nSourcesRegXZGrid, sourcesPositionRegXYGrid, nSourcesRegXYGrid, sourcesPositionRegYZGrid, nSourcesRegYZGrid, receiversPositionRegCenterGrid, nReceiversRegCenterGrid, receiversPositionRegXGrid, nReceiversRegXGrid, receiversPositionRegYGrid, nReceiversRegYGrid, receiversPositionRegZGrid, nReceiversRegZGrid, receiversPositionRegXZGrid, nReceiversRegXZGrid, receiversPositionRegXYGrid, nReceiversRegXYGrid, receiversPositionRegYZGrid, nReceiversRegYZGrid, nx, ny, nz, iGpu, iGpuId);

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

	cudaStream_t compStream, transferStream;
	// Create streams for saving the wavefield
	cudaStreamCreate(&compStream);
	cudaStreamCreate(&transferStream);

	long long nModel = nx;
	nModel *= ny * nz;

	wavefieldSliceInitializeGpu_3D(nModel, iGpu);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
			for (int it2 = 1; it2 < host_sub+1; it2++){

				// Step forward
				launchFwdStepKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, compStream);
				// Inject source
				launchFwdInjectSourceKernels_3D(nblockSouCenterGrid, nblockSouXGrid, nblockSouYGrid, nblockSouZGrid,nblockSouXZGrid, nblockSouXYGrid, nblockSouYZGrid, nSourcesRegCenterGrid, nSourcesRegXGrid, nSourcesRegYGrid, nSourcesRegZGrid, nSourcesRegXZGrid, nSourcesRegXYGrid, nSourcesRegYZGrid, its, it2, iGpu, compStream);
				// Damp wavefields
				launchDampCosineEdgeKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, compStream);
				// Extract wavefield
				kernel_stream_exec(interpWavefieldVxVyVz_3D<<<dimGrid, dimBlock, 0, compStream>>>(dev_wavefieldVx_left[iGpu], dev_wavefieldVx_right[iGpu], dev_wavefieldVy_left[iGpu], dev_wavefieldVy_right[iGpu], dev_wavefieldVz_left[iGpu], dev_wavefieldVz_right[iGpu], dev_p0_vx[iGpu], dev_p0_vy[iGpu], dev_p0_vz[iGpu], nx, ny, nz, it2));
				// Switch pointers
				switchPointers_3D(iGpu);

			}

			/* Note: At that point pLeft [its] is ready to be transfered back to host */

			// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
			cuda_call(cudaStreamSynchronize(transferStream));

			// Asynchronous copy of dev_wavefieldDts_left => dev_pStream [its] [compute]
			cuda_call(cudaMemcpyAsync(dev_pStream_Vx[iGpu], dev_wavefieldVx_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
			cuda_call(cudaMemcpyAsync(dev_pStream_Vy[iGpu], dev_wavefieldVy_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
			cuda_call(cudaMemcpyAsync(dev_pStream_Vz[iGpu], dev_wavefieldVz_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));

			// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
			cuda_call(cudaStreamSynchronize(compStream));

			// Asynchronous transfer of pStream => pin [its] [transfer]
			// Launch the transfer while we compute the next coarse time sample
			cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vx[iGpu]+its*nModel, dev_pStream_Vx[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
			cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vy[iGpu]+its*nModel, dev_pStream_Vy[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
			cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vz[iGpu]+its*nModel, dev_pStream_Vz[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));

			// Switch pointers
			switchPointers_wavefield2Slices_3D(iGpu);
			// Reinitialize dev_wavefieldDts_right to zero
			cuda_call(cudaMemsetAsync(dev_wavefieldVx_right[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_wavefieldVy_right[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_wavefieldVz_right[iGpu], 0, nModel*sizeof(double), compStream));

	}
	// Note: At that point, dev_wavefieldDts_left contains the value of the wavefield at nts-1 (last time sample) and the wavefield only has values up to nts-2
	cuda_call(cudaStreamSynchronize(compStream));
	cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vx[iGpu]+(host_nts-1)*nModel, dev_wavefieldVx_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
	cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vy[iGpu]+(host_nts-1)*nModel, dev_wavefieldVy_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
	cuda_call(cudaMemcpyAsync(host_pinned_wavefield_vz[iGpu]+(host_nts-1)*nModel, dev_wavefieldVz_left[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream));
	cuda_call(cudaStreamSynchronize(transferStream));

	/************************** Receiver wavefield computation ****************************/
	// Initialize time slices on device
	wavefieldInitializeOnGpu_3D(nModel, iGpu);
	SecondarySourceInitializeOnGpu_3D(nModel, iGpu);

	// Copy model perturbations to device (done within setupBornFwdGpu function)
	//Note the perturbations have been already scaled by the wave-equation source scaling factor outside of this function

	// Copy wavefield time-slice its = 0: pinned -> dev_pSourceWavefield_old
	cuda_call(cudaMemcpyAsync(dev_wavefieldVx_left[iGpu], host_pinned_wavefield_vx[iGpu]+(host_nts-1)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_wavefieldVy_left[iGpu], host_pinned_wavefield_vy[iGpu]+(host_nts-1)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_wavefieldVz_left[iGpu], host_pinned_wavefield_vz[iGpu]+(host_nts-1)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));

	// Copy wavefield time-slice its = 1: pinned -> dev_pSourceWavefield_cur
	cuda_call(cudaMemcpyAsync(dev_wavefieldVx_cur[iGpu], host_pinned_wavefield_vx[iGpu]+(host_nts-2)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_wavefieldVy_cur[iGpu], host_pinned_wavefield_vy[iGpu]+(host_nts-2)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaMemcpyAsync(dev_wavefieldVz_cur[iGpu], host_pinned_wavefield_vz[iGpu]+(host_nts-2)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
	cuda_call(cudaStreamSynchronize(transferStream));

	// Start propagation
	for (int its = host_nts-2; its > -1; its--){

			// Copy new slice from RAM -> pinned for time its-1 -> transfer to pStream
			if(its > 0){
				// Wait until compStream has done copying wavefield value its+1 (previous) from pStream -> dev_pSourceWavefield
				cuda_call(cudaStreamSynchronize(compStream));
				cuda_call(cudaMemcpyAsync(dev_pStream_Vx[iGpu], host_pinned_wavefield_vx[iGpu]+(its-1)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
				cuda_call(cudaMemcpyAsync(dev_pStream_Vy[iGpu], host_pinned_wavefield_vy[iGpu]+(its-1)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
				cuda_call(cudaMemcpyAsync(dev_pStream_Vz[iGpu], host_pinned_wavefield_vz[iGpu]+(its-1)*nModel, nModel*sizeof(double), cudaMemcpyHostToDevice, transferStream));
			}

			for (int it2 = host_sub-1; it2 > -1; it2--){
					// Step back in time
					launchAdjStepKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, compStream);
					// Inject data
					launchAdjInterpInjectDataKernels_3D(nblockDataCenterGrid, nblockDataXGrid, nblockDataYGrid, nblockDataZGrid, nblockDataXZGrid, nblockDataXYGrid, nblockDataYZGrid, nReceiversRegCenterGrid, nReceiversRegXGrid, nReceiversRegYGrid, nReceiversRegZGrid, nReceiversRegXZGrid, nReceiversRegXYGrid, nReceiversRegYZGrid, its, it2, iGpu, compStream);
					// Damp wavefield
					launchDampCosineEdgeKernels_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, compStream);
					// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
					launchExtractInterpAdjointWavefield_3D(dimGrid, dimBlock, nx, ny, nz, iGpu, it2, compStream);

					// Switch pointers
					switchPointers_3D(iGpu);
			}

			// Apply extended imaging condition for its+1
			if (its == host_nts-2){
				// Last step must be different since left => (its-3), cur => (its-2), right => (its-1)
				kernel_stream_exec(imagingElaAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream>>>(dev_wavefieldVx_cur[iGpu], dev_wavefieldVx_right[iGpu], NULL, dev_wavefieldVy_cur[iGpu], dev_wavefieldVy_right[iGpu], NULL, dev_wavefieldVz_cur[iGpu], dev_wavefieldVz_right[iGpu], NULL, dev_ssVxRight[iGpu], dev_ssVyRight[iGpu], dev_ssVzRight[iGpu], dev_ssSigmaxxRight[iGpu], dev_ssSigmayyRight[iGpu], dev_ssSigmazzRight[iGpu], dev_ssSigmaxzRight[iGpu], dev_ssSigmaxyRight[iGpu], dev_ssSigmayzRight[iGpu], dev_drhox[iGpu], dev_drhoy[iGpu], dev_drhoz[iGpu], dev_dlame[iGpu], dev_dmu[iGpu], dev_dmuxz[iGpu], dev_dmuxy[iGpu], dev_dmuyz[iGpu], nx, ny, nz, its+1));
			} else {
				kernel_stream_exec(imagingElaAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream>>>(dev_wavefieldVx_left[iGpu], dev_wavefieldVx_cur[iGpu], dev_wavefieldVx_right[iGpu], dev_wavefieldVy_left[iGpu], dev_wavefieldVy_cur[iGpu], dev_wavefieldVy_right[iGpu], dev_wavefieldVz_left[iGpu], dev_wavefieldVz_cur[iGpu], dev_wavefieldVz_right[iGpu], dev_ssVxRight[iGpu], dev_ssVyRight[iGpu], dev_ssVzRight[iGpu], dev_ssSigmaxxRight[iGpu], dev_ssSigmayyRight[iGpu], dev_ssSigmazzRight[iGpu], dev_ssSigmaxzRight[iGpu], dev_ssSigmaxyRight[iGpu], dev_ssSigmayzRight[iGpu], dev_drhox[iGpu], dev_drhoy[iGpu], dev_drhoz[iGpu], dev_dlame[iGpu], dev_dmu[iGpu], dev_dmuxz[iGpu], dev_dmuxy[iGpu], dev_dmuyz[iGpu], nx, ny, nz, its+1));
			}

			// Wait until transfer stream has finished copying slice its from pinned -> pStream
			cuda_call(cudaStreamSynchronize(transferStream));

			if (its < host_nts-2){
				cuda_call(cudaStreamSynchronize(compStream)); //DEBUG
				// Streams related pointers
				switchPointers_wavefield3Slices_adj_3D(iGpu);
			}

			// Copy source wavefield slice its-1 to dev_pSourceWavefield
			if(its > 0){
				cuda_call(cudaMemcpyAsync(dev_wavefieldVx_left[iGpu], dev_pStream_Vx[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
				cuda_call(cudaMemcpyAsync(dev_wavefieldVy_left[iGpu], dev_pStream_Vy[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
				cuda_call(cudaMemcpyAsync(dev_wavefieldVz_left[iGpu], dev_pStream_Vz[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream));
			}


			// Switch pointers for secondary source and setting right slices to zero
			switchPointersSecondarySource_3D(iGpu);
			cuda_call(cudaMemsetAsync(dev_ssVxLeft[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_ssVyLeft[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_ssVzLeft[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_ssSigmaxxLeft[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_ssSigmayyLeft[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_ssSigmazzLeft[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_ssSigmaxzLeft[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_ssSigmaxyLeft[iGpu], 0, nModel*sizeof(double), compStream));
			cuda_call(cudaMemsetAsync(dev_ssSigmayzLeft[iGpu], 0, nModel*sizeof(double), compStream));


	}// Finished main loop - we still have to compute imaging condition for its=0

	kernel_stream_exec(imagingElaAdjGpu_3D<<<dimGrid, dimBlock, 0, compStream>>>(NULL, dev_wavefieldVx_cur[iGpu], dev_wavefieldVx_right[iGpu], NULL, dev_wavefieldVy_cur[iGpu], dev_wavefieldVy_right[iGpu], NULL, dev_wavefieldVz_cur[iGpu], dev_wavefieldVz_right[iGpu], dev_ssVxRight[iGpu], dev_ssVyRight[iGpu], dev_ssVzRight[iGpu], dev_ssSigmaxxRight[iGpu], dev_ssSigmayyRight[iGpu], dev_ssSigmazzRight[iGpu], dev_ssSigmaxzRight[iGpu], dev_ssSigmaxyRight[iGpu], dev_ssSigmayzRight[iGpu], dev_drhox[iGpu], dev_drhoy[iGpu], dev_drhoz[iGpu], dev_dlame[iGpu], dev_dmu[iGpu], dev_dmuxz[iGpu], dev_dmuxy[iGpu], dev_dmuyz[iGpu], nx, ny, nz, 0));
	cuda_call(cudaStreamSynchronize(compStream));


	// Copy model back to host
	cuda_call(cudaMemcpy(drhox, dev_drhox[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(drhoy, dev_drhox[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(drhoz, dev_drhoz[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(dlame, dev_dlame[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(dmu, dev_dmu[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(dmuxz, dev_dmuxz[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(dmuxy, dev_dmuxy[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));
	cuda_call(cudaMemcpy(dmuyz, dev_dmuyz[iGpu], nModel*sizeof(double), cudaMemcpyDeviceToHost));


	// Deallocate all slices
	cuda_call(cudaFree(dev_sourceRegDts_vx[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_vy[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_vz[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmaxx[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmayy[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmazz[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmaxz[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmaxy[iGpu]));
	cuda_call(cudaFree(dev_sourceRegDts_sigmayz[iGpu]));

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

	// Destroying streams
	cuda_call(cudaStreamDestroy(compStream));
	cuda_call(cudaStreamDestroy(transferStream));

}
