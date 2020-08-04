#ifndef NONLINEAR_PROP_ELASTIC_GPU_FUNCTIONS_3D_H
#define NONLINEAR_PROP_ELASTIC_GPU_FUNCTIONS_3D_H 1
#include <vector>

/*********************************** Initialization **************************************/
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumber);
// void initNonlinearElasticGpu_3D(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
// void allocateNonlinearElasticGpu_3D(double *rhoxDtw, double *rhozDtw, double *lamb2MuDt, double *lambDtw, double *muxzDt,int iGpu, int iGpuId);
// void deallocateNonlinearElasticGpu_3D(int iGpu, int iGpuId);




#endif
