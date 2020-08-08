#ifndef NONLINEAR_PROP_ELASTIC_GPU_FUNCTIONS_3D_H
#define NONLINEAR_PROP_ELASTIC_GPU_FUNCTIONS_3D_H 1
#include <vector>

/*********************************** Initialization **************************************/
// Gpu info
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumber);
void initNonlinearElasticGpu_3D(double dz, double dx, double dy, int nz, int nx, double ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateNonlinearElasticGpu_3D(double *rhoxDtw, double *rhoyDtw, double *rhozDtw, double *lamb2MuDt, double *lambDtw, double *muxzDtw, double *muxyDtw, double *muyzDtw, int nx, int ny, int nz, int iGpu, int iGpuId);
void deallocateNonlinearElasticGpu_3D(int iGpu, int iGpuId);

/*********************************** Nonlinear FWD **************************************/
void propElasticFwdGpu_3D(double *modelRegDts_vx, double *modelRegDts_vy, double *modelRegDts_vz, double *modelRegDts_sigmaxx, double *modelRegDts_sigmayy, double *modelRegDts_sigmazz, double *modelRegDts_sigmaxz, double *modelRegDts_sigmaxy, double *modelRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, int nx, int ny, int nz, int iGpu, int iGpuId);

/*********************************** Nonlinear ADJ **************************************/




#endif
