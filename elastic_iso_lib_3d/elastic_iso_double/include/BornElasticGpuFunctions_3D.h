#ifndef BORN_PROP_ELASTIC_GPU_FUNCTIONS_3D_H
#define BORN_PROP_ELASTIC_GPU_FUNCTIONS_3D_H 1
#include <vector>

/*********************************** Initialization ****************************/
// Gpu info
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumber);
void setGpuP2P(int nGpu, int info, std::vector<int> gpuList);
void initBornElasticGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornElasticGpu_3D(double *rhoxDtw, double *rhoyDtw, double *rhozDtw, double *lamb2MuDt, double *lambDtw, double *muxzDtw, double *muxyDtw, double *muyzDtw, int nx, int ny, int nz, int iGpu, int iGpuId);
void deallocateBornElasticGpu_3D(int iGpu, int iGpuId);
//Functions to allocate and deallocate the pinned-memory wavefield arrays
void allocatePinBornElasticGpu_3D(int nx, int ny, int nz, int nts, int nGpu, int iGpuId, int iGpu, int iGpuAlloc);
void deallocatePinBornElasticGpu_3D(int iGpu, int iGpuId);

/*********************************** Born FWD *********************************/
void BornElasticFwdGpu_3D(double *sourceRegDts_vx, double *sourceRegDts_vy, double *sourceRegDts_vz, double *sourceRegDts_sigmaxx, double *sourceRegDts_sigmayy, double *sourceRegDts_sigmazz, double *sourceRegDts_sigmaxz, double *sourceRegDts_sigmaxy, double *sourceRegDts_sigmayz, double *dataRegDts_vx, double *dataRegDts_vy, double *dataRegDts_vz, double *dataRegDts_sigmaxx, double *dataRegDts_sigmayy, double *dataRegDts_sigmazz, double *dataRegDts_sigmaxz, double *dataRegDts_sigmaxy, double *dataRegDts_sigmayz, long long *sourcesPositionRegCenterGrid, long long nSourcesRegCenterGrid, long long *sourcesPositionRegXGrid, long long nSourcesRegXGrid, long long *sourcesPositionRegYGrid, long long nSourcesRegYGrid, long long *sourcesPositionRegZGrid, long long nSourcesRegZGrid, long long *sourcesPositionRegXZGrid, long long nSourcesRegXZGrid, long long *sourcesPositionRegXYGrid, long long nSourcesRegXYGrid, long long *sourcesPositionRegYZGrid, long long nSourcesRegYZGrid, long long *receiversPositionRegCenterGrid, long long nReceiversRegCenterGrid, long long *receiversPositionRegXGrid, long long nReceiversRegXGrid, long long *receiversPositionRegYGrid, long long nReceiversRegYGrid, long long *receiversPositionRegZGrid, long long nReceiversRegZGrid, long long *receiversPositionRegXZGrid, long long nReceiversRegXZGrid, long long *receiversPositionRegXYGrid, long long nReceiversRegXYGrid, long long *receiversPositionRegYZGrid, long long nReceiversRegYZGrid, double *drhox, double *drhoy, double *drhoz, double *dlame,  double *dmu, double *dmuxz, double *dmuxy, double *dmuyz, int nx, int ny, int nz, int iGpu, int iGpuId);


/*********************************** Born ADJ *********************************/


#endif
