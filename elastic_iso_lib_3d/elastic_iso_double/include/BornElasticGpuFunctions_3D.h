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

/*********************************** Born FWD *********************************/



/*********************************** Born ADJ *********************************/


#endif
