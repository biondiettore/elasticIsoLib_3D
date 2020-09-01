#include <vector>
#include <ctime>
#include "BornElasticGpu_3D.h"
#include <cstring>
#include <stdexcept>


BornElasticGpu_3D::BornElasticGpu_3D(std::shared_ptr<fdParamElastic_3D> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParamElastic = fdParamElastic;
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;
	_saveWavefield = par->getInt("saveWavefield", 0);
	_domDec = par->getInt("domDec", 0); //Flag to use domain decomposition or not
	_freeSurface = par->getInt("freeSurface", 0); //Flag to use free surface boundary condition or not

	// Initialize GPU
	initBornElasticGpu_3D(_fdParamElastic->_dz, _fdParamElastic->_dx, _fdParamElastic->_dy, _fdParamElastic->_nz, _fdParamElastic->_nx, _fdParamElastic->_ny, _fdParamElastic->_nts, _fdParamElastic->_dts, _fdParamElastic->_sub, _fdParamElastic->_minPad, _fdParamElastic->_blockSize, _fdParamElastic->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);

	// Allocate on GPUs
	allocateBornElasticGpu_3D(_fdParamElastic->_rhoxDtw, _fdParamElastic->_rhoyDtw, _fdParamElastic->_rhozDtw, _fdParamElastic->_lamb2MuDtw, _fdParamElastic->_lambDtw, _fdParamElastic->_muxzDtw, _fdParamElastic->_muxyDtw, _fdParamElastic->_muyzDtw, _fdParamElastic->_nx, _fdParamElastic->_ny, _fdParamElastic->_nz, _iGpu, iGpuId);
	setAllWavefields_3D(0); // By default, do not record the scattered wavefields
}



bool BornElasticGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::double4DReg> model, const std::shared_ptr<SEP::double3DReg> data) const{

	if (_fdParamElastic->checkParfileConsistencyTime_3D(data, 1, "Data File") != true) {return false;} // Check data time axis
	if (_fdParamElastic->checkParfileConsistencyTime_3D(_sourcesSignals, 1, "Source File") != true) {return false;}; // Check sources time axis
	if (_fdParamElastic->checkParfileConsistencySpace_3D(model, "Model File") != true) {return false;}; // Check model axis

	return true;
}

void BornElasticGpu_3D::setAllWavefields_3D(int wavefieldFlag){
	_srcWavefield = setWavefield_3D(wavefieldFlag);
	_secWavefield = setWavefield_3D(wavefieldFlag);
}
