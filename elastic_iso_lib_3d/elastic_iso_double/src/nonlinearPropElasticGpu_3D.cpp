#include <vector>
#include <ctime>
#include "nonlinearPropElasticGpu_3D.h"
#include <cstring>

nonlinearPropElasticGpu_3D::nonlinearPropElasticGpu_3D(std::shared_ptr<fdParamElastic_3D> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParamElastic = fdParamElastic;
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;
	_saveWavefield = par->getInt("saveWavefield", 0);
	_useStreams = par->getInt("useStreams", 0); //Flag whether to use streams to save the wavefield
	_domDec = par->getInt("domDec", 0); //Flag to use domain decomposition or not
	_freeSurface = par->getInt("freeSurface", 0); //Flag to use free surface boundary condition or not

	// Initialize GPU
	initNonlinearElasticGpu_3D(_fdParamElastic->_dz, _fdParamElastic->_dx, _fdParamElastic->_dy, _fdParamElastic->_nz, _fdParamElastic->_nx, _fdParamElastic->_ny, _fdParamElastic->_nts, _fdParamElastic->_dts, _fdParamElastic->_sub, _fdParamElastic->_minPad, _fdParamElastic->_blockSize, _fdParamElastic->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);

	/// Alocate on GPUs
	allocateNonlinearElasticGpu_3D(_fdParamElastic->_rhoxDtw, _fdParamElastic->_rhoyDtw, _fdParamElastic->_rhozDtw, _fdParamElastic->_lamb2MuDtw, _fdParamElastic->_lambDtw, _fdParamElastic->_muxzDtw, _fdParamElastic->_muxyDtw, _fdParamElastic->_muyzDtw, _fdParamElastic->_nx, _fdParamElastic->_ny, _fdParamElastic->_nz, _iGpu, iGpuId);
	setAllWavefields_3D(0); // By default, do not record the scattered wavefields
}

void nonlinearPropElasticGpu_3D::setAllWavefields_3D(int wavefieldFlag){
	_wavefield = setWavefield_3D(wavefieldFlag);
}

bool nonlinearPropElasticGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double3DReg> data) const{

	if (_fdParamElastic->checkParfileConsistencyTime_3D(data, 1, "Data File") != true) {return false;} // Check data time axis
	if (_fdParamElastic->checkParfileConsistencyTime_3D(model,1, "Model File") != true) {return false;}; // Check model time axis

	return true;
}

// model is seismic sources and data are receiver recordings
void nonlinearPropElasticGpu_3D::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

	/* Allocation */
  std::shared_ptr<double2DReg> modelIrreg_vx(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXGrid));
	std::shared_ptr<double2DReg> modelIrreg_vy(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregYGrid));
  std::shared_ptr<double2DReg> modelIrreg_vz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregZGrid));
  std::shared_ptr<double2DReg> modelIrreg_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
	std::shared_ptr<double2DReg> modelIrreg_sigmayy(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
  std::shared_ptr<double2DReg> modelIrreg_sigmazz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
  std::shared_ptr<double2DReg> modelIrreg_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXZGrid));
	std::shared_ptr<double2DReg> modelIrreg_sigmaxy(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXYGrid));
	std::shared_ptr<double2DReg> modelIrreg_sigmayz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregYZGrid));

  std::shared_ptr<double2DReg> modelRegDts_vx(new double2DReg(_fdParamElastic->_nts, _nSourcesRegXGrid));
	std::shared_ptr<double2DReg> modelRegDts_vy(new double2DReg(_fdParamElastic->_nts, _nSourcesRegYGrid));
  std::shared_ptr<double2DReg> modelRegDts_vz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegZGrid));
  std::shared_ptr<double2DReg> modelRegDts_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
	std::shared_ptr<double2DReg> modelRegDts_sigmayy(new double2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
  std::shared_ptr<double2DReg> modelRegDts_sigmazz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegCenterGrid));
  std::shared_ptr<double2DReg> modelRegDts_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegXZGrid));
	std::shared_ptr<double2DReg> modelRegDts_sigmaxy(new double2DReg(_fdParamElastic->_nts, _nSourcesRegXYGrid));
	std::shared_ptr<double2DReg> modelRegDts_sigmayz(new double2DReg(_fdParamElastic->_nts, _nSourcesRegYZGrid));

  std::shared_ptr<double2DReg> dataRegDts_vx(new double2DReg(_fdParamElastic->_nts, _nReceiversRegXGrid));
	std::shared_ptr<double2DReg> dataRegDts_vy(new double2DReg(_fdParamElastic->_nts, _nReceiversRegYGrid));
  std::shared_ptr<double2DReg> dataRegDts_vz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegZGrid));
  std::shared_ptr<double2DReg> dataRegDts_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nReceiversRegCenterGrid));
	std::shared_ptr<double2DReg> dataRegDts_sigmayy(new double2DReg(_fdParamElastic->_nts, _nReceiversRegCenterGrid));
  std::shared_ptr<double2DReg> dataRegDts_sigmazz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegCenterGrid));
	std::shared_ptr<double2DReg> dataRegDts_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegXZGrid));
  std::shared_ptr<double2DReg> dataRegDts_sigmaxy(new double2DReg(_fdParamElastic->_nts, _nReceiversRegXYGrid));
	std::shared_ptr<double2DReg> dataRegDts_sigmayz(new double2DReg(_fdParamElastic->_nts, _nReceiversRegYZGrid));

  std::shared_ptr<double2DReg> dataIrreg_vx(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregXGrid));
	std::shared_ptr<double2DReg> dataIrreg_vy(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregYGrid));
  std::shared_ptr<double2DReg> dataIrreg_vz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregZGrid));
  std::shared_ptr<double2DReg> dataIrreg_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregCenterGrid));
	std::shared_ptr<double2DReg> dataIrreg_sigmayy(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregCenterGrid));
  std::shared_ptr<double2DReg> dataIrreg_sigmazz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregCenterGrid));
  std::shared_ptr<double2DReg> dataIrreg_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregXZGrid));
	std::shared_ptr<double2DReg> dataIrreg_sigmaxy(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregXYGrid));
	std::shared_ptr<double2DReg> dataIrreg_sigmayz(new double2DReg(_fdParamElastic->_nts, _nReceiversIrregYZGrid));

	if (!add){
	  data->scale(0.0);
  } else {
	  /* Copy the data to the temporary array */
		// Vx
		long long shift;
		long long chunk = _nReceiversIrregXGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_vx->getVals(),data->getVals(), chunk*sizeof(double));
		// Vy
		shift = chunk;
		chunk = _nReceiversIrregYGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_vy->getVals(), data->getVals()+shift, chunk*sizeof(double));
		// Vz
		shift += chunk;
		chunk = _nReceiversIrregZGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_vz->getVals(), data->getVals()+shift, chunk*sizeof(double));
		// Sigmaxx
		shift += chunk;
		chunk = _nReceiversIrregCenterGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_sigmaxx->getVals(), data->getVals()+shift, chunk*sizeof(double));
		// Sigmayy
		shift += chunk;
		chunk = _nReceiversIrregCenterGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_sigmayy->getVals(), data->getVals()+shift, chunk*sizeof(double));
		// Sigmazz
		shift += chunk;
		chunk = _nReceiversIrregCenterGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_sigmazz->getVals(), data->getVals()+shift, chunk*sizeof(double));
		// Sigmaxz
		shift += chunk;
		chunk = _nReceiversIrregXZGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_sigmaxz->getVals(), data->getVals()+shift, chunk*sizeof(double));
		// Sigmaxy
		shift += chunk;
		chunk = _nReceiversIrregXYGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_sigmaxy->getVals(), data->getVals()+shift, chunk*sizeof(double));
		// Sigmayz
		shift += chunk;
		chunk = _nReceiversIrregYZGrid*_fdParamElastic->_nts;
	  std::memcpy(dataIrreg_sigmayz->getVals(), data->getVals()+shift, chunk*sizeof(double));
  }

	/* Copy from 3d model to respective 2d model components */
	// fx
	long long shift;
	long long chunk = _nSourcesIrregXGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_vx->getVals(), model->getVals(), chunk*sizeof(double));
	// fy
	shift = chunk;
	chunk = _nSourcesIrregYGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_vy->getVals(), model->getVals()+shift, chunk*sizeof(double));
	// fz
	shift += chunk;
	chunk = _nSourcesIrregZGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_vz->getVals(), model->getVals()+shift, chunk*sizeof(double));
	// mxx
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_sigmaxx->getVals(), model->getVals()+shift, chunk*sizeof(double));
	// myy
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_sigmayy->getVals(), model->getVals()+shift, chunk*sizeof(double));
	// mzz
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_sigmazz->getVals(), model->getVals()+shift, chunk*sizeof(double));
	// mxz
	shift += chunk;
	chunk = _nSourcesIrregXZGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_sigmaxz->getVals(), model->getVals()+shift, chunk*sizeof(double));
	// mxz
	shift += chunk;
	chunk = _nSourcesIrregXYGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_sigmaxy->getVals(), model->getVals()+shift, chunk*sizeof(double));
	// myz
	shift += chunk;
	chunk = _nSourcesIrregYZGrid*_fdParamElastic->_nts;
  std::memcpy(modelIrreg_sigmayz->getVals(), model->getVals()+shift, chunk*sizeof(double));

	/* Interpolate model (seismic source) to regular grid */
  _sourcesXGrid->adjoint(false, modelRegDts_vx, modelIrreg_vx);
	_sourcesYGrid->adjoint(false, modelRegDts_vy, modelIrreg_vy);
  _sourcesZGrid->adjoint(false, modelRegDts_vz, modelIrreg_vz);
  _sourcesCenterGrid->adjoint(false, modelRegDts_sigmaxx, modelIrreg_sigmaxx);
	_sourcesCenterGrid->adjoint(false, modelRegDts_sigmayy, modelIrreg_sigmayy);
  _sourcesCenterGrid->adjoint(false, modelRegDts_sigmazz, modelIrreg_sigmazz);
  _sourcesXZGrid->adjoint(false, modelRegDts_sigmaxz, modelIrreg_sigmaxz);
	_sourcesXYGrid->adjoint(false, modelRegDts_sigmaxy, modelIrreg_sigmaxy);
	_sourcesYZGrid->adjoint(false, modelRegDts_sigmayz, modelIrreg_sigmayz);

	/* Scale source signals model */
	// fx
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegXGrid; is++){ //loop over number of reg sources x grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*modelRegDts_vx->_mat)[is][it] *= _fdParamElastic->_rhoxDtw[(_sourcesXGrid->getRegPosUnique())[is]];
		}
  }
	// fy
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegYGrid; is++){ //loop over number of reg sources y grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*modelRegDts_vy->_mat)[is][it] *= _fdParamElastic->_rhoyDtw[(_sourcesYGrid->getRegPosUnique())[is]];
		}
  }
	// fz
  #pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegZGrid; is++){ //loop over number of reg sources z grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*modelRegDts_vz->_mat)[is][it] *= _fdParamElastic->_rhozDtw[(_sourcesZGrid->getRegPosUnique())[is]];
		}
  }

	// moment tensor components
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegCenterGrid; is++){ //loop over number of reg sources xz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
			double mxx = 0;
			double myy = 0;
			double mzz = 0;
			// mxx
	  	mxx = (*modelRegDts_sigmaxx->_mat)[is][it] * _fdParamElastic->_lamb2MuDtw[(_sourcesCenterGrid->getRegPosUnique())[is]]+
								((*modelRegDts_sigmayy->_mat)[is][it] + (*modelRegDts_sigmazz->_mat)[is][it] ) * _fdParamElastic->_lambDtw[(_sourcesCenterGrid->getRegPosUnique())[is]];
			// myy
			myy = (*modelRegDts_sigmayy->_mat)[is][it] * _fdParamElastic->_lamb2MuDtw[(_sourcesCenterGrid->getRegPosUnique())[is]]+
								((*modelRegDts_sigmaxx->_mat)[is][it] + (*modelRegDts_sigmazz->_mat)[is][it] ) * _fdParamElastic->_lambDtw[(_sourcesCenterGrid->getRegPosUnique())[is]];
			// mzz
			mzz = (*modelRegDts_sigmazz->_mat)[is][it] * _fdParamElastic->_lamb2MuDtw[(_sourcesCenterGrid->getRegPosUnique())[is]]+
								((*modelRegDts_sigmaxx->_mat)[is][it] + (*modelRegDts_sigmayy->_mat)[is][it] ) * _fdParamElastic->_lambDtw[(_sourcesCenterGrid->getRegPosUnique())[is]];
		  // Setting values
			(*modelRegDts_sigmaxx->_mat)[is][it] = mxx;
			(*modelRegDts_sigmayy->_mat)[is][it] = myy;
			(*modelRegDts_sigmazz->_mat)[is][it] = mzz;
		}
  }
	// mxz
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegXZGrid; is++){ //loop over number of reg sources xz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*modelRegDts_sigmaxz->_mat)[is][it] *= _fdParamElastic->_muxzDtw[(_sourcesXZGrid->getRegPosUnique())[is]];
		}
  }
	// mxy
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegXYGrid; is++){ //loop over number of reg sources xz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*modelRegDts_sigmaxy->_mat)[is][it] *= _fdParamElastic->_muxyDtw[(_sourcesYZGrid->getRegPosUnique())[is]];
		}
  }
	// myz
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegYZGrid; is++){ //loop over number of reg sources yz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*modelRegDts_sigmayz->_mat)[is][it] *= _fdParamElastic->_muyzDtw[(_sourcesYZGrid->getRegPosUnique())[is]];
		}
  }

	/*Scaling by the inverse of the space discretization*/
	double area_scale = 1.0/(_fdParamElastic->_dx * _fdParamElastic->_dy * _fdParamElastic->_dz);
	modelRegDts_vx->scale(area_scale);
	modelRegDts_vy->scale(area_scale);
  modelRegDts_vz->scale(area_scale);
  modelRegDts_sigmaxx->scale(area_scale);
	modelRegDts_sigmayy->scale(area_scale);
  modelRegDts_sigmazz->scale(area_scale);
  modelRegDts_sigmaxz->scale(area_scale);
	modelRegDts_sigmaxy->scale(area_scale);
	modelRegDts_sigmayz->scale(area_scale);

	/* Propagate */
	if (_domDec == 0){
		propElasticFwdGpu_3D(modelRegDts_vx->getVals(), modelRegDts_vy->getVals(), modelRegDts_vz->getVals(), modelRegDts_sigmaxx->getVals(), modelRegDts_sigmayy->getVals(), modelRegDts_sigmazz->getVals(), modelRegDts_sigmaxz->getVals(), modelRegDts_sigmaxy->getVals(), modelRegDts_sigmayz->getVals(), dataRegDts_vx->getVals(), dataRegDts_vy->getVals(), dataRegDts_vz->getVals(), dataRegDts_sigmaxx->getVals(), dataRegDts_sigmayy->getVals(), dataRegDts_sigmazz->getVals(), dataRegDts_sigmaxz->getVals(), dataRegDts_sigmaxy->getVals(), dataRegDts_sigmayz->getVals(), _sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid, _sourcesPositionRegXGrid, _nSourcesRegXGrid, _sourcesPositionRegYGrid, _nSourcesRegYGrid, _sourcesPositionRegZGrid, _nSourcesRegZGrid, _sourcesPositionRegXZGrid, _nSourcesRegXZGrid, _sourcesPositionRegXYGrid, _nSourcesRegXYGrid, _sourcesPositionRegYZGrid, _nSourcesRegYZGrid, _receiversPositionRegCenterGrid, _nReceiversRegCenterGrid, _receiversPositionRegXGrid, _nReceiversRegXGrid, _receiversPositionRegYGrid, _nReceiversRegYGrid, _receiversPositionRegZGrid, _nReceiversRegZGrid, _receiversPositionRegXZGrid, _nReceiversRegXZGrid, _receiversPositionRegXYGrid, _nReceiversRegXYGrid, _receiversPositionRegYZGrid, _nReceiversRegYZGrid, _fdParamElastic->_nx, _fdParamElastic->_ny, _fdParamElastic->_nz, _iGpu, _iGpuId);
	} else {
		throw std::runtime_error("ERROR! Domain decomposition not implemented yet!");
	}

	/* Interpolate to irregular grid */
	_receiversXGrid->forward(true, dataRegDts_vx, dataIrreg_vx);
	_receiversYGrid->forward(true, dataRegDts_vy, dataIrreg_vy);
  _receiversZGrid->forward(true, dataRegDts_vz, dataIrreg_vz);
  _receiversCenterGrid->forward(true, dataRegDts_sigmaxx, dataIrreg_sigmaxx);
	_receiversCenterGrid->forward(true, dataRegDts_sigmayy, dataIrreg_sigmayy);
  _receiversCenterGrid->forward(true, dataRegDts_sigmazz, dataIrreg_sigmazz);
  _receiversXZGrid->forward(true, dataRegDts_sigmaxz, dataIrreg_sigmaxz);
	_receiversXYGrid->forward(true, dataRegDts_sigmaxy, dataIrreg_sigmaxy);
	_receiversYZGrid->forward(true, dataRegDts_sigmayz, dataIrreg_sigmayz);

	/* Copy each component data into one cube */
	// Vx
	shift = 0;
	chunk = _nReceiversIrregXGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals(), dataIrreg_vx->getVals(), chunk*sizeof(double) );
	// Vy
	shift += chunk;
	chunk = _nReceiversIrregYGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals()+shift, dataIrreg_vy->getVals(), chunk*sizeof(double) );
	// Vz
	shift += chunk;
	chunk = _nReceiversIrregZGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals()+shift, dataIrreg_vz->getVals(), chunk*sizeof(double) );
	// Sigmaxx
	shift += chunk;
	chunk = _nReceiversIrregCenterGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals()+shift, dataIrreg_sigmaxx->getVals(), chunk*sizeof(double) );
	// Sigmayy
	shift += chunk;
	chunk = _nReceiversIrregCenterGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals()+shift, dataIrreg_sigmayy->getVals(), chunk*sizeof(double) );
	// Sigmazz
	shift += chunk;
	chunk = _nReceiversIrregCenterGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals()+shift, dataIrreg_sigmazz->getVals(), chunk*sizeof(double) );
	// Sigmaxz
	shift += chunk;
	chunk = _nReceiversIrregXZGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals()+shift, dataIrreg_sigmaxz->getVals(), chunk*sizeof(double) );
	// Sigmaxy
	shift += chunk;
	chunk = _nReceiversIrregXYGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals()+shift, dataIrreg_sigmaxy->getVals(), chunk*sizeof(double) );
	// Sigmayz
	shift += chunk;
	chunk = _nReceiversIrregYZGrid*_fdParamElastic->_nts;
	std::memcpy(data->getVals()+shift, dataIrreg_sigmayz->getVals(), chunk*sizeof(double) );

}

void nonlinearPropElasticGpu_3D::adjoint(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

}
