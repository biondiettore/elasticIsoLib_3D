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
  for(long long is = 0; is < _nSourcesRegCenterGrid; is++){ //loop over number of reg sources central grid
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
  for(long long is = 0; is < _nSourcesRegXYGrid; is++){ //loop over number of reg sources xy grid
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

	if (!add) {
		model->scale(0.0);
		modelIrreg_vx -> scale(0.0);
		modelIrreg_vy -> scale(0.0);
		modelIrreg_vz -> scale(0.0);
		modelIrreg_sigmaxx -> scale(0.0);
		modelIrreg_sigmayy -> scale(0.0);
		modelIrreg_sigmazz -> scale(0.0);
		modelIrreg_sigmaxz -> scale(0.0);
		modelIrreg_sigmaxy -> scale(0.0);
		modelIrreg_sigmayz -> scale(0.0);
	} else {
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
	}

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

	/* Interpolate to irregular grid */
	_receiversXGrid->adjoint(false, dataRegDts_vx, dataIrreg_vx);
	_receiversYGrid->adjoint(false, dataRegDts_vy, dataIrreg_vy);
  _receiversZGrid->adjoint(false, dataRegDts_vz, dataIrreg_vz);
  _receiversCenterGrid->adjoint(false, dataRegDts_sigmaxx, dataIrreg_sigmaxx);
	_receiversCenterGrid->adjoint(false, dataRegDts_sigmayy, dataIrreg_sigmayy);
  _receiversCenterGrid->adjoint(false, dataRegDts_sigmazz, dataIrreg_sigmazz);
  _receiversXZGrid->adjoint(false, dataRegDts_sigmaxz, dataIrreg_sigmaxz);
	_receiversXYGrid->adjoint(false, dataRegDts_sigmaxy, dataIrreg_sigmaxy);
	_receiversYZGrid->adjoint(false, dataRegDts_sigmayz, dataIrreg_sigmayz);


	/* Scale recorded data */
	// Vx
	#pragma omp parallel for collapse(2)
  for(long long ir = 0; ir < _nReceiversRegXGrid; ir++){ //loop over number of reg receiver x grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*dataRegDts_vx->_mat)[ir][it] *= _fdParamElastic->_rhoxDtw[(_receiversXGrid->getRegPosUnique())[ir]];
		}
  }
	// Vy
	#pragma omp parallel for collapse(2)
  for(long long ir = 0; ir < _nReceiversRegYGrid; ir++){ //loop over number of reg receiver y grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*dataRegDts_vy->_mat)[ir][it] *= _fdParamElastic->_rhoyDtw[(_receiversYGrid->getRegPosUnique())[ir]];
		}
  }
	// Vz
	#pragma omp parallel for collapse(2)
  for(long long ir = 0; ir < _nReceiversRegZGrid; ir++){ //loop over number of reg receiver z grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*dataRegDts_vz->_mat)[ir][it] *= _fdParamElastic->_rhozDtw[(_receiversZGrid->getRegPosUnique())[ir]];
		}
  }

	// Stress tensor components
	#pragma omp parallel for collapse(2)
  for(long long ir = 0; ir < _nReceiversRegCenterGrid; ir++){ //loop over number of reg receiver central grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
			double Sxx = 0;
			double Syy = 0;
			double Szz = 0;
			// Sxx
	  	Sxx = (*dataRegDts_sigmaxx->_mat)[ir][it] * _fdParamElastic->_lamb2MuDtw[(_receiversCenterGrid->getRegPosUnique())[ir]]+
								((*dataRegDts_sigmayy->_mat)[ir][it] + (*dataRegDts_sigmazz->_mat)[ir][it] ) * _fdParamElastic->_lambDtw[(_receiversCenterGrid->getRegPosUnique())[ir]];
			// Syy
			Syy = (*dataRegDts_sigmayy->_mat)[ir][it] * _fdParamElastic->_lamb2MuDtw[(_receiversCenterGrid->getRegPosUnique())[ir]]+
								((*dataRegDts_sigmaxx->_mat)[ir][it] + (*dataRegDts_sigmazz->_mat)[ir][it] ) * _fdParamElastic->_lambDtw[(_receiversCenterGrid->getRegPosUnique())[ir]];
			// Szz
			Szz = (*dataRegDts_sigmazz->_mat)[ir][it] * _fdParamElastic->_lamb2MuDtw[(_receiversCenterGrid->getRegPosUnique())[ir]]+
								((*dataRegDts_sigmaxx->_mat)[ir][it] + (*dataRegDts_sigmayy->_mat)[ir][it] ) * _fdParamElastic->_lambDtw[(_receiversCenterGrid->getRegPosUnique())[ir]];
		  // Setting values
			(*dataRegDts_sigmaxx->_mat)[ir][it] = Sxx;
			(*dataRegDts_sigmayy->_mat)[ir][it] = Syy;
			(*dataRegDts_sigmazz->_mat)[ir][it] = Szz;
		}
  }
	// Sxz
	#pragma omp parallel for collapse(2)
  for(long long ir = 0; ir < _nReceiversRegXZGrid; ir++){ //loop over number of reg receiver xz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*dataRegDts_sigmaxz->_mat)[ir][it] *= _fdParamElastic->_muxzDtw[(_receiversXZGrid->getRegPosUnique())[ir]];
		}
  }
	// Sxy
	#pragma omp parallel for collapse(2)
  for(long long ir = 0; ir < _nReceiversRegXYGrid; ir++){ //loop over number of reg receiver xy grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*dataRegDts_sigmaxy->_mat)[ir][it] *= _fdParamElastic->_muxzDtw[(_receiversXYGrid->getRegPosUnique())[ir]];
		}
  }
	// Syz
	#pragma omp parallel for collapse(2)
  for(long long ir = 0; ir < _nReceiversRegYZGrid; ir++){ //loop over number of reg receiver yz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*dataRegDts_sigmayz->_mat)[ir][it] *= _fdParamElastic->_muxzDtw[(_receiversYZGrid->getRegPosUnique())[ir]];
		}
  }

	/*Scaling by the inverse of the space discretization*/
	double area_scale = 1.0/(_fdParamElastic->_dx * _fdParamElastic->_dy * _fdParamElastic->_dz);
	dataRegDts_vx->scale(area_scale);
	dataRegDts_vy->scale(area_scale);
  dataRegDts_vz->scale(area_scale);
  dataRegDts_sigmaxx->scale(area_scale);
	dataRegDts_sigmayy->scale(area_scale);
  dataRegDts_sigmazz->scale(area_scale);
  dataRegDts_sigmaxz->scale(area_scale);
	dataRegDts_sigmaxy->scale(area_scale);
	dataRegDts_sigmayz->scale(area_scale);

	/* Propagate */
	if (_domDec == 0){
		// propElasticAdjGpu_3D(modelRegDts_vx->getVals(), modelRegDts_vy->getVals(), modelRegDts_vz->getVals(), modelRegDts_sigmaxx->getVals(), modelRegDts_sigmayy->getVals(), modelRegDts_sigmazz->getVals(), modelRegDts_sigmaxz->getVals(), modelRegDts_sigmaxy->getVals(), modelRegDts_sigmayz->getVals(), dataRegDts_vx->getVals(), dataRegDts_vy->getVals(), dataRegDts_vz->getVals(), dataRegDts_sigmaxx->getVals(), dataRegDts_sigmayy->getVals(), dataRegDts_sigmazz->getVals(), dataRegDts_sigmaxz->getVals(), dataRegDts_sigmaxy->getVals(), dataRegDts_sigmayz->getVals(), _sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid, _sourcesPositionRegXGrid, _nSourcesRegXGrid, _sourcesPositionRegYGrid, _nSourcesRegYGrid, _sourcesPositionRegZGrid, _nSourcesRegZGrid, _sourcesPositionRegXZGrid, _nSourcesRegXZGrid, _sourcesPositionRegXYGrid, _nSourcesRegXYGrid, _sourcesPositionRegYZGrid, _nSourcesRegYZGrid, _receiversPositionRegCenterGrid, _nReceiversRegCenterGrid, _receiversPositionRegXGrid, _nReceiversRegXGrid, _receiversPositionRegYGrid, _nReceiversRegYGrid, _receiversPositionRegZGrid, _nReceiversRegZGrid, _receiversPositionRegXZGrid, _nReceiversRegXZGrid, _receiversPositionRegXYGrid, _nReceiversRegXYGrid, _receiversPositionRegYZGrid, _nReceiversRegYZGrid, _fdParamElastic->_nx, _fdParamElastic->_ny, _fdParamElastic->_nz, _iGpu, _iGpuId);
	} else {
		throw std::runtime_error("ERROR! Domain decomposition not implemented yet!");
	}

	/* Interpolate model (seismic source) to regular grid */
  _sourcesXGrid->forward(true, modelRegDts_vx, modelIrreg_vx);
	_sourcesYGrid->forward(true, modelRegDts_vy, modelIrreg_vy);
  _sourcesZGrid->forward(true, modelRegDts_vz, modelIrreg_vz);
  _sourcesCenterGrid->forward(true, modelRegDts_sigmaxx, modelIrreg_sigmaxx);
	_sourcesCenterGrid->forward(true, modelRegDts_sigmayy, modelIrreg_sigmayy);
  _sourcesCenterGrid->forward(true, modelRegDts_sigmazz, modelIrreg_sigmazz);
  _sourcesXZGrid->forward(true, modelRegDts_sigmaxz, modelIrreg_sigmaxz);
	_sourcesXYGrid->forward(true, modelRegDts_sigmaxy, modelIrreg_sigmaxy);
	_sourcesYZGrid->forward(true, modelRegDts_sigmayz, modelIrreg_sigmayz);

	/* Copy from 3d model to respective 2d model components */
	// fx
	chunk = _nSourcesIrregXGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals(), modelIrreg_vx->getVals(), chunk*sizeof(double));
	// fy
	shift = chunk;
	chunk = _nSourcesIrregYGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals()+shift, modelIrreg_vy->getVals(), chunk*sizeof(double));
	// fz
	shift += chunk;
	chunk = _nSourcesIrregZGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals()+shift, modelIrreg_vz->getVals(), chunk*sizeof(double));
	// mxx
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals()+shift, modelIrreg_sigmaxx->getVals(), chunk*sizeof(double));
	// myy
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals()+shift, modelIrreg_sigmayy->getVals(), chunk*sizeof(double));
	// mzz
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals()+shift, modelIrreg_sigmazz->getVals(), chunk*sizeof(double));
	// mxz
	shift += chunk;
	chunk = _nSourcesIrregXZGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals()+shift, modelIrreg_sigmaxz->getVals(), chunk*sizeof(double));
	// mxz
	shift += chunk;
	chunk = _nSourcesIrregXYGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals()+shift, modelIrreg_sigmaxy->getVals(), chunk*sizeof(double));
	// myz
	shift += chunk;
	chunk = _nSourcesIrregYZGrid*_fdParamElastic->_nts;
	std::memcpy(model->getVals()+shift, modelIrreg_sigmayz->getVals(), chunk*sizeof(double));

}
