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



void BornElasticGpu_3D::forward(const bool add, const std::shared_ptr<double4DReg> model, std::shared_ptr<double3DReg> data) const {

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
	  std::memcpy(dataIrreg_vx->getVals(), data->getVals(), chunk*sizeof(double));
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

	int nx = _fdParamElastic->_nx;
	int ny = _fdParamElastic->_ny;
	int nz = _fdParamElastic->_nz;
	long long nModel = nx;
	nModel *= ny * nz;

	//Getting already staggered model perturbations
	double *drhox_in = model->getVals();
	double *drhoy_in = model->getVals()+nModel;
	double *drhoz_in = model->getVals()+2*nModel;
	double *dlame_in = model->getVals()+3*nModel;
	double *dmu_in   = model->getVals()+4*nModel;
	double *dmuxz_in = model->getVals()+5*nModel;
	double *dmuxy_in = model->getVals()+6*nModel;
	double *dmuyz_in = model->getVals()+7*nModel;

	// Apply Born operator
	if (_domDec == 0){
		BornElasticFwdGpu_3D(_sourceRegDtw_vx->getVals(), _sourceRegDtw_vy->getVals(), _sourceRegDtw_vz->getVals(), _sourceRegDtw_sigmaxx->getVals(), _sourceRegDtw_sigmayy->getVals(), _sourceRegDtw_sigmazz->getVals(), _sourceRegDtw_sigmaxz->getVals(), _sourceRegDtw_sigmaxy->getVals(), _sourceRegDtw_sigmayz->getVals(), dataRegDts_vx->getVals(), dataRegDts_vy->getVals(), dataRegDts_vz->getVals(), dataRegDts_sigmaxx->getVals(), dataRegDts_sigmayy->getVals(), dataRegDts_sigmazz->getVals(), dataRegDts_sigmaxz->getVals(), dataRegDts_sigmaxy->getVals(), dataRegDts_sigmayz->getVals(), _sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid, _sourcesPositionRegXGrid, _nSourcesRegXGrid, _sourcesPositionRegYGrid, _nSourcesRegYGrid, _sourcesPositionRegZGrid, _nSourcesRegZGrid, _sourcesPositionRegXZGrid, _nSourcesRegXZGrid, _sourcesPositionRegXYGrid, _nSourcesRegXYGrid, _sourcesPositionRegYZGrid, _nSourcesRegYZGrid, _receiversPositionRegCenterGrid, _nReceiversRegCenterGrid, _receiversPositionRegXGrid, _nReceiversRegXGrid, _receiversPositionRegYGrid, _nReceiversRegYGrid, _receiversPositionRegZGrid, _nReceiversRegZGrid, _receiversPositionRegXZGrid, _nReceiversRegXZGrid, _receiversPositionRegXYGrid, _nReceiversRegXYGrid, _receiversPositionRegYZGrid, _nReceiversRegYZGrid, drhox_in, drhoy_in, drhoz_in, dlame_in, dmu_in, dmuxz_in, dmuxy_in, dmuyz_in, nx, ny, nz, _iGpu, _iGpuId);
	} else {
		throw std::logic_error( "Error! Born forward with domain decomposition not implemented yet!" );
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
	long long shift;
	long long chunk = _nReceiversIrregXGrid*_fdParamElastic->_nts;
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


void BornElasticGpu_3D::adjoint(const bool add, const std::shared_ptr<double4DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) {
		model->scale(0.0);
	}

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
	  		(*dataRegDts_sigmaxy->_mat)[ir][it] *= _fdParamElastic->_muxyDtw[(_receiversXYGrid->getRegPosUnique())[ir]];
		}
  }
	// Syz
	#pragma omp parallel for collapse(2)
  for(long long ir = 0; ir < _nReceiversRegYZGrid; ir++){ //loop over number of reg receiver yz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*dataRegDts_sigmayz->_mat)[ir][it] *= _fdParamElastic->_muyzDtw[(_receiversYZGrid->getRegPosUnique())[ir]];
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

	int nx = _fdParamElastic->_nx;
	int ny = _fdParamElastic->_ny;
	int nz = _fdParamElastic->_nz;
	long long nModel = nx;
	nModel *= ny * nz;

	//Getting already staggered model perturbations
	double *drhox_in = model->getVals();
	double *drhoy_in = model->getVals()+nModel;
	double *drhoz_in = model->getVals()+2*nModel;
	double *dlame_in = model->getVals()+3*nModel;
	double *dmu_in   = model->getVals()+4*nModel;
	double *dmuxz_in = model->getVals()+5*nModel;
	double *dmuxy_in = model->getVals()+6*nModel;
	double *dmuyz_in = model->getVals()+7*nModel;

	// Apply Born operator
	if (_domDec == 0){
		BornElasticAdjGpu_3D(_sourceRegDtw_vx->getVals(), _sourceRegDtw_vy->getVals(), _sourceRegDtw_vz->getVals(), _sourceRegDtw_sigmaxx->getVals(), _sourceRegDtw_sigmayy->getVals(), _sourceRegDtw_sigmazz->getVals(), _sourceRegDtw_sigmaxz->getVals(), _sourceRegDtw_sigmaxy->getVals(), _sourceRegDtw_sigmayz->getVals(), dataRegDts_vx->getVals(), dataRegDts_vy->getVals(), dataRegDts_vz->getVals(), dataRegDts_sigmaxx->getVals(), dataRegDts_sigmayy->getVals(), dataRegDts_sigmazz->getVals(), dataRegDts_sigmaxz->getVals(), dataRegDts_sigmaxy->getVals(), dataRegDts_sigmayz->getVals(), _sourcesPositionRegCenterGrid, _nSourcesRegCenterGrid, _sourcesPositionRegXGrid, _nSourcesRegXGrid, _sourcesPositionRegYGrid, _nSourcesRegYGrid, _sourcesPositionRegZGrid, _nSourcesRegZGrid, _sourcesPositionRegXZGrid, _nSourcesRegXZGrid, _sourcesPositionRegXYGrid, _nSourcesRegXYGrid, _sourcesPositionRegYZGrid, _nSourcesRegYZGrid, _receiversPositionRegCenterGrid, _nReceiversRegCenterGrid, _receiversPositionRegXGrid, _nReceiversRegXGrid, _receiversPositionRegYGrid, _nReceiversRegYGrid, _receiversPositionRegZGrid, _nReceiversRegZGrid, _receiversPositionRegXZGrid, _nReceiversRegXZGrid, _receiversPositionRegXYGrid, _nReceiversRegXYGrid, _receiversPositionRegYZGrid, _nReceiversRegYZGrid, drhox_in, drhoy_in, drhoz_in, dlame_in, dmu_in, dmuxz_in, dmuxy_in, dmuyz_in, nx, ny, nz, _iGpu, _iGpuId);
	} else {
		throw std::logic_error( "Error! Born adjoint with domain decomposition not implemented yet!" );
	}

}
