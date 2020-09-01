template <class V1, class V2>
void seismicElasticOperator3D <V1, V2>::setSources_3D(std::shared_ptr<spaceInterpGpu_3D> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYZGrid){

	_sourcesCenterGrid = sourcesCenterGrid;
	_sourcesXGrid = sourcesXGrid;
	_sourcesYGrid = sourcesYGrid;
	_sourcesZGrid = sourcesZGrid;
	_sourcesXZGrid = sourcesXZGrid;
	_sourcesXYGrid = sourcesXYGrid;
	_sourcesYZGrid = sourcesYZGrid;

	_nSourcesRegCenterGrid = _sourcesCenterGrid->getNDeviceReg();
	_nSourcesRegXGrid = _sourcesXGrid->getNDeviceReg();
	_nSourcesRegYGrid = _sourcesYGrid->getNDeviceReg();
	_nSourcesRegZGrid = _sourcesZGrid->getNDeviceReg();
	_nSourcesRegXZGrid = _sourcesXZGrid->getNDeviceReg();
	_nSourcesRegXYGrid = _sourcesXYGrid->getNDeviceReg();
	_nSourcesRegYZGrid = _sourcesYZGrid->getNDeviceReg();

  _nSourcesIrregCenterGrid = _sourcesCenterGrid->getNDeviceIrreg();
	_nSourcesIrregXGrid = _sourcesXGrid->getNDeviceIrreg();
	_nSourcesIrregYGrid = _sourcesYGrid->getNDeviceIrreg();
	_nSourcesIrregZGrid = _sourcesZGrid->getNDeviceIrreg();
	_nSourcesIrregXZGrid = _sourcesXZGrid->getNDeviceIrreg();
	_nSourcesIrregXYGrid = _sourcesXYGrid->getNDeviceIrreg();
	_nSourcesIrregYZGrid = _sourcesYZGrid->getNDeviceIrreg();

	_sourcesPositionRegCenterGrid = _sourcesCenterGrid->getRegPosUnique();
	_sourcesPositionRegXGrid = _sourcesXGrid->getRegPosUnique();
	_sourcesPositionRegYGrid = _sourcesYGrid->getRegPosUnique();
	_sourcesPositionRegZGrid = _sourcesZGrid->getRegPosUnique();
	_sourcesPositionRegXZGrid = _sourcesXZGrid->getRegPosUnique();
	_sourcesPositionRegXYGrid = _sourcesXYGrid->getRegPosUnique();
	_sourcesPositionRegYZGrid = _sourcesYZGrid->getRegPosUnique();

}

// Sources setup for Born
template <class V1, class V2>
void seismicElasticOperator3D <V1, V2>::setSources_3D(std::shared_ptr<spaceInterpGpu_3D> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYZGrid, std::shared_ptr<V2> sourcesSignals){

	_sourcesCenterGrid = sourcesCenterGrid;
	_sourcesXGrid = sourcesXGrid;
	_sourcesYGrid = sourcesYGrid;
	_sourcesZGrid = sourcesZGrid;
	_sourcesXZGrid = sourcesXZGrid;
	_sourcesXYGrid = sourcesXYGrid;
	_sourcesYZGrid = sourcesYZGrid;

	_nSourcesRegCenterGrid = _sourcesCenterGrid->getNDeviceReg();
	_nSourcesRegXGrid = _sourcesXGrid->getNDeviceReg();
	_nSourcesRegYGrid = _sourcesYGrid->getNDeviceReg();
	_nSourcesRegZGrid = _sourcesZGrid->getNDeviceReg();
	_nSourcesRegXZGrid = _sourcesXZGrid->getNDeviceReg();
	_nSourcesRegXYGrid = _sourcesXYGrid->getNDeviceReg();
	_nSourcesRegYZGrid = _sourcesYZGrid->getNDeviceReg();

  _nSourcesIrregCenterGrid = _sourcesCenterGrid->getNDeviceIrreg();
	_nSourcesIrregXGrid = _sourcesXGrid->getNDeviceIrreg();
	_nSourcesIrregYGrid = _sourcesYGrid->getNDeviceIrreg();
	_nSourcesIrregZGrid = _sourcesZGrid->getNDeviceIrreg();
	_nSourcesIrregXZGrid = _sourcesXZGrid->getNDeviceIrreg();
	_nSourcesIrregXYGrid = _sourcesXYGrid->getNDeviceIrreg();
	_nSourcesIrregYZGrid = _sourcesYZGrid->getNDeviceIrreg();

	_sourcesPositionRegCenterGrid = _sourcesCenterGrid->getRegPosUnique();
	_sourcesPositionRegXGrid = _sourcesXGrid->getRegPosUnique();
	_sourcesPositionRegYGrid = _sourcesYGrid->getRegPosUnique();
	_sourcesPositionRegZGrid = _sourcesZGrid->getRegPosUnique();
	_sourcesPositionRegXZGrid = _sourcesXZGrid->getRegPosUnique();
	_sourcesPositionRegXYGrid = _sourcesXYGrid->getRegPosUnique();
	_sourcesPositionRegYZGrid = _sourcesYZGrid->getRegPosUnique();

	//Constructing source term
	_sourcesSignals = sourcesSignals->clone();

	/* Allocation */
  std::shared_ptr<double2DReg> sourceIrreg_vx(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXGrid));
	std::shared_ptr<double2DReg> sourceIrreg_vy(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregYGrid));
  std::shared_ptr<double2DReg> sourceIrreg_vz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregZGrid));
  std::shared_ptr<double2DReg> sourceIrreg_sigmaxx(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
	std::shared_ptr<double2DReg> sourceIrreg_sigmayy(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
  std::shared_ptr<double2DReg> sourceIrreg_sigmazz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregCenterGrid));
  std::shared_ptr<double2DReg> sourceIrreg_sigmaxz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXZGrid));
	std::shared_ptr<double2DReg> sourceIrreg_sigmaxy(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregXYGrid));
	std::shared_ptr<double2DReg> sourceIrreg_sigmayz(new double2DReg(_fdParamElastic->_nts, _nSourcesIrregYZGrid));

	_sourceRegDtw_vx = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegXGrid);
	_sourceRegDtw_vy = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegYGrid);
	_sourceRegDtw_vz = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegZGrid);
	_sourceRegDtw_sigmaxx = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegCenterGrid);
	_sourceRegDtw_sigmayy = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegCenterGrid);
	_sourceRegDtw_sigmazz = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegCenterGrid);
	_sourceRegDtw_sigmaxz = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegXZGrid);
	_sourceRegDtw_sigmaxy = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegXYGrid);
	_sourceRegDtw_sigmayz = std::make_shared<double2DReg>(_fdParamElastic->_nts, _nSourcesRegYZGrid);

	/* Copy from 3d source to respective 2d source components */
	// fx
	long long shift;
	long long chunk = _nSourcesIrregXGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_vx->getVals(), _sourcesSignals->getVals(), chunk*sizeof(double));
	// fy
	shift = chunk;
	chunk = _nSourcesIrregYGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_vy->getVals(), _sourcesSignals->getVals()+shift, chunk*sizeof(double));
	// fz
	shift += chunk;
	chunk = _nSourcesIrregZGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_vz->getVals(), _sourcesSignals->getVals()+shift, chunk*sizeof(double));
	// mxx
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_sigmaxx->getVals(), _sourcesSignals->getVals()+shift, chunk*sizeof(double));
	// myy
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_sigmayy->getVals(), _sourcesSignals->getVals()+shift, chunk*sizeof(double));
	// mzz
	shift += chunk;
	chunk = _nSourcesIrregCenterGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_sigmazz->getVals(), _sourcesSignals->getVals()+shift, chunk*sizeof(double));
	// mxz
	shift += chunk;
	chunk = _nSourcesIrregXZGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_sigmaxz->getVals(), _sourcesSignals->getVals()+shift, chunk*sizeof(double));
	// mxz
	shift += chunk;
	chunk = _nSourcesIrregXYGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_sigmaxy->getVals(), _sourcesSignals->getVals()+shift, chunk*sizeof(double));
	// myz
	shift += chunk;
	chunk = _nSourcesIrregYZGrid*_fdParamElastic->_nts;
  std::memcpy(sourceIrreg_sigmayz->getVals(), _sourcesSignals->getVals()+shift, chunk*sizeof(double));

	/* Interpolate source (seismic source) to regular grid */
	_sourcesXGrid->adjoint(false, _sourceRegDtw_vx, sourceIrreg_vx);
	_sourcesYGrid->adjoint(false, _sourceRegDtw_vy, sourceIrreg_vy);
	_sourcesZGrid->adjoint(false, _sourceRegDtw_vz, sourceIrreg_vz);
	_sourcesCenterGrid->adjoint(false, _sourceRegDtw_sigmaxx, sourceIrreg_sigmaxx);
	_sourcesCenterGrid->adjoint(false, _sourceRegDtw_sigmayy, sourceIrreg_sigmayy);
	_sourcesCenterGrid->adjoint(false, _sourceRegDtw_sigmazz, sourceIrreg_sigmazz);
	_sourcesXZGrid->adjoint(false, _sourceRegDtw_sigmaxz, sourceIrreg_sigmaxz);
	_sourcesXYGrid->adjoint(false, _sourceRegDtw_sigmaxy, sourceIrreg_sigmaxy);
	_sourcesYZGrid->adjoint(false, _sourceRegDtw_sigmayz, sourceIrreg_sigmayz);

	/* Scale source signals */
	/* Scale source signals model */
	// fx
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegXGrid; is++){ //loop over number of reg sources x grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*_sourceRegDtw_vx->_mat)[is][it] *= _fdParamElastic->_rhoxDtw[(_sourcesXGrid->getRegPosUnique())[is]];
		}
  }
	// fy
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegYGrid; is++){ //loop over number of reg sources y grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*_sourceRegDtw_vy->_mat)[is][it] *= _fdParamElastic->_rhoyDtw[(_sourcesYGrid->getRegPosUnique())[is]];
		}
  }
	// fz
  #pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegZGrid; is++){ //loop over number of reg sources z grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*_sourceRegDtw_vz->_mat)[is][it] *= _fdParamElastic->_rhozDtw[(_sourcesZGrid->getRegPosUnique())[is]];
		}
  }

	// moment tensor components
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegCenterGrid; is++){ //loop over number of reg sources central grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
			double mxx = 0.0;
			double myy = 0.0;
			double mzz = 0.0;
			// mxx
	  	mxx = (*_sourceRegDtw_sigmaxx->_mat)[is][it] * _fdParamElastic->_lamb2MuDtw[(_sourcesCenterGrid->getRegPosUnique())[is]]+
								((*_sourceRegDtw_sigmayy->_mat)[is][it] + (*_sourceRegDtw_sigmazz->_mat)[is][it] ) * _fdParamElastic->_lambDtw[(_sourcesCenterGrid->getRegPosUnique())[is]];
			// myy
			myy = (*_sourceRegDtw_sigmayy->_mat)[is][it] * _fdParamElastic->_lamb2MuDtw[(_sourcesCenterGrid->getRegPosUnique())[is]]+
								((*_sourceRegDtw_sigmaxx->_mat)[is][it] + (*_sourceRegDtw_sigmazz->_mat)[is][it] ) * _fdParamElastic->_lambDtw[(_sourcesCenterGrid->getRegPosUnique())[is]];
			// mzz
			mzz = (*_sourceRegDtw_sigmazz->_mat)[is][it] * _fdParamElastic->_lamb2MuDtw[(_sourcesCenterGrid->getRegPosUnique())[is]]+
								((*_sourceRegDtw_sigmaxx->_mat)[is][it] + (*_sourceRegDtw_sigmayy->_mat)[is][it] ) * _fdParamElastic->_lambDtw[(_sourcesCenterGrid->getRegPosUnique())[is]];
		  // Setting values
			(*_sourceRegDtw_sigmaxx->_mat)[is][it] = mxx;
			(*_sourceRegDtw_sigmayy->_mat)[is][it] = myy;
			(*_sourceRegDtw_sigmazz->_mat)[is][it] = mzz;
		}
  }
	// mxz
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegXZGrid; is++){ //loop over number of reg sources xz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*_sourceRegDtw_sigmaxz->_mat)[is][it] *= _fdParamElastic->_muxzDtw[(_sourcesXZGrid->getRegPosUnique())[is]];
		}
  }
	// mxy
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegXYGrid; is++){ //loop over number of reg sources xy grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*_sourceRegDtw_sigmaxy->_mat)[is][it] *= _fdParamElastic->_muxyDtw[(_sourcesXYGrid->getRegPosUnique())[is]];
		}
  }
	// myz
	#pragma omp parallel for collapse(2)
  for(long long is = 0; is < _nSourcesRegYZGrid; is++){ //loop over number of reg sources yz grid
		for(int it = 0; it < _fdParamElastic->_nts; it++){ //loop over time steps
	  		(*_sourceRegDtw_sigmayz->_mat)[is][it] *= _fdParamElastic->_muyzDtw[(_sourcesYZGrid->getRegPosUnique())[is]];
		}
  }

	/*Scaling by the inverse of the space discretization*/
	double area_scale = 1.0/(_fdParamElastic->_dx * _fdParamElastic->_dy * _fdParamElastic->_dz);
	_sourceRegDtw_vx->scale(area_scale);
	_sourceRegDtw_vy->scale(area_scale);
  _sourceRegDtw_vz->scale(area_scale);
  _sourceRegDtw_sigmaxx->scale(area_scale);
	_sourceRegDtw_sigmayy->scale(area_scale);
  _sourceRegDtw_sigmazz->scale(area_scale);
  _sourceRegDtw_sigmaxz->scale(area_scale);
	_sourceRegDtw_sigmaxy->scale(area_scale);
	_sourceRegDtw_sigmayz->scale(area_scale);



}

// Receivers setup for Nonlinear modeling, Born
template <class V1, class V2>
void seismicElasticOperator3D <V1, V2>::setReceivers_3D(std::shared_ptr<spaceInterpGpu_3D> receiversCenterGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYZGrid){

	_receiversCenterGrid = receiversCenterGrid;
	_receiversXGrid = receiversXGrid;
	_receiversYGrid = receiversYGrid;
	_receiversZGrid = receiversZGrid;
	_receiversXZGrid = receiversXZGrid;
	_receiversXYGrid = receiversXYGrid;
	_receiversYZGrid = receiversYZGrid;

	_nReceiversRegCenterGrid = _receiversCenterGrid->getNDeviceReg();
	_nReceiversRegXGrid = _receiversXGrid->getNDeviceReg();
	_nReceiversRegYGrid = _receiversYGrid->getNDeviceReg();
	_nReceiversRegZGrid = _receiversZGrid->getNDeviceReg();
	_nReceiversRegXZGrid = _receiversXZGrid->getNDeviceReg();
	_nReceiversRegXYGrid = _receiversXYGrid->getNDeviceReg();
	_nReceiversRegYZGrid = _receiversYZGrid->getNDeviceReg();

	_nReceiversIrregCenterGrid = _receiversCenterGrid->getNDeviceIrreg();
	_nReceiversIrregXGrid = _receiversXGrid->getNDeviceIrreg();
	_nReceiversIrregYGrid = _receiversYGrid->getNDeviceIrreg();
	_nReceiversIrregZGrid = _receiversZGrid->getNDeviceIrreg();
	_nReceiversIrregXZGrid = _receiversXZGrid->getNDeviceIrreg();
	_nReceiversIrregXYGrid = _receiversXYGrid->getNDeviceIrreg();
	_nReceiversIrregYZGrid = _receiversYZGrid->getNDeviceIrreg();

	_receiversPositionRegCenterGrid = _receiversCenterGrid->getRegPosUnique();
	_receiversPositionRegXGrid = _receiversXGrid->getRegPosUnique();
	_receiversPositionRegYGrid = _receiversYGrid->getRegPosUnique();
	_receiversPositionRegZGrid = _receiversZGrid->getRegPosUnique();
	_receiversPositionRegXZGrid = _receiversXZGrid->getRegPosUnique();
	_receiversPositionRegXYGrid = _receiversXYGrid->getRegPosUnique();
	_receiversPositionRegYZGrid = _receiversYZGrid->getRegPosUnique();
}

// Set acquisiton for Nonlinear modeling
template <class V1, class V2>
void seismicElasticOperator3D <V1, V2>::setAcquisition_3D(std::shared_ptr<spaceInterpGpu_3D> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversCenterGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYZGrid, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
	setSources_3D(sourcesCenterGrid, sourcesXGrid, sourcesYGrid, sourcesZGrid, sourcesXZGrid, sourcesXYGrid, sourcesYZGrid);
	setReceivers_3D(receiversCenterGrid, receiversXGrid, receiversYGrid, receiversZGrid, receiversXZGrid, receiversXYGrid, receiversYZGrid);
	this->setDomainRange(model, data);
	if ( not checkParfileConsistency_3D(model, data)){
		throw std::runtime_error("");
	}
}

// Set acquisiton for Born modeling
template <class V1, class V2>
void seismicElasticOperator3D <V1, V2>::setAcquisition_3D(std::shared_ptr<spaceInterpGpu_3D> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYZGrid, std::shared_ptr<V2> sourcesSignals, std::shared_ptr<spaceInterpGpu_3D> receiversCenterGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYZGrid, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
	setSources_3D(sourcesCenterGrid, sourcesXGrid, sourcesYGrid, sourcesZGrid, sourcesXZGrid, sourcesXYGrid, sourcesYZGrid, sourcesSignals);
	setReceivers_3D(receiversCenterGrid, receiversXGrid, receiversYGrid, receiversZGrid, receiversXZGrid, receiversXYGrid, receiversYZGrid);
	this->setDomainRange(model, data);
	if (not checkParfileConsistency_3D(model, data)){
		throw std::runtime_error("");
	};
}


// Wavefield setup
template <class V1, class V2>
std::shared_ptr<double5DReg> seismicElasticOperator3D <V1, V2>:: setWavefield_3D(int wavefieldFlag){

	_saveWavefield = wavefieldFlag;

	std::shared_ptr<double5DReg> wavefield;
	if (wavefieldFlag == 1) {
		wavefield = std::make_shared<double5DReg>(_fdParamElastic->_zAxis, _fdParamElastic->_yAxis, _fdParamElastic->_xAxis, _fdParamElastic->_wavefieldCompAxis, _fdParamElastic->_timeAxisCoarse);
		unsigned long long int wavefieldSize = _fdParamElastic->_zAxis.n * _fdParamElastic->_yAxis.n * _fdParamElastic->_xAxis.n;
		wavefieldSize *= _fdParamElastic->_wavefieldCompAxis.n *_fdParamElastic->_nts*sizeof(double);
		memset(wavefield->getVals(), 0, wavefieldSize);
		return wavefield;
	}
	else {
		wavefield = std::make_shared<double5DReg>(1, 1, 1, 1, 1);
		unsigned long long int wavefieldSize = 1*sizeof(double);
		memset(wavefield->getVals(), 0, wavefieldSize);
		return wavefield;
	}
}
