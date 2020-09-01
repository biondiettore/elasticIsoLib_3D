#include <vector>
#include <omp.h>
#include "BornElasticShotsGpu_3D.h"
// #include "BornElasticGpu_3D.h"
#include <stagger_3D.h>
#include <ctime>


BornElasticShotsGpu_3D::BornElasticShotsGpu_3D(std::shared_ptr<SEP::double4DReg> elasticParam, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<SEP::double3DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorCenterGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorYZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorCenterGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorYZGrid){

    // Setup parameters
  	_par = par;
  	_elasticParam = elasticParam;
		_sourcesSignalsVector = sourcesSignalsVector;
  	_nExp = par->getInt("nExp");
		_ginsu = par->getInt("ginsu",0);
		_domDec = par->getInt("domDec",0);
		createGpuIdList_3D();
  	_info = par->getInt("info", 0);
  	_deviceNumberInfo = par->getInt("deviceNumberInfo", 0);
  	// if( not getGpuInfo_3D(_gpuList, _info, _deviceNumberInfo)){
		// 	throw std::runtime_error("Error in getGpuInfo_3D");
		// }; // Get info on GPU cluster and check that there are enough available GPUs
  	_saveWavefield = _par->getInt("saveWavefield", 0);
  	_wavefieldShotNumber = _par->getInt("wavefieldShotNumber", 0);
  	if (_info == 1 && _saveWavefield == 1){
			std::cerr << "Saving wavefield(s) for shot # " << _wavefieldShotNumber << std::endl;
		}
  	_sourcesVectorCenterGrid = sourcesVectorCenterGrid;
    _sourcesVectorXGrid = sourcesVectorXGrid;
		_sourcesVectorYGrid = sourcesVectorYGrid;
    _sourcesVectorZGrid = sourcesVectorZGrid;
    _sourcesVectorXZGrid = sourcesVectorXZGrid;
		_sourcesVectorXYGrid = sourcesVectorXYGrid;
		_sourcesVectorYZGrid = sourcesVectorYZGrid;

    _receiversVectorCenterGrid = receiversVectorCenterGrid;
    _receiversVectorXGrid = receiversVectorXGrid;
		_receiversVectorYGrid = receiversVectorYGrid;
    _receiversVectorZGrid = receiversVectorZGrid;
    _receiversVectorXZGrid = receiversVectorXZGrid;
		_receiversVectorXYGrid = receiversVectorXYGrid;
		_receiversVectorYZGrid = receiversVectorYZGrid;

    _fdParamElastic = std::make_shared<fdParamElastic_3D>(_elasticParam, _par);

		// Creating domain decomposition information
		if (_domDec == 1){
			if (_nGpu == 1){
				throw std::runtime_error("ERROR![BornElasticShotsGpu_3D] Cannot use domain decomposition on single card!");
			}
			int ny = _fdParamElastic->_ny;
			int yPad = _fdParamElastic->_yPad;
			int fat = _par->getInt("fat",4);
			int ny_chunk = ny/_nGpu;
			if (ny_chunk <= 3*fat && _nGpu > 2 || ny_chunk <= 2*fat && _nGpu == 2 || ny_chunk-2*fat <= _fdParamElastic->_minPad){
				//We must have at least one sample in the internal part
				throw std::runtime_error("ERROR![BornElasticShotsGpu_3D] Decomposition strategy not feasible with ny size; Use less cards");
			}
			if (_info == 1){
				std::cout << "Using domain decomposition with " << _nGpu << " GPUs with ny sizes of: ";
			}
			for (int iGpu=0; iGpu < _nGpu; iGpu++){
				int ny_size;
				if (iGpu == 0){
					ny_size = ny_chunk+fat;
				} else if (iGpu == _nGpu-1) {
					ny_size = ny - ny_chunk*(_nGpu-1)+fat;
				} else {
					ny_size = ny_chunk+2*fat;
				}
				_ny_domDec.push_back(ny_size);
			}
			if (_info == 1){
				for (int iGpu=0; iGpu < _nGpu; iGpu++){std::cout << _ny_domDec[iGpu] << " ";}
				std::cout << std::endl;
			}
			//Enable P2P memcpy
			// setGpuP2P(_nGpu, par->getInt("info", 0), _gpuList);

		}

}


void BornElasticShotsGpu_3D::createGpuIdList_3D(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);

	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){
		std::cout << "**** ERROR [BornElasticShotsGpu_3D]: Please provide a list of GPUs to be used ****" << std::endl;
		throw std::runtime_error("");
	}

	// If user does not provide a valid list but provides nGpu -> use id: 0,...,nGpu-1
	if (_nGpu>0 && _gpuList[0]<0){
		_gpuList.clear();
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			_gpuList.push_back(iGpu);
		}
	}

	// If the user provides a list -> use that list and ignore nGpu for the parfile
	if (_gpuList[0]>=0){
		_nGpu = _gpuList.size();
		sort(_gpuList.begin(), _gpuList.end());
		std::vector<int>::iterator it = std::unique(_gpuList.begin(), _gpuList.end());
		bool isUnique = (it==_gpuList.end());
		if (isUnique==0) {
			std::cout << "**** ERROR [BornElasticShotsGpu_3D]: Please make sure there are no duplicates in the GPU Id list ****" << std::endl; throw std::runtime_error("");
		}
	}

	// Check that the user does not ask for more GPUs than shots to be modeled
	if (_nGpu > _nExp && _domDec == 0){std::cout << "**** ERROR [BornElasticShotsGpu_3D]: User required more GPUs than shots to be modeled ****" << std::endl; throw std::runtime_error("");}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

void BornElasticShotsGpu_3D::forward(const bool add, const std::shared_ptr<double4DReg> model, std::shared_ptr<double4DReg> data) const{
	if (!add) data->scale(0.0);

	if (_domDec == 0){
		// Not using domain decomposition
		// Variable declaration
		int omp_get_thread_num();
		int constantSrcSignal, constantRecGeom;

		// Check whether we use the same source signals for all shots
    if (_sourcesSignalsVector.size() == 1) {constantSrcSignal = 1;}
    else {constantSrcSignal=0;}

    // Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
    if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom = 1;}
    else {constantRecGeom=0;}

		// Create vectors for each GPU
    std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3), SEP::axis(8))); //This model slice contains the staggered and scaled model perturbations (drhox, drhoy, drhoz, dlame, dmu, dmuxz, dmuxy, dmuyz)
    std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
    std::vector<std::shared_ptr<double4DReg>> modelSlicesVector;
    std::vector<std::shared_ptr<double3DReg>> dataSlicesVector;
    // std::vector<std::shared_ptr<BornElasticGpu_3D>> BornObjectVector;

		//Staggering and scaling input model perturbations
    std::shared_ptr<double3DReg> temp_stag(new double3DReg(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3)));
    std::shared_ptr<double3DReg> temp_stag1(new double3DReg(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3)));
    std::shared_ptr<SEP::double4DReg> modelSlice(new SEP::double4DReg(hyperModelSlice));

		//stagger 3d density, mu
    std::shared_ptr<staggerX> staggerXop(new staggerX(temp_stag,temp_stag1));
		std::shared_ptr<staggerY> staggerYop(new staggerY(temp_stag,temp_stag1));
    std::shared_ptr<staggerZ> staggerZop(new staggerZ(temp_stag,temp_stag1));

		// Model size
		int nx = _fdParamElastic->_nx;
		int ny = _fdParamElastic->_ny;
		int nz = _fdParamElastic->_nz;
		long long nModel = nx;
		nModel *= ny * nz;

    //Density perturbation
    //drho_x
    std::memcpy(temp_stag->getVals(), model->getVals(), nModel*sizeof(double));
    staggerXop->adjoint(false, temp_stag1, temp_stag);
    std::memcpy(modelSlice->getVals(), temp_stag1->getVals(), nModel*sizeof(double));

    //drho_y
    staggerYop->adjoint(false, temp_stag1, temp_stag);
    std::memcpy(modelSlice->getVals()+nModel, temp_stag1->getVals(), nModel*sizeof(double));

		//drho_z
    staggerZop->adjoint(false, temp_stag1, temp_stag);
    std::memcpy(modelSlice->getVals()+2*nModel, temp_stag1->getVals(), nModel*sizeof(double));

    //dlame
    std::memcpy(modelSlice->getVals()+3*nModel, model->getVals()+nModel, nModel*sizeof(double) );

    //Shear modulus perturbations
    //dmu
    std::memcpy(modelSlice->getVals()+4*nModel, model->getVals()+2*nModel, nModel*sizeof(double) );
    //dmu_xz
    std::memcpy(temp_stag->getVals(), model->getVals()+2*nModel, nModel*sizeof(double) );
    staggerXop->adjoint(false, temp_stag1, temp_stag);
    staggerZop->adjoint(false, temp_stag, temp_stag1);
    std::memcpy(modelSlice->getVals()+5*nModel, temp_stag->getVals(), nModel*sizeof(double) );
		//dmu_xy
    std::memcpy(temp_stag->getVals(), model->getVals()+2*nModel, nModel*sizeof(double) );
    staggerXop->adjoint(false, temp_stag1, temp_stag);
    staggerYop->adjoint(false, temp_stag, temp_stag1);
    std::memcpy(modelSlice->getVals()+6*nModel, temp_stag->getVals(), nModel*sizeof(double) );
		//dmu_yz
    std::memcpy(temp_stag->getVals(), model->getVals()+2*nModel, nModel*sizeof(double) );
    staggerYop->adjoint(false, temp_stag1, temp_stag);
    staggerZop->adjoint(false, temp_stag, temp_stag1);
    std::memcpy(modelSlice->getVals()+7*nModel, temp_stag->getVals(), nModel*sizeof(double) );

		//Scaling of the perturbations
    #pragma omp for collapse(3)
		for (long long iy = 0; iy < ny; iy++){
    	for (long long ix = 0; ix < nx; ix++){
    		for (long long iz = 0; iz < nz; iz++) {
    			(*modelSlice->_mat)[0][iy][ix][iz] *= (*_fdParamElastic->_rhoxDtwReg->_mat)[iy][ix][iz];
					(*modelSlice->_mat)[1][iy][ix][iz] *= (*_fdParamElastic->_rhoyDtwReg->_mat)[iy][ix][iz];
    			(*modelSlice->_mat)[2][iy][ix][iz] *= (*_fdParamElastic->_rhozDtwReg->_mat)[iy][ix][iz];
	    		// (*modelSlice->_mat)[3][iy][ix][iz] *= 2.0*_fdParamElastic->_dtw; // No need to do this scaling; done in the kernel
	    		// (*modelSlice->_mat)[4][iy][ix][iz] *= 2.0*_fdParamElastic->_dtw; // No need to do this scaling; done in the kernel
	    		(*modelSlice->_mat)[5][iy][ix][iz] *= (*_fdParamElastic->_muxzDtwReg->_mat)[iy][ix][iz];
					(*modelSlice->_mat)[6][iy][ix][iz] *= (*_fdParamElastic->_muxyDtwReg->_mat)[iy][ix][iz];
					(*modelSlice->_mat)[7][iy][ix][iz] *= (*_fdParamElastic->_muyzDtwReg->_mat)[iy][ix][iz];
    		}
    	}
		}

		// Initialization for each GPU:
  	// (1) Creation of vector of objects, model, and data.
  	// (2) Memory allocation on GPU
  	for (int iGpu=0; iGpu<_nGpu; iGpu++){

  		// Born object
  		std::shared_ptr<BornElasticGpu_3D> BornGpuObject(new BornElasticGpu_3D(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
  		BornObjectVector.push_back(BornGpuObject);

  		// Display finite-difference parameters info
  		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
  			BornGpuObject->getFdParam()->getInfo();
  		}

  		// Model slice
  		modelSlicesVector.push_back(modelSlice);

  		// Data slice
  		std::shared_ptr<SEP::double3DReg> dataSlices(new SEP::double3DReg(hyperDataSlices));
  		dataSlicesVector.push_back(dataSlices);

  	}

		//will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
    #pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
    for (int iExp=0; iExp<_nExp; iExp++){

			int iGpu = omp_get_thread_num();
  	  int iGpuId = _gpuList[iGpu];

      // Set acquisition geometry
  	  if ( (constantRecGeom == 1) && (constantSrcSignal == 1) || (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
            BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _sourcesSignalsVector[0], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorYGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0], _receiversVectorXYGrid[0], _receiversVectorYZGrid[0], modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
  	  }
  	  if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
            BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _sourcesSignalsVector[0], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorYGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp], _receiversVectorXYGrid[iExp], _receiversVectorYZGrid[iExp], modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
  	  }
  	  if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
            BornObjectVector[iGpu]->setAcquisition(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _sourcesSignalsVector[iGpu], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorYGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp], _receiversVectorXYGrid[iExp], _receiversVectorYZGrid[iExp], modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
  	  }

      // Set GPU number for propagator object
      BornObjectVector[iGpu]->setGpuNumber(iGpu,iGpuId);

      //Launch modeling
      BornObjectVector[iGpu]->forward(false, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);

      // Store dataSlice into data
      #pragma omp parallel for collapse(3)
      for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
        for (int iReceiver=0; iReceiver<hyperDataSlices->getAxis(2).n; iReceiver++){
          for (int its=0; its<hyperDataSlices->getAxis(1).n; its++){
            (*data->_mat)[iExp][iwc][iReceiver][its] += (*dataSlicesVector[iGpu]->_mat)[iwc][iReceiver][its];
          }
        }
      }

		}

		// Deallocate memory on device
    for (int iGpu=0; iGpu<_nGpu; iGpu++){
      deallocateBornElasticGpu_3D(iGpu,_gpuList[iGpu],_useStreams);
    }

	} else {

	}
}

void BornElasticShotsGpu_3D::adjoint(const bool add, const std::shared_ptr<double4DReg> model, std::shared_ptr<double4DReg> data) const{
	if (!add) model->scale(0.0);

	if (_domDec == 0){

	} else {

	}
}
