#include <vector>
#include <omp.h>
#include "nonlinearPropElasticShotsGpu_3D.h"
#include "nonlinearPropElasticGpu_3D.h"

nonlinearPropElasticShotsGpu_3D::nonlinearPropElasticShotsGpu_3D(std::shared_ptr<SEP::float4DReg> elasticParam, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorCenterGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorYZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorCenterGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorYZGrid){

    // Setup parameters
  	_par = par;
  	_elasticParam = elasticParam;
  	_nExp = par->getInt("nExp");
		_ginsu = par->getInt("ginsu",0);
		_domDec = par->getInt("domDec",0);
		createGpuIdList_3D();
  	_info = par->getInt("info", 0);
  	_deviceNumberInfo = par->getInt("deviceNumberInfo", 0);
  	if( not getGpuInfo_3D(_gpuList, _info, _deviceNumberInfo)){
			throw std::runtime_error("Error in getGpuInfo_3D");
		}; // Get info on GPU cluster and check that there are enough available GPUs
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
				throw std::runtime_error("ERROR![nonlinearPropElasticShotsGpu_3D] Cannot use domain decomposition on single card!");
			}
			int ny = _fdParamElastic->_ny;
			int yPad = _fdParamElastic->_yPad;
			int fat = _par->getInt("fat",4);
			int ny_chunk = ny/_nGpu;
			if (ny_chunk <= 3*fat && _nGpu > 2 || ny_chunk <= 2*fat && _nGpu == 2 || ny_chunk-2*fat <= _fdParamElastic->_minPad){
				//We must have at least one sample in the internal part
				throw std::runtime_error("ERROR![nonlinearPropElasticShotsGpu_3D] Decomposition strategy not feasible with ny size; Use less cards");
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
			setGpuP2P(_nGpu, par->getInt("info", 0), _gpuList);

		}

}

void nonlinearPropElasticShotsGpu_3D::createGpuIdList_3D(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);

	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){
		std::cout << "**** ERROR [nonlinearPropElasticShotsGpu_3D]: Please provide a list of GPUs to be used ****" << std::endl;
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
			std::cout << "**** ERROR [nonlinearPropElasticShotsGpu_3D]: Please make sure there are no duplicates in the GPU Id list ****" << std::endl; throw std::runtime_error("");
		}
	}

	// Check that the user does not ask for more GPUs than shots to be modeled
	if (_nGpu > _nExp && _domDec == 0){std::cout << "**** ERROR [nonlinearPropElasticShotsGpu_3D]: User required more GPUs than shots to be modeled ****" << std::endl; throw std::runtime_error("");}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

void nonlinearPropElasticShotsGpu_3D::forward(const bool add, const std::shared_ptr<float4DReg> model, std::shared_ptr<float4DReg> data) const{
	if (!add) data->zero();

	if (_domDec == 0){
		// Not using domain decomposition
		// Variable declaration
		int omp_get_thread_num();
		int constantSrcSignal, constantRecGeom;

		//check that we have five wavefield componenets
		if (model->getHyper()->getAxis(3).n != 9) {
			throw std::runtime_error("**** ERROR [nonlinearPropElasticShotsGpu_3D]: Number of components in source model different than 9 (fx,fy,fz,mxx,myy,mzz,mxz,mxy,myz) ****");
		}

		// Check whether we use the same source signals for all shots
		if (model->getHyper()->getAxis(4).n == 1) {constantSrcSignal = 1;}
		else {constantSrcSignal=0;}

		// Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
		if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
		else {constantRecGeom=0;}

		// Create vectors for each GPU
		std::shared_ptr<SEP::hypercube> hyperModelSlices(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
		std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
		std::vector<std::shared_ptr<float3DReg>> modelSlicesVector;
		std::vector<std::shared_ptr<float3DReg>> dataSlicesVector;
		std::vector<std::shared_ptr<nonlinearPropElasticGpu_3D>> propObjectVector;

		// Initialization for each GPU:
		// (1) Creation of vector of objects, model, and data.
		// (2) Memory allocation on GPU
		for (int iGpu=0; iGpu<_nGpu; iGpu++){

			// Nonlinear propagator object
			std::shared_ptr<nonlinearPropElasticGpu_3D> propGpuObject(new nonlinearPropElasticGpu_3D(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
			propObjectVector.push_back(propGpuObject);

			// Display finite-difference parameters info
			if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
				propGpuObject->getFdParam_3D()->getInfo_3D();
			}

			// Model slice
			std::shared_ptr<SEP::float3DReg> modelSlices(new SEP::float3DReg(hyperModelSlices));
			modelSlicesVector.push_back(modelSlices);

			// Data slice
			std::shared_ptr<SEP::float3DReg> dataSlices(new SEP::float3DReg(hyperDataSlices));
			dataSlicesVector.push_back(dataSlices);

		}

		// Launch nonlinear forward

		//will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
		#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
		for (int iExp=0; iExp<_nExp; iExp++){

			int iGpu = omp_get_thread_num();
			int iGpuId = _gpuList[iGpu];

			// Copy model slice
			long long sourceLength;
			sourceLength = hyperModelSlices->getAxis(1).n*hyperModelSlices->getAxis(2).n;
			sourceLength *= hyperModelSlices->getAxis(3).n;
			if(constantSrcSignal == 1) {
				memcpy(modelSlicesVector[iGpu]->getVals(), &(model->getVals()[0]), sizeof(float)*sourceLength);
			} else {
				memcpy(modelSlicesVector[iGpu]->getVals(), &(model->getVals()[iExp*sourceLength]), sizeof(float)*sourceLength);
			}
			// Set acquisition geometry
			if (constantRecGeom == 1) {
				propObjectVector[iGpu]->setAcquisition_3D(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorYGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0], _receiversVectorXYGrid[0], _receiversVectorYZGrid[0], modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
			} else {
				propObjectVector[iGpu]->setAcquisition_3D(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorYGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp], _receiversVectorXYGrid[iExp], _receiversVectorYZGrid[iExp], modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
			}

			// Set GPU number for propagator object
			propObjectVector[iGpu]->setGpuNumber_3D(iGpu,iGpuId);

			//Launch modeling
			propObjectVector[iGpu]->forward(false, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);

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

	} else {
		// Using domain decomposition
		// Variable declaration
		int constantSrcSignal, constantRecGeom;

		//check that we have five wavefield componenets
		if (model->getHyper()->getAxis(3).n != 9) {
			throw std::runtime_error("**** ERROR [nonlinearPropElasticShotsGpu_3D]: Number of components in source model different than 9 (fx,fy,fz,mxx,myy,mzz,mxz,mxy,myz) ****");
		}

		// Check whether we use the same source signals for all shots
		if (model->getHyper()->getAxis(4).n == 1) {constantSrcSignal = 1;}
		else {constantSrcSignal=0;}

		// Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
		if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
		else {constantRecGeom=0;}

		// Create temporary model and data slices
		std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
		std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
		// Model slice
		std::shared_ptr<float3DReg> modelSlice(new SEP::float3DReg(hyperModelSlice));
		// Data slice
		std::shared_ptr<float3DReg> dataSlice(new SEP::float3DReg(hyperDataSlice));

		// Nonlinear propagator object
		std::shared_ptr<nonlinearPropElasticGpu_3D> propGpuObject(new nonlinearPropElasticGpu_3D(_fdParamElastic, _par, _nGpu, _gpuList, _iGpuAlloc, _ny_domDec));


		// Display finite-difference parameters info
		if ((_info == 1)){
			propGpuObject->getFdParam_3D()->getInfo_3D();
		}

		// Launch nonlinear forward

		//will loop over number of experiments
		for (int iExp=0; iExp<_nExp; iExp++){

			// Copy model slice
			long long sourceLength;
			sourceLength = hyperModelSlice->getAxis(1).n*hyperModelSlice->getAxis(2).n;
			sourceLength *= hyperModelSlice->getAxis(3).n;
			if(constantSrcSignal == 1) {
				memcpy(modelSlice->getVals(), &(model->getVals()[0]), sizeof(float)*sourceLength);
			} else {
				memcpy(modelSlice->getVals(), &(model->getVals()[iExp*sourceLength]), sizeof(float)*sourceLength);
			}
			// Set acquisition geometry
			if (constantRecGeom == 1) {
				propGpuObject->setAcquisition_3D(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorYGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0], _receiversVectorXYGrid[0], _receiversVectorYZGrid[0], modelSlice, dataSlice);
			} else {
				propGpuObject->setAcquisition_3D(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorYGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp], _receiversVectorXYGrid[iExp], _receiversVectorYZGrid[iExp], modelSlice, dataSlice);
			}


			//Launch modeling
			propGpuObject->forward(false, modelSlice, dataSlice);

			// Store dataSlice into data
			#pragma omp parallel for collapse(3)
			for (int iwc=0; iwc<hyperDataSlice->getAxis(3).n; iwc++){ // wavefield component
				for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
					for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
						(*data->_mat)[iExp][iwc][iReceiver][its] += (*dataSlice->_mat)[iwc][iReceiver][its];
					}
				}
			}

		}

	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateNonlinearElasticGpu_3D(iGpu,_gpuList[iGpu]);
	}
}

void nonlinearPropElasticShotsGpu_3D::adjoint(const bool add, const std::shared_ptr<float4DReg> model, std::shared_ptr<float4DReg> data) const{

	if (!add) model->zero();

	if (_domDec == 0){
		// Variable declaration
		int omp_get_thread_num();
		int constantSrcSignal, constantRecGeom;

		//check that we have five wavefield componenets
		if (model->getHyper()->getAxis(3).n != 9) {
			throw std::runtime_error("**** ERROR [nonlinearPropElasticShotsGpu_3D]: Number of components in source model different than 9 (fx,fy,fz,mxx,myy,mzz,mxz,mxy,myz) ****");
		}

		// Check whether we use the same source signals for all shots
		if (model->getHyper()->getAxis(4).n == 1) {constantSrcSignal = 1;}
		else {constantSrcSignal=0;}

		// Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
		if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
		else {constantRecGeom=0;}

		// Create vectors for each GPU
		std::shared_ptr<SEP::hypercube> hyperModelSlices(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
		std::shared_ptr<SEP::hypercube> hyperDataSlices(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
		std::vector<std::shared_ptr<float3DReg>> modelSlicesVector;
		std::vector<std::shared_ptr<float3DReg>> dataSlicesVector;
		std::vector<std::shared_ptr<nonlinearPropElasticGpu_3D>> propObjectVector;

		// Initialization for each GPU:
		// (1) Creation of vector of objects, model, and data.
		// (2) Memory allocation on GPU
		for (int iGpu=0; iGpu<_nGpu; iGpu++){

			// Nonlinear propagator object
			std::shared_ptr<nonlinearPropElasticGpu_3D> propGpuObject(new nonlinearPropElasticGpu_3D(_fdParamElastic, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
			propObjectVector.push_back(propGpuObject);

			// Display finite-difference parameters info
			if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
				propGpuObject->getFdParam_3D()->getInfo_3D();
			}

			// Model slice
			std::shared_ptr<SEP::float3DReg> modelSlices(new SEP::float3DReg(hyperModelSlices));
			modelSlicesVector.push_back(modelSlices);

			// Data slice
			std::shared_ptr<SEP::float3DReg> dataSlices(new SEP::float3DReg(hyperDataSlices));
			dataSlicesVector.push_back(dataSlices);

		}

		// Launch nonlinear adjoint

		//will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
		#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
		for (int iExp=0; iExp<_nExp; iExp++){

			int iGpu = omp_get_thread_num();
			int iGpuId = _gpuList[iGpu];

			// Copy model slice
			long long dataLength = hyperDataSlices->getAxis(1).n*hyperDataSlices->getAxis(2).n;
			dataLength *= hyperDataSlices->getAxis(3).n;
			// Copy data slice
	    memcpy(dataSlicesVector[iGpu]->getVals(), &(data->getVals()[iExp*dataLength]), sizeof(float)*dataLength);
			// Set acquisition geometry
			if (constantRecGeom == 1) {
				propObjectVector[iGpu]->setAcquisition_3D(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorYGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0], _receiversVectorXYGrid[0], _receiversVectorYZGrid[0], modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
			} else {
				propObjectVector[iGpu]->setAcquisition_3D(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorYGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp], _receiversVectorXYGrid[iExp], _receiversVectorYZGrid[iExp], modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
			}

			// Set GPU number for propagator object
			propObjectVector[iGpu]->setGpuNumber_3D(iGpu,iGpuId);

			// Launch modeling
	      if (constantSrcSignal == 1){
	      	// Stack all shots for the same iGpu (and we need to re-stack everything at the end)
	        propObjectVector[iGpu]->adjoint(true, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
	      }
	      else {
	        // Copy the shot into model slice --> Is there a way to parallelize this?
	        propObjectVector[iGpu]->adjoint(false, modelSlicesVector[iGpu], dataSlicesVector[iGpu]);
	        #pragma omp parallel for collapse(3)
	        for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
	          for (int iSource=0; iSource<hyperModelSlices->getAxis(2).n; iSource++){
	            for (int its=0; its<hyperModelSlices->getAxis(1).n; its++){
	            	(*model->_mat)[iExp][iwc][iSource][its] += (*modelSlicesVector[iGpu]->_mat)[iwc][iSource][its];
	            }
	          }
	        }
	      }

		}

		// If same sources for all shots, stack all shots from all iGpus
	    if (constantSrcSignal == 1){
	      #pragma omp parallel for collapse(4)
	      for (int iwc=0; iwc<hyperDataSlices->getAxis(3).n; iwc++){ // wavefield component
	      	for (int iSource=0; iSource<hyperModelSlices->getAxis(2).n; iSource++){
	      		for (int its=0; its<hyperModelSlices->getAxis(1).n; its++){
	      			for (int iGpu=0; iGpu<_nGpu; iGpu++){
	      				(*model->_mat)[0][iwc][iSource][its]	+= (*modelSlicesVector[iGpu]->_mat)[iwc][iSource][its];
	      			}
	      		}
	      	}
	      }
	    }

		// Deallocate memory on device
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			deallocateNonlinearElasticGpu_3D(iGpu,_gpuList[iGpu]);
		}
	} else {
		// Variable declaration
		int omp_get_thread_num();
		int constantSrcSignal, constantRecGeom;

		//check that we have five wavefield componenets
		if (model->getHyper()->getAxis(3).n != 9) {
			throw std::runtime_error("**** ERROR [nonlinearPropElasticShotsGpu_3D]: Number of components in source model different than 9 (fx,fy,fz,mxx,myy,mzz,mxz,mxy,myz) ****");
		}

		// Check whether we use the same source signals for all shots
		if (model->getHyper()->getAxis(4).n == 1) {constantSrcSignal = 1;}
		else {constantSrcSignal=0;}

		// Check if we have constant receiver geometry. If _receiversVectorCenterGrid size==1 then all receiver vectors should be as well.
		if (_receiversVectorCenterGrid.size() == 1) {constantRecGeom=1;}
		else {constantRecGeom=0;}

		// Create temporary model and data slices
		std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
		std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
		// Model slice
		std::shared_ptr<float3DReg> modelSlice(new SEP::float3DReg(hyperModelSlice));
		// Data slice
		std::shared_ptr<float3DReg> dataSlice(new SEP::float3DReg(hyperDataSlice));

		// Nonlinear propagator object
		std::shared_ptr<nonlinearPropElasticGpu_3D> propGpuObject(new nonlinearPropElasticGpu_3D(_fdParamElastic, _par, _nGpu, _gpuList, _iGpuAlloc, _ny_domDec));

		// Display finite-difference parameters info
		if ((_info == 1)){
			propGpuObject->getFdParam_3D()->getInfo_3D();
		}

		// Launch nonlinear adjoint

		//will loop over number of experiments in parallel. each thread launching one experiment at a time on one gpu.
		for (int iExp=0; iExp<_nExp; iExp++){

			// Copy model slice
			long long dataLength = hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n;
			dataLength *= hyperDataSlice->getAxis(3).n;
			// Copy data slice
	    memcpy(dataSlice->getVals(), &(data->getVals()[iExp*dataLength]), sizeof(float)*dataLength);
			// Set acquisition geometry
			if (constantRecGeom == 1) {
				propGpuObject->setAcquisition_3D(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _receiversVectorCenterGrid[0], _receiversVectorXGrid[0], _receiversVectorYGrid[0], _receiversVectorZGrid[0], _receiversVectorXZGrid[0], _receiversVectorXYGrid[0], _receiversVectorYZGrid[0], modelSlice, dataSlice);
			} else {
				propGpuObject->setAcquisition_3D(_sourcesVectorCenterGrid[iExp], _sourcesVectorXGrid[iExp], _sourcesVectorYGrid[iExp], _sourcesVectorZGrid[iExp], _sourcesVectorXZGrid[iExp], _sourcesVectorXYGrid[iExp], _sourcesVectorYZGrid[iExp], _receiversVectorCenterGrid[iExp], _receiversVectorXGrid[iExp], _receiversVectorYGrid[iExp], _receiversVectorZGrid[iExp], _receiversVectorXZGrid[iExp], _receiversVectorXYGrid[iExp], _receiversVectorYZGrid[iExp], modelSlice, dataSlice);
			}

			// Launch modeling
	      if (constantSrcSignal == 1){
	      	// Stack all shots for the same iGpu (and we need to re-stack everything at the end)
	        propGpuObject->adjoint(true, modelSlice, dataSlice);
	      }
	      else {
	        // Copy the shot into model slice --> Is there a way to parallelize this?
	        propGpuObject->adjoint(false, modelSlice, dataSlice);
	        #pragma omp parallel for collapse(3)
	        for (int iwc=0; iwc<hyperDataSlice->getAxis(3).n; iwc++){ // wavefield component
	          for (int iSource=0; iSource<hyperModelSlice->getAxis(2).n; iSource++){
	            for (int its=0; its<hyperModelSlice->getAxis(1).n; its++){
	            	(*model->_mat)[iExp][iwc][iSource][its] += (*modelSlice->_mat)[iwc][iSource][its];
	            }
	          }
	        }
	      }

		}

		// If same sources for all shots, stack all shots from all iGpus
    if (constantSrcSignal == 1){
      #pragma omp parallel for collapse(3)
      for (int iwc=0; iwc<hyperDataSlice->getAxis(3).n; iwc++){ // wavefield component
      	for (int iSource=0; iSource<hyperModelSlice->getAxis(2).n; iSource++){
      		for (int its=0; its<hyperModelSlice->getAxis(1).n; its++){
      			(*model->_mat)[0][iwc][iSource][its]	+= (*modelSlice->_mat)[iwc][iSource][its];
      		}
      	}
      }
    }

		// Deallocate memory on device
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			deallocateNonlinearElasticGpu_3D(iGpu,_gpuList[iGpu]);
		}
	}

}
