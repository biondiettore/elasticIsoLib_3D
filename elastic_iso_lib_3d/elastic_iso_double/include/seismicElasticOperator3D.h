#ifndef SEISMIC_ELASTIC_OPERATOR_3D_H
#define SEISMIC_ELASTIC_OPERATOR_3D_H 1

#include "operator.h"
#include "double2DReg.h"
#include "double5DReg.h"
#include "ioModes.h"
#include "operator.h"
#include "fdParamElastic_3D.h"
#include "spaceInterpGpu_3D.h"
#include <omp.h>

using namespace SEP;

template <class V1, class V2>
class seismicElasticOperator3D : public Operator <V1, V2> {

	protected:

		std::shared_ptr<fdParamElastic_3D> _fdParamElastic;
		std::shared_ptr<spaceInterpGpu_3D> _sourcesCenterGrid, _sourcesXGrid, _sourcesYGrid, _sourcesZGrid, _sourcesXZGrid, _sourcesXYGrid, _sourcesYZGrid;
		std::shared_ptr<spaceInterpGpu_3D> _receiversCenterGrid, _receiversXGrid, _receiversYGrid, _receiversZGrid, _receiversXZGrid, _receiversXYGrid, _receiversYZGrid;
		long long *_sourcesPositionRegCenterGrid, *_sourcesPositionRegXGrid, *_sourcesPositionRegYGrid, *_sourcesPositionRegZGrid, *_sourcesPositionRegXZGrid, *_sourcesPositionRegXYGrid, *_sourcesPositionRegYZGrid;
		long long *_receiversPositionRegCenterGrid, *_receiversPositionRegXGrid, *_receiversPositionRegYGrid, *_receiversPositionRegZGrid, *_receiversPositionRegXZGrid, *_receiversPositionRegXYGrid, *_receiversPositionRegYZGrid;
		int _nSourcesRegCenterGrid,_nSourcesRegXGrid,_nSourcesRegYGrid,_nSourcesRegZGrid,_nSourcesRegXZGrid,_nSourcesRegXYGrid,_nSourcesRegYZGrid;
		int _nSourcesIrregCenterGrid,_nSourcesIrregXGrid,_nSourcesIrregYGrid,_nSourcesIrregZGrid,_nSourcesIrregXZGrid,_nSourcesIrregXYGrid,_nSourcesIrregYZGrid;
		int _nReceiversRegCenterGrid,_nReceiversRegXGrid,_nReceiversRegYGrid,_nReceiversRegZGrid,_nReceiversRegXZGrid,_nReceiversRegXYGrid,_nReceiversRegYZGrid;
		int _nReceiversIrregCenterGrid,_nReceiversIrregXGrid,_nReceiversIrregYGrid,_nReceiversIrregZGrid,_nReceiversIrregXZGrid,_nReceiversIrregXYGrid,_nReceiversIrregYZGrid;
		int _nts;
		int _saveWavefield,_useStreams;
		int _iGpu, _nGpu, _iGpuId;
		int _domDec;

    //these variables hold all five components of elastic source signal. Should be a 3d reg
		std::shared_ptr<V2> _sourcesSignals;
		std::shared_ptr<double2DReg> _sourceRegDtw_vx, _sourceRegDtw_vy, _sourceRegDtw_vz, _sourceRegDtw_sigmaxx, _sourceRegDtw_sigmayy, _sourceRegDtw_sigmazz, _sourceRegDtw_sigmaxz, _sourceRegDtw_sigmaxy, _sourceRegDtw_sigmayz;

	public:

		// QC
		virtual bool checkParfileConsistency_3D(std::shared_ptr<V1> model, std::shared_ptr<V2> data) const = 0; // Pure virtual: needs to implemented in derived class

		// Sources
		void setSources_3D(std::shared_ptr<spaceInterpGpu_3D> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYZGrid); // This one is for the nonlinear modeling operator
		void setSources_3D(std::shared_ptr<spaceInterpGpu_3D> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYZGrid, std::shared_ptr<V2> sourcesSignals); // For Born

		// Receivers
		void setReceivers_3D(std::shared_ptr<spaceInterpGpu_3D> receiversCenterGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYZGrid);

		// Acquisition
		void setAcquisition_3D(std::shared_ptr<spaceInterpGpu_3D> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversCenterGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYZGrid, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data); // Nonlinear

		void setAcquisition_3D(std::shared_ptr<spaceInterpGpu_3D> sourcesCenterGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXZGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesXYGrid, std::shared_ptr<spaceInterpGpu_3D> sourcesYZGrid, std::shared_ptr<V2> sourcesSignals, std::shared_ptr<spaceInterpGpu_3D> receiversCenterGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXZGrid, std::shared_ptr<spaceInterpGpu_3D> receiversXYGrid, std::shared_ptr<spaceInterpGpu_3D> receiversYZGrid, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data); // Born

		// Other mutators
		void setGpuNumber_3D(int iGpu, int iGpuId){_iGpu = iGpu; _iGpuId = iGpuId;}
		std::shared_ptr<double5DReg> setWavefield_3D(int wavefieldFlag); // Allocates and returns a wavefield if flag = 1
		virtual void setAllWavefields_3D(int wavefieldFlag) = 0; // Allocates all wavefields associated with a seismic operator --> this function has to be implemented by child classes

		// Accessors
		std::shared_ptr<fdParamElastic_3D> getFdParam_3D(){ return _fdParamElastic; }

};

#include "seismicElasticOperator3D.cpp"

#endif
