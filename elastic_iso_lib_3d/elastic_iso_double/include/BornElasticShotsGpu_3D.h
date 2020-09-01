#ifndef BORN_SHOTS_GPU_3D_H
#define BORN_SHOTS_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "double4DReg.h"
#include "double5DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu_3D.h"
#include "fdParamElastic_3D.h"
#include "operator.h"

using namespace SEP;

class BornElasticShotsGpu_3D : public Operator<SEP::double4DReg, SEP::double4DReg> {

	private:
		int _nExp, _nGpu, _info, _deviceNumberInfo, _iGpuAlloc, _ginsu, _domDec;
		int _saveWavefield, _wavefieldShotNumber;
		std::shared_ptr<SEP::double4DReg> _elasticParam;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<spaceInterpGpu_3D>> _sourcesVectorCenterGrid, _sourcesVectorXGrid, _sourcesVectorYGrid, _sourcesVectorZGrid, _sourcesVectorXZGrid, _sourcesVectorXYGrid, _sourcesVectorYZGrid;
		std::vector<std::shared_ptr<spaceInterpGpu_3D>> _receiversVectorCenterGrid, _receiversVectorXGrid, _receiversVectorYGrid, _receiversVectorZGrid, _receiversVectorXZGrid, _receiversVectorXYGrid, _receiversVectorYZGrid;
		std::shared_ptr<fdParamElastic_3D> _fdParamElastic;
		std::vector<int> _gpuList, _ny_domDec;
	  std::vector<std::shared_ptr<SEP::double3DReg>> _sourcesSignalsVector;

	protected:
		std::shared_ptr<SEP::double5DReg> _srcWavefield, _secWavefield;

	public:
	  /* Overloaded constructors */
		BornElasticShotsGpu_3D(std::shared_ptr<SEP::double4DReg> elasticParam, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<SEP::double3DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorCenterGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorYZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorCenterGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorYZGrid);

		/* Destructor */
		~BornElasticShotsGpu_3D(){};

		/* Create Gpu list */
		void createGpuIdList_3D();

	    /* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<SEP::double4DReg> model, std::shared_ptr<SEP::double4DReg> data) const;
		// void forwardWavefield(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double4DReg> data);
		void adjoint(const bool add, std::shared_ptr<SEP::double4DReg> model, const std::shared_ptr<SEP::double4DReg> data) const;
		// void adjointWavefield(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double4DReg> data);

		/* Accessor */
		std::shared_ptr<double5DReg> getSrcWavefield() { return _srcWavefield; }
		std::shared_ptr<double5DReg> getSecWavefield() { return _secWavefield; }

		/* Mutators */
		void setBackground(std::shared_ptr<double4DReg> elasticParam){ _fdParamElastic = std::make_shared<fdParamElastic_3D>(elasticParam, _par); }

};

#endif
