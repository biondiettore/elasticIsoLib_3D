#ifndef NL_PROP_ELASTIC_SHOTS_GPU_3D_H
#define NL_PROP_ELASTIC_SHOTS_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "float4DReg.h"
#include "float5DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu_3D.h"
#include "fdParamElastic_3D.h"
#include "operator.h"

using namespace SEP;

class nonlinearPropElasticShotsGpu_3D : public Operator<SEP::float4DReg, SEP::float4DReg> {

	private:
		int _nExp, _nGpu, _info, _deviceNumberInfo, _iGpuAlloc, _ginsu, _domDec;
		int _saveWavefield, _wavefieldShotNumber;
		std::shared_ptr<SEP::float4DReg> _elasticParam;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<spaceInterpGpu_3D>> _sourcesVectorCenterGrid, _sourcesVectorXGrid, _sourcesVectorYGrid, _sourcesVectorZGrid, _sourcesVectorXZGrid, _sourcesVectorXYGrid, _sourcesVectorYZGrid;
		std::vector<std::shared_ptr<spaceInterpGpu_3D>> _receiversVectorCenterGrid, _receiversVectorXGrid, _receiversVectorYGrid, _receiversVectorZGrid, _receiversVectorXZGrid, _receiversVectorXYGrid, _receiversVectorYZGrid;
		std::shared_ptr<fdParamElastic_3D> _fdParamElastic;
		std::vector<int> _gpuList, _ny_domDec;
	protected:
		std::shared_ptr<SEP::float5DReg> _wavefield;

	public:

		/* Overloaded constructors */
		nonlinearPropElasticShotsGpu_3D(std::shared_ptr<SEP::float4DReg> elasticParam, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorCenterGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorXYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> sourcesVectorYZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorCenterGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXZGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorXYGrid, std::vector<std::shared_ptr<spaceInterpGpu_3D>> receiversVectorYZGrid);

		/* Destructor */
		~nonlinearPropElasticShotsGpu_3D(){};

		/* Create Gpu list */
		void createGpuIdList_3D();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<SEP::float4DReg> model, std::shared_ptr<SEP::float4DReg> data) const;
		void forwardWavefield(const bool add, const std::shared_ptr<float4DReg> model, std::shared_ptr<float4DReg> data);
		void adjoint(const bool add, std::shared_ptr<SEP::float4DReg> model, const std::shared_ptr<SEP::float4DReg> data) const;
		void adjointWavefield(const bool add, std::shared_ptr<float4DReg> model, const std::shared_ptr<float4DReg> data);

		//! Accesor
		std::shared_ptr<float5DReg> getWavefield() { return _wavefield; }

		/* Mutators */
		void setBackground(std::shared_ptr<float4DReg> elasticParam){ _fdParamElastic = std::make_shared<fdParamElastic_3D>(elasticParam, _par); }

};

#endif
