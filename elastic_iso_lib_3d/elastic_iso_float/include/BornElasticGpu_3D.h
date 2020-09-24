#ifndef BORN_ELASTIC_GPU_3D_H
#define BORN_ELASTIC_GPU_3D_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu_3D.h"
#include "fdParamElastic_3D.h"
#include "seismicElasticOperator3D.h"
#include "BornElasticGpuFunctions_3D.h"

using namespace SEP;
//! Propogates one elastic wavefield for one shot on one gpu.
/*!
 A more elaborate description of the class.
*/
class BornElasticGpu_3D : public seismicElasticOperator3D<SEP::float4DReg, SEP::float3DReg> {

	protected:

		std::shared_ptr<float5DReg> _srcWavefield, _secWavefield;

	public:
	  	//! Constructor.
		BornElasticGpu_3D(std::shared_ptr<fdParamElastic_3D> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);
		//! Constructor for domain decomposition
		BornElasticGpu_3D(std::shared_ptr<fdParamElastic_3D> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, std::vector<int> gpuList, int iGpuAlloc, std::vector<int> ny_domDec);

		//! Mutators.
		void setAllWavefields_3D(int wavefieldFlag);

  	//! QC
		virtual bool checkParfileConsistency_3D(std::shared_ptr<SEP::float4DReg> model, std::shared_ptr<SEP::float3DReg> data) const;

  	//! FWD
  	void forward(const bool add, const std::shared_ptr<float4DReg> model, std::shared_ptr<float3DReg> data) const;

		//! ADJ
		void adjoint(const bool add, std::shared_ptr<float4DReg> model, const std::shared_ptr<float3DReg> data) const;

		//! Desctructor
		~BornElasticGpu_3D(){};

		//! Accesor
		std::shared_ptr<float5DReg> getSrcWavefield() { return _srcWavefield; }
    std::shared_ptr<float5DReg> getSecWavefield() { return _secWavefield; }


};

#endif
