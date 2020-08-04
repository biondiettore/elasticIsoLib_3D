#ifndef NL_PROP_ELASTIC_GPU_3D_H
#define NL_PROP_ELASTIC_GPU_3D_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "double3DReg.h"
#include "double5DReg.h"
#include "ioModes.h"
#include "spaceInterpGpu_3D.h"
#include "fdParamElastic_3D.h"
#include "seismicElasticOperator3D.h"
#include "nonlinearPropElasticGpuFunctions_3D.h"

using namespace SEP;
//! Propogates one elastic wavefield for one shot on one gpu.
/*!
 A more elaborate description of the class.
*/
class nonlinearPropElasticGpu_3D : public seismicElasticOperator3D<SEP::double3DReg, SEP::double3DReg> {

	protected:

		std::shared_ptr<double5DReg> _wavefield;

	public:
    //! Constructor.
		nonlinearPropElasticGpu_3D(std::shared_ptr<fdParamElastic_3D> fdParamElastic, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		//! Mutators.
		void setAllWavefields_3D(int wavefieldFlag);

  	//! QC
		virtual bool checkParfileConsistency_3D(std::shared_ptr<SEP::double3DReg> model, std::shared_ptr<SEP::double3DReg> data) const;

  	//! FWD
  	void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;

	  //! ADJ
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const;

		//! Desctructor
		~nonlinearPropElasticGpu_3D(){};

		//! Accesor
		std::shared_ptr<double5DReg> getWavefield_3D() { return _wavefield; }

};

#endif
