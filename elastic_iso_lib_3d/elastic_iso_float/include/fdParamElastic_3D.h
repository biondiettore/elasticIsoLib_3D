#ifndef FD_PARAM_ELASTIC_3D_H
#define FD_PARAM_ELASTIC_3D_H 1

#include <string>
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"
#include <iostream>

using namespace SEP;

class fdParamElastic_3D{

	public:

		// Constructor
		/** given a parameter file and an elastic model, ensure the dimensions match the prop will be stable and will avoid dispersion.*/
		fdParamElastic_3D(const std::shared_ptr<float4DReg> elasticParam, const std::shared_ptr<paramObj> par);

    // Ginsu mutator
		// void setFdParamElasticGinsu_3D(std::shared_ptr<SEP::hypercube> elasticParamHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu);

    // Destructor
		~fdParamElastic_3D();

		// QC methods
		/** ensure time axis of traces matches nts from parfile */
		bool checkParfileConsistencyTime_3D(const std::shared_ptr<float3DReg> seismicTraces, int timeAxisIndex, std::string fileToCheck) const;
		/** ensure space axes of model matches those from parfile */
		bool checkParfileConsistencySpace_3D(const std::shared_ptr<float4DReg> model, std::string fileToCheck) const;

		bool checkFdStability_3D(float courantMax=0.45);
		bool checkFdDispersion_3D(float dispersionRatioMin=3.0);
		bool checkModelSize_3D(); // Make sure the domain size (without the FAT) is a multiple of the dimblock size
		void getInfo_3D();

		// Variables
		std::shared_ptr<paramObj> _par;
		std::shared_ptr<float4DReg> _elasticParam, _smallElasticParam; //[rho; lambda; mu] [0; 1; 2]

		axis _timeAxisCoarse, _timeAxisFine, _zAxis, _yAxis, _xAxis, _extAxis, _wavefieldCompAxis;

    // Precomputed scaling dtw / rho_x , dtw / rho_y, dtw / rho_z , (lambda + 2*mu) * dtw , lambda * dtw , mu_xz * dtw, mu_xy * dtw, mu_yz * dtw
    std::shared_ptr<float3DReg> _rhoxDtwReg, _rhoyDtwReg, _rhozDtwReg, _lamb2MuDtwReg, _lambDtwReg, _muxzDtwReg, _muxyDtwReg, _muyzDtwReg;
    //pointers to float arrays holding values. These are later passed to the device.
    float *_rhoxDtw,*_rhoyDtw,*_rhozDtw,*_lamb2MuDtw,*_lambDtw,*_muxzDtw,*_muxyDtw,*_muyzDtw;

		float _errorTolerance;
		float _minVpVs, _maxVpVs, _minDzDxDy, _maxDzDxDy;
		int _nts, _sub, _ntw;
		float _ots, _dts, _otw, _dtw;
		float _Courant, _dispersionRatio;
		int _nz, _ny, _nx;
		unsigned long long _nModel;
    const int _nwc=9;
		int _zPadMinus, _zPadPlus, _xPadMinus, _xPadPlus, _zPad, _xPad, _yPad, _minPad;
		float _dz, _dy, _dx, _oz, _oy, _ox, _fMax;
		int _saveWavefield, _blockSize, _fat;
		float _alphaCos;
    int _freeSurface, _mod_par;

		// Ginsu parameters
		// float *_vel2Dtw2Ginsu, *_reflectivityScaleGinsu;
		// int _nzGinsu, _nxGinsu, _nyGinsu;
		// int _zPadMinusGinsu, _zPadPlusGinsu, _xPadMinusGinsu, _xPadPlusGinsu, _yPadMinusGinsu, _yPadPlusGinsu, _zPadGinsu, _xPadGinsu, _yPadGinsu, _minPadGinsu;
		// float _ozGinsu, _dzGinsu, _oxGinsu, _dxGinsu, _oyGinsu, _dyGinsu;
		// int _izGinsu, _ixGinsu, _iyGinsu;


};


#endif
