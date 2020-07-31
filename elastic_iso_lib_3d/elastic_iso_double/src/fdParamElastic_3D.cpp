#include <string>
#include <double4DReg.h>
#include "fdParamElastic_3D.h"
#include <math.h>
#include <iomanip>
#include <iostream>
#include <stagger_3D.h>
#include <cstring>
#include "varElaDeclare_3D.h"
using namespace SEP;


fdParamElastic_3D::fdParamElastic_3D(const std::shared_ptr<double4DReg> elasticParam, const std::shared_ptr<paramObj> par) {

	_elasticParam = elasticParam; //[rho; lambda; mu] [0; 1; 2]
	_par = par;

	/***** Coarse time-sampling *****/
	_freeSurface = _par->getInt("freeSurface",0);

	/***** Coarse time-sampling *****/
	_nts = _par->getInt("nts");
	_dts = _par->getFloat("dts",0.0);
	_ots = _par->getFloat("ots", 0.0);
	_sub = _par->getInt("sub");
	_timeAxisCoarse = axis(_nts, _ots, _dts);

	/***** Fine time-sampling *****/
	_sub = _sub*2;
	if(_sub > SUB_MAX) {
		std::cerr << "**** ERROR: 2*_sub is greater than the allowed SUB_MAX value. " << _sub << " > " << SUB_MAX << " ****" << std::endl;
		throw std::runtime_error("");
	}
	_ntw = (_nts - 1) * _sub + 1;
	_dtw = _dts / double(_sub);
	 //since the time stepping is a central difference first order derivative we need to take twice as many time steps as the given sub variable.
	_otw = _ots;
	_timeAxisFine = axis(_ntw, _otw, _dtw);

	/***** Vertical axis *****/
	_nz = _par->getInt("nz");
	_zPadPlus = _par->getInt("zPadPlus");
	_zPadMinus = _par->getInt("zPadMinus");
	if(_freeSurface==0) _zPad = std::min(_zPadMinus, _zPadPlus);
	else if(_freeSurface==1) _zPad = _zPadPlus;
	_mod_par = _par->getInt("mod_par",2);
	_dz = _par->getFloat("dz",-1.0);
	_oz = _elasticParam->getHyper()->getAxis(1).o;
	_zAxis = axis(_nz, _oz, _dz);

	/***** Horizontal x axis *****/
	_nx = _par->getInt("nx");
	_xPadPlus = _par->getInt("xPadPlus");
	_xPadMinus = _par->getInt("xPadMinus");
	_xPad = std::min(_xPadMinus, _xPadPlus);
	_dx = _par->getFloat("dx",-1.0);
	_ox = _elasticParam->getHyper()->getAxis(2).o;
	_xAxis = axis(_nx, _ox, _dx);

	/***** Horizontal x axis *****/
	_ny = _par->getInt("ny");
	_yPadPlus = _par->getInt("yPadPlus");
	_yPadMinus = _par->getInt("yPadMinus");
	_yPad = std::min(_yPadMinus, _yPadPlus);
	_dy = _par->getFloat("dy",-1.0);
	_oy = _elasticParam->getHyper()->getAxis(3).o;
	_yAxis = axis(_ny, _oy, _dy);

	// Number of elements in the model
	_nModel = _nx;
	_nModel = _nModel * _ny * _nz;

	if (_mod_par == 2){
		// Scaling spatial samplings if km were provided
		_dz *= 1000.;
		_oz *= 1000.;
		_ox *= 1000.;
		_dx *= 1000.;
		_oy *= 1000.;
		_dy *= 1000.;
	}

	/***** Wavefield component axis *****/
	_wavefieldCompAxis = axis(_nwc, 0, 1);

	/***** Other parameters *****/
	_fMax = _par->getFloat("fMax",1000.0);
	_blockSize = _par->getInt("blockSize");
	_fat = _par->getInt("fat",4);
	_minPad = std::min(_zPad, _xPad);
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_alphaCos = _par->getFloat("alphaCos", 0.99);
	_errorTolerance = _par->getFloat("errorTolerance", 0.000001);

	/***** Other parameters *****/

	_minVpVs = 10000;
	_maxVpVs = -1;
	//#pragma omp for collapse(2)
	for (int iy = _fat; iy < _ny-2*_fat; iy++){
		for (int ix = _fat; ix < _nx-2*_fat; ix++){
			for (int iz = _fat; iz < _nz-2*_fat; iz++){
				double rhoTemp = (*_elasticParam->_mat)[0][iy][ix][iz];
				double lamdbTemp = (*_elasticParam->_mat)[1][iy][ix][iz];
				double muTemp = (*_elasticParam->_mat)[2][iy][ix][iz];

				double vpTemp = sqrt((lamdbTemp + 2*muTemp)/rhoTemp);
				double vsTemp = sqrt(muTemp/rhoTemp);

				if (vpTemp < _minVpVs) _minVpVs = vpTemp;
				if (vpTemp > _maxVpVs) _maxVpVs = vpTemp;
				if (vsTemp < _minVpVs && vsTemp!=0) _minVpVs = vsTemp;
				if (vsTemp > _maxVpVs) _maxVpVs = vsTemp;
			}
		}
	}

	/***** QC *****/
	if( not checkParfileConsistencySpace_3D(_elasticParam, "Elastic Parameters")){
		throw std::runtime_error("");
	}; // Parfile - velocity file consistency
	if( not checkFdStability_3D()){
		throw std::runtime_error("");
	};
	if ( not checkFdDispersion_3D()){
		throw std::runtime_error("");
	};
	if( not checkModelSize_3D()){
		throw std::runtime_error("");
	};

	/***** Scaling for propagation *****/

	//initialize 3d slices
	_rhoxDtwReg = std::make_shared<double3DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3));
	_rhoyDtwReg = std::make_shared<double3DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3));
	_rhozDtwReg = std::make_shared<double3DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3));
	_lambDtwReg = std::make_shared<double3DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3));
	_lamb2MuDtwReg = std::make_shared<double3DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3));
	_muxzDtwReg = std::make_shared<double3DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3));
	_muxyDtwReg = std::make_shared<double3DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3));
	_muyzDtwReg = std::make_shared<double3DReg>(_elasticParam->getHyper()->getAxis(1), _elasticParam->getHyper()->getAxis(2), _elasticParam->getHyper()->getAxis(3));

	//stagger 3d density, mu
	std::shared_ptr<staggerX> _staggerX(new staggerX(_rhoxDtwReg,_rhoxDtwReg));
	std::shared_ptr<staggerY> _staggerY(new staggerY(_rhoyDtwReg,_rhoyDtwReg));
	std::shared_ptr<staggerZ> _staggerZ(new staggerZ(_rhozDtwReg,_rhozDtwReg));

	//slice _elasticParam into 3d
	std::memcpy( _lambDtwReg->getVals(), _elasticParam->getVals(), _nModel*sizeof(double) );
	std::memcpy( _lamb2MuDtwReg->getVals(), _elasticParam->getVals()+2*_nModel, _nModel*sizeof(double) );

	//_lambDtwReg holds density. _rhoxDtwReg, _rhoyDtwReg, _rhozDtwReg are empty.
	_staggerX->adjoint(0, _rhoxDtwReg, _lambDtwReg);
	_staggerY->adjoint(0, _rhoyDtwReg, _lambDtwReg);
	_staggerZ->adjoint(0, _rhozDtwReg, _lambDtwReg);

	//_lamb2MuDtwReg holds mu. _lambDtwReg holds density still, but will be zeroed.
	_staggerX->adjoint(0, _lambDtwReg, _lamb2MuDtwReg); //_lambDtwReg now holds x staggered _mu
	_staggerZ->adjoint(0, _muxzDtwReg, _lambDtwReg); //_muxzDtwReg now holds x and z staggered mu

	//_lamb2MuDtwReg holds mu. _lambDtwReg holds density still, but will be zeroed.
	_staggerX->adjoint(0, _lambDtwReg, _lamb2MuDtwReg); //_lambDtwReg now holds x staggered _mu
	_staggerY->adjoint(0, _muxyDtwReg, _lambDtwReg); //_muxyDtwReg now holds x and y staggered mu

	//_lamb2MuDtwReg holds mu. _lambDtwReg holds density still, but will be zeroed.
	_staggerY->adjoint(0, _lambDtwReg, _lamb2MuDtwReg); //_lambDtwReg now holds y staggered _mu
	_staggerZ->adjoint(0, _muyzDtwReg, _lambDtwReg); //_muyzDtwReg now holds y and z staggered mu

	//scaling factor for shifted
	#pragma omp for collapse(3)
	for (int iy = 0; iy < _ny; iy++){
		for (int ix = 0; ix < _nx; ix++){
			for (int iz = 0; iz < _nz; iz++) {
				(*_rhoxDtwReg->_mat)[iy][ix][iz] = 2.0*_dtw / (*_rhoxDtwReg->_mat)[iy][ix][iz];
				(*_rhoyDtwReg->_mat)[iy][ix][iz] = 2.0*_dtw / (*_rhoyDtwReg->_mat)[iy][ix][iz];
				(*_rhozDtwReg->_mat)[iy][ix][iz] = 2.0*_dtw / (*_rhozDtwReg->_mat)[iy][ix][iz];
				(*_lambDtwReg->_mat)[iy][ix][iz] = 2.0*_dtw * (*_elasticParam->_mat)[1][iy][ix][iz];
				(*_lamb2MuDtwReg->_mat)[iy][ix][iz] = 2.0*_dtw * ((*_elasticParam->_mat)[1][iy][ix][iz] + 2 * (*_elasticParam->_mat)[2][iy][ix][iz]);
				(*_muxzDtwReg->_mat)[iy][ix][iz] = 2.0*_dtw * (*_muxzDtwReg->_mat)[iy][ix][iz];
				(*_muxyDtwReg->_mat)[iy][ix][iz] = 2.0*_dtw * (*_muxyDtwReg->_mat)[iy][ix][iz];
				(*_muyzDtwReg->_mat)[iy][ix][iz] = 2.0*_dtw * (*_muyzDtwReg->_mat)[iy][ix][iz];
			}
		}
	}

	// //get pointer to double array holding values. This is later passed to the device.
	_rhoxDtw = _rhoxDtwReg->getVals();
	_rhoyDtw = _rhoyDtwReg->getVals();
	_rhozDtw = _rhozDtwReg->getVals();
	_lambDtw = _lambDtwReg->getVals();
	_lamb2MuDtw = _lamb2MuDtwReg->getVals();
	_muxzDtw = _muxzDtwReg->getVals();
	_muxyDtw = _muxyDtwReg->getVals();
	_muyzDtw = _muyzDtwReg->getVals();

}

void fdParamElastic_3D::getInfo_3D(){

		std::cerr << " " << std::endl;
		std::cerr << "*******************************************************************" << std::endl;
		std::cerr << "************************ FD PARAMETERS INFO ***********************" << std::endl;
		std::cerr << "*******************************************************************" << std::endl;
		std::cerr << " " << std::endl;

		// Coarse time sampling
		std::cerr << "------------------------ Coarse time sampling ---------------------" << std::endl;
		std::cerr << std::fixed;
		std::cerr << std::setprecision(3);
		std::cerr << "nts = " << _nts << " [samples], dts = " << _dts << " [s], ots = " << _ots << " [s]" << std::endl;
		std::cerr << std::setprecision(1);
		std::cerr << "Nyquist frequency = " << 1.0/(2.0*_dts) << " [Hz]" << std::endl;
		std::cerr << "Maximum frequency from seismic source = " << _fMax << " [Hz]" << std::endl;
		std::cerr << std::setprecision(6);
		std::cerr << "Total recording time = " << (_nts-1) * _dts << " [s]" << std::endl;
		std::cerr << "Subsampling = " << _sub << std::endl;
		std::cerr << " " << std::endl;

		// Coarse time sampling
		std::cerr << "------------------------ Fine time sampling -----------------------" << std::endl;
		std::cerr << "ntw = " << _ntw << " [samples], dtw = " << _dtw << " [s], otw = " << _otw << " [s]" << std::endl;
		std::cerr << "derivative sampling (dtw*2) = " << _dtw*2 << std::endl;
		std::cerr << " " << std::endl;

		// Vertical spatial sampling
		std::cerr << "-------------------- Vertical spatial sampling --------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "nz = " << _nz << " [samples], dz = " << _dz << "[m], oz = " << _oz+(_fat+_zPadMinus)*_dz << " [m]" << std::endl;
		std::cerr << "Model depth = " << (_nz-2*_fat-_zPadMinus-_zPadPlus-1)*_dz << " [m]" << std::endl;
		std::cerr << "Top padding = " << _zPadMinus << " [samples], bottom padding = " << _zPadPlus << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// Horizontal x spatial sampling
		std::cerr << "-------------------- Horizontal x spatial sampling ------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "nx = " << _nx << " [samples], dx = " << _dx << " [m], ox = " << _ox+(_fat+_xPadMinus)*_dx << " [m]" << std::endl;
		std::cerr << "Model width = " << (_nx-2*_fat-_xPadMinus-_xPadPlus-1)*_dx << " [m]" << std::endl;
		std::cerr << "Left padding = " << _xPadMinus << " [samples], right padding = " << _xPadPlus << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// Horizontal x spatial sampling
		std::cerr << "-------------------- Horizontal y spatial sampling ------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "ny = " << _ny << " [samples], dy = " << _dy << " [m], ox = " << _oy+(_fat+_yPad)*_dx << " [m]" << std::endl;
		std::cerr << "Model width = " << (_ny-2*_fat-2*_yPad-1)*_dy << " [m]" << std::endl;
		std::cerr << "Left padding = " << _yPad << " [samples], right padding = " << _yPad << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// GPU FD parameters
		std::cerr << "---------------------- GPU kernels parameters ---------------------" << std::endl;
		std::cerr << "Block size in z-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cerr << "Block size in x-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cerr << "Block size in y-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cerr << "Halo size for staggered 8th-order derivative [FAT] = " << _fat << " [samples]" << std::endl;
		std::cerr << " " << std::endl;

		// Stability and dispersion
		std::cerr << "---------------------- Stability and dispersion -------------------" << std::endl;
		std::cerr << std::setprecision(2);
		std::cerr << "Courant number = " << _Courant << " [-]" << std::endl;
		std::cerr << "Dispersion ratio = " << _dispersionRatio << " [points/min wavelength]" << std::endl;
		std::cerr << "Minimum velocity value (of either vp or vs) = " << _minVpVs << " [m/s]" << std::endl;
		std::cerr << "Maximum velocity value (of either vp or vs) = " << _maxVpVs << " [m/s]" << std::endl;
		std::cerr << std::setprecision(1);
		std::cerr << "Maximum frequency without dispersion = " << _minVpVs/(3.0*std::max(_dz, std::max(_dx, _dy))) << " [Hz]" << std::endl;
		std::cerr << " " << std::endl;


		// Free Surface Condition
		std::cerr << "----------------------- Surface Condition --------------------" << std::endl;
		std::cerr << "Chosen surface condition parameter: ";
		if(_freeSurface==0) std::cerr << "(0) no free surface condition" << '\n';
		else if(_freeSurface==1 ) std::cerr << "(1) free surface condition from Robertsson (1998) chosen." << '\n';
		else{
			std::cerr << "ERROR NO IMPROPER FREE SURFACE PARAMETER PROVIDED" << '\n';
			throw std::runtime_error("");;
		}

		std::cerr << "\n----------------------- Source Interp Info -----------------------" << std::endl;
		std::cerr << "Chosen source interpolation method: " << _par->getString("sourceInterpMethod","linear") << std::endl;
		std::cerr << "Chosen number of filter on one side of device: " << _par->getInt("sourceInterpNumFilters",1) << std::endl;
		std::cerr << " " << std::endl;
		std::cerr << "\n*******************************************************************" << std::endl;
		std::cerr << " " << std::endl;
		std::cerr << std::scientific; // Reset to scientific formatting notation
		std::cerr << std::setprecision(6); // Reset the default formatting precision
}

bool fdParamElastic_3D::checkFdStability_3D(double CourantMax){
	_minDzDxDy = std::min(_dz, std::min(_dx, _dy));
	_Courant = _maxVpVs * _dtw * 2.0 / _minDzDxDy;
	if (_Courant > CourantMax){
		std::cerr << "**** ERROR [fdParamElastic_3D]: Courant is too big: " << _Courant << " ****" << std::endl;
		std::cerr << "Max velocity value: " << _maxVpVs << " [m/s]" << std::endl;
		std::cerr << "Dtw: " << _dtw << " [s]" << std::endl;
		std::cerr << "Min (dz, dx): " << _minDzDxDy << " [m]" << std::endl;
		return false;
	}
	return true;
}

bool fdParamElastic_3D::checkFdDispersion_3D(double dispersionRatioMin){
	_maxDzDxDy = std::max(_dz, std::max(_dx, _dy));
	_dispersionRatio = _minVpVs / (_fMax*_maxDzDxDy);

	if (_dispersionRatio < dispersionRatioMin){
		std::cerr << "**** ERROR [fdParamElastic_3D]: Dispersion is too small: " << _dispersionRatio <<  " > " << dispersionRatioMin << " ****" << std::endl;
		std::cerr << "Min velocity value = " << _minVpVs << " [m/s]" << std::endl;
		std::cerr << "Max (dz, dx) = " << _maxDzDxDy << " [m]" << std::endl;
		std::cerr << "Max frequency = " << _fMax << " [Hz]" << std::endl;
		return false;
	}
	return true;
}

bool fdParamElastic_3D::checkModelSize_3D(){

	if ( (_nz-2*_fat) % _blockSize != 0 ) {
		std::cout << "**** ERROR [fdParamElastic_3D]: nz-2xFAT not a multiple of block size ****" << std::endl;
		return false;
	}
	if ((_nx-2*_fat) % _blockSize != 0) {
		std::cout << "**** ERROR [fdParamElastic_3D]: nx-2xFAT not a multiple of block size ****" << std::endl;
		return false;
	}

	return true;
}

bool fdParamElastic_3D::checkParfileConsistencyTime_3D(const std::shared_ptr<double3DReg> seismicTraces, int timeAxisIndex, std::string fileToCheck) const {
	if (_nts != seismicTraces->getHyper()->getAxis(timeAxisIndex).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: nts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dts - seismicTraces->getHyper()->getAxis(timeAxisIndex).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: dts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ots - seismicTraces->getHyper()->getAxis(timeAxisIndex).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: ots not consistent with parfile ****" << std::endl; return false;}
	return true;
}

// check consistency with elastic model
bool fdParamElastic_3D::checkParfileConsistencySpace_3D(const std::shared_ptr<double4DReg> model, std::string fileToCheck) const {

	// Vertical axis
	if (_nz != model->getHyper()->getAxis(1).n) {std::cout << "**** ["<< fileToCheck << "] ERROR [fdParamElastic_3D]: nz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dz - model->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: dz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oz - model->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: oz not consistent with parfile ****" << std::endl; return false;}

	// Horizontal x-axis
	if (_nx != model->getHyper()->getAxis(2).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: nx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dx - model->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: dx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ox - model->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: ox not consistent with parfile ****" << std::endl; return false;}

	// Horizontal y-axis
	if (_ny != model->getHyper()->getAxis(3).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: ny not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dy - model->getHyper()->getAxis(3).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: dy not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oy - model->getHyper()->getAxis(3).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParamElastic_3D]: oy not consistent with parfile ****" << std::endl; return false;}

	return true;
}

fdParamElastic_3D::~fdParamElastic_3D(){
  _rhoxDtw = NULL;
	_rhoyDtw = NULL;
  _rhozDtw = NULL;
  _lamb2MuDtw = NULL;
  _lambDtw = NULL;
  _muxzDtw = NULL;
	_muxyDtw = NULL;
	_muyzDtw = NULL;
}
