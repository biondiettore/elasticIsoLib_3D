#include <float1DReg.h>
#include <float2DReg.h>
#include <iostream>
#include "spaceInterpGpu_3D.h"
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/special_functions/cos_pi.hpp>
#include <boost/math/distributions/normal.hpp>
#include <math.h>
#include <vector>


// Constructor #1 -- Only for irregular geometry
spaceInterpGpu_3D::spaceInterpGpu_3D(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<float1DReg> yCoord, const std::shared_ptr<SEP::hypercube> elasticParamHypercube, int &nt, std::shared_ptr<paramObj> par, int dipole, float zDipoleShift, float xDipoleShift, float yDipoleShift, std::string interpMethod, int hFilter1d){

	// Get domain dimensions
	_par = par;
	_oz = elasticParamHypercube->getAxis(1).o;
	_dz = elasticParamHypercube->getAxis(1).d;
	_nz = elasticParamHypercube->getAxis(1).n;
	_ox = elasticParamHypercube->getAxis(2).o;
	_dx = elasticParamHypercube->getAxis(2).d;
	_nx = elasticParamHypercube->getAxis(2).n;
	_oy = elasticParamHypercube->getAxis(3).o;
	_dy = elasticParamHypercube->getAxis(3).d;
	_ny = elasticParamHypercube->getAxis(3).n;
	_fat = _par->getInt("fat");
	_zPadMinus = _par->getInt("zPadMinus");
	_zPadPlus = _par->getInt("zPadPlus");
	_xPadMinus = _par->getInt("xPadMinus");
	_xPadPlus = _par->getInt("xPadPlus");
	_yPad = _par->getInt("yPad");
	_errorTolerance = par->getFloat("errorTolerance", 1e-6);

	// Get positions of aquisition devices + other parameters
	_zCoord = zCoord;
	_xCoord = xCoord;
	_yCoord = yCoord;
	_dipole = dipole; // Default value is 0
	_zDipoleShift = zDipoleShift; // Default value is 0
	_xDipoleShift = xDipoleShift; // Default value is 0
	_yDipoleShift = yDipoleShift; // Default value is 0
	checkOutOfBounds(_zCoord, _xCoord, _yCoord); // Check that none of the acquisition devices are out of bounds
	_hFilter1d = hFilter1d; // Half-length of the filter for each dimension. For sinc, the filter in each direction z, x, y has the same length
	_interpMethod = interpMethod; // Default is linear
	_nDeviceIrreg = _zCoord->getHyper()->getAxis(1).n; // Nb of devices on irregular grid
	_nt = nt;

	// Check that for the linear interpolation case, the filter half-length is 1
	if (_interpMethod == "linear" && _hFilter1d != 1){
		std::cout << "**** ERROR [spaceInterpGpu_3D]: Half-length of interpolation filter should be set to 1 for linear interpolation ****" << std::endl;
		throw std::runtime_error("");
	}

	// Compute the total number of points on the grid for each axis that will be involved in the interpolation
	_nFilter1d = 2*_hFilter1d; // Filter length for each dimension. For sinc, we have "_hFilter" number of points on each side
	_nFilter3d = _nFilter1d*_nFilter1d*_nFilter1d; // Total number of points involved in the interpolation in 3D

	// Dipole case: we use twice as many points
	if (_dipole == 1){
		_nFilter3dDipole = _nFilter3d;
		_nFilter3d = _nFilter3d*2;
	}

	_gridPointIndex = new long long[_nFilter3d*_nDeviceIrreg]; // 1d-array containing the index of all the grid points (on the regular grid) that will be used in the interpolation. The indices are not unique
	_weight = new float[_nFilter3d*_nDeviceIrreg]; // Weights corresponding to the points on the grid stored in _gridPointIndex

	// Compute list of all grid points used in the interpolation
	if (_interpMethod == "linear"){
		calcLinearWeights();
	} else if (_interpMethod == "sinc"){
		calcSincWeights();
	} else {
		std::cerr << "**** ERROR [spaceInterpGpu_3D]: Space interpolation method not supported ****" << std::endl;
	}

	// Convert the list -> Create a new list with unique indices of the regular grid points involved in the interpolation
	convertIrregToReg();

}

// Update spatial interpolation parameters for Ginsu
void spaceInterpGpu_3D::setspaceInterpGpu_3D(const std::shared_ptr<SEP::hypercube> elasticParamHypercubeGinsu, const int xPadMinusGinsu, const int xPadPlusGinsu){

	// Get domain dimensions
	_oz = elasticParamHypercubeGinsu->getAxis(1).o;
	_dz = elasticParamHypercubeGinsu->getAxis(1).d;
	_nz = elasticParamHypercubeGinsu->getAxis(1).n;
	_ox = elasticParamHypercubeGinsu->getAxis(2).o;
	_dx = elasticParamHypercubeGinsu->getAxis(2).d;
	_nx = elasticParamHypercubeGinsu->getAxis(2).n;
	_oy = elasticParamHypercubeGinsu->getAxis(3).o;
	_dy = elasticParamHypercubeGinsu->getAxis(3).d;
	_ny = elasticParamHypercubeGinsu->getAxis(3).n;
	_xPadMinus = xPadMinusGinsu;
	_xPadPlus = xPadPlusGinsu;

	checkOutOfBounds(_zCoord, _xCoord, _yCoord); // Check that none of the acquisition devices are out of bounds

	// Compute list of all grid points used in the interpolation
	if (_interpMethod == "linear"){
		calcLinearWeights();
	} else if (_interpMethod == "sinc"){
		calcSincWeights();
	} else {
		std::cerr << "**** ERROR [spaceInterpGpu_3D]: Space interpolation method not supported ****" << std::endl;
	}

	// Convert the list -> Create a new list with unique indices of the regular grid points involved in the interpolation
	convertIrregToReg();

}

void spaceInterpGpu_3D::convertIrregToReg() {

	/* (1) Create map where:
		- Key = excited grid point index (points are unique)
		- Value = signal trace number
		(2) Create a vector containing the indices of the excited grid points
	*/

	_nDeviceReg = 0; // Initialize the number of regular devices to zero
	_gridPointIndexUnique.clear(); // Initialize to empty vector
	_indexMap.clear(); // Clear the map to reinitialize the unique device positions array

	for (long long iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over gridPointIndex array
		for (long long iFilter = 0; iFilter < _nFilter3d; iFilter++){
			long long i1 = iDevice * _nFilter3d + iFilter;

			// If the grid point is not already in the list
			if (_indexMap.count(_gridPointIndex[i1]) == 0) {
				// std::cout << "Adding" << std::endl;
				_nDeviceReg++; // Increment the number of (unique) grid points excited by the signal
				_indexMap[_gridPointIndex[i1]] = _nDeviceReg - 1; // Add the pair to the map
				_gridPointIndexUnique.push_back(_gridPointIndex[i1]); // Append vector containing all unique grid point index
			}
		}
	}
}

void spaceInterpGpu_3D::forward(const bool add, const std::shared_ptr<float2DReg> signalReg, std::shared_ptr<float2DReg> signalIrreg) const {

	/* FORWARD: Go from REGULAR grid -> IRREGULAR grid */
	if (!add) signalIrreg->scale(0.0);

	std::shared_ptr<float2D> d = signalIrreg->_mat;
	std::shared_ptr<float2D> m = signalReg->_mat;

	for (long long iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over device
		for (long long iFilter = 0; iFilter < _nFilter3d; iFilter++){ // Loop over neighboring points on regular grid
			long long i1 = iDevice * _nFilter3d + iFilter;
			long long i2 = _indexMap.find(_gridPointIndex[i1])->second;
			for (long long it = 0; it < _nt; it++){
				(*d)[iDevice][it] += _weight[i1] * (*m)[i2][it];
			}
		}
	}
}

void spaceInterpGpu_3D::adjoint(const bool add, std::shared_ptr<float2DReg> signalReg, const std::shared_ptr<float2DReg> signalIrreg) const {

	/* ADJOINT: Go from IRREGULAR grid -> REGULAR grid */
	if (!add) signalReg->scale(0.0);
	std::shared_ptr<float2D> d = signalIrreg->_mat;
	std::shared_ptr<float2D> m = signalReg->_mat;

	for (long long iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over acquisition devices' positions
		for (long long iFilter = 0; iFilter < _nFilter3d; iFilter++){ // Loop over neighboring points on regular grid
			long long i1 = iDevice * _nFilter3d + iFilter; // Grid point index
			long long i2 = _indexMap.find(_gridPointIndex[i1])->second; // Get trace number for signalReg
			for (int it = 0; it < _nt; it++){
				(*m)[i2][it] += _weight[i1] * (*d)[iDevice][it];
			}
		}
	}
}

void spaceInterpGpu_3D::checkOutOfBounds(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<float1DReg> yCoord){

	long long nDevice = zCoord->getHyper()->getAxis(1).n;
	_nzSmall = _nz - 2 * _fat - _zPadMinus - _zPadPlus;
	_nxSmall = _nx - 2 * _fat - _xPadMinus - _xPadPlus;
	_nySmall = _ny - 2 * _fat - 2 * _yPad;

	float zMin = _oz + (_fat + _zPadMinus) * _dz;
	float xMin = _ox + (_fat + _xPadMinus) * _dx;
	float yMin = _oy + (_fat + _yPad) * _dy;

	float zMax = zMin + (_nzSmall - 1) * _dz;
	float xMax = xMin + (_nxSmall - 1) * _dx;
  float yMax = yMin + (_nySmall - 1) * _dy;

	for (long long iDevice = 0; iDevice < nDevice; iDevice++){
		if ( (*zCoord->_mat)[iDevice] - zMax > _errorTolerance || (*zCoord->_mat)[iDevice] - zMin < -_errorTolerance || (*xCoord->_mat)[iDevice] - xMax > _errorTolerance || (*xCoord->_mat)[iDevice] - xMin < -_errorTolerance || (*yCoord->_mat)[iDevice] - yMax > _errorTolerance || (*yCoord->_mat)[iDevice] - yMin < -_errorTolerance ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of the acquisition devices is out of bounds ****" << std::endl;
			std::cout << "iDevice = " << iDevice << std::endl;
			std::cout << "zMin = " << zMin << std::endl;
			std::cout << "zMax = " << zMax << std::endl;
			std::cout << "xMin = " << xMin << std::endl;
			std::cout << "xMax = " << xMax << std::endl;
			std::cout << "yMin = " << yMin << std::endl;
			std::cout << "yMax = " << yMax << std::endl;
			std::cout << "zCoord = " << (*zCoord->_mat)[iDevice] << std::endl;
			std::cout << "xCoord = " << (*xCoord->_mat)[iDevice] << std::endl;
			std::cout << "yCoord = " << (*yCoord->_mat)[iDevice] << std::endl;
			std::cout << "(*zCoord->_mat)[iDevice] - zMin = " << (*zCoord->_mat)[iDevice] - zMin << std::endl;
			std::cout << "-_errorTolerance = " << -_errorTolerance << std::endl;
			std::cout << "-------------------------------" << std::endl;
			throw std::runtime_error("");
		}
	}
}

// Compute weights for linear interpolation
void spaceInterpGpu_3D::calcLinearWeights(){

	for (long long iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {

		// Find the 8 neighboring points for all devices and compute the weights for the spatial interpolation
		long long i1 = iDevice * _nFilter3d;
		float wz = ( (*_zCoord->_mat)[iDevice] - _oz ) / _dz;
		float wx = ( (*_xCoord->_mat)[iDevice] - _ox ) / _dx;
    float wy = ( (*_yCoord->_mat)[iDevice] - _oy ) / _dy;
		long long zReg = wz; // z-coordinate on regular grid
		wz = wz - zReg;
		wz = 1.0 - wz;
		long long xReg = wx; // x-coordinate on regular grid
		wx = wx - xReg;
		wx = 1.0 - wx;
    long long yReg = wy; // y-coordinate on regular grid
    wy = wy - yReg;
    wy = 1.0 - wy;

		// Check for the y-axis
		if ( (yReg < _fat + _yPad) || (yReg + 1 >= _ny - _fat - _yPad) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the y-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}
		// Check for the x-axis
		if ( (xReg < _fat + _xPadMinus) || (xReg + 1 >= _nx - _fat - _xPadPlus) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the x-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}
		// Check for the z-axis
		if ( (zReg < _fat + _zPadMinus) || (zReg + 1 >= _nz - _fat - _zPadPlus) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the z-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}

		// Top front left
		_gridPointIndex[i1] = yReg * _nz * _nx + xReg * _nz + zReg; // Index of this point for a 1D array representation
		_weight[i1] = wz * wx * wy;

		// Bottom front left
		_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
		_weight[i1+1] = (1.0 - wz) * wx * wy;

		// Top front right
		_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
		_weight[i1+2] = wz * (1.0 - wx) * wy;

		// Bottom front right
		_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
		_weight[i1+3] = (1.0 - wz) * (1.0 - wx) * wy;

    // Top back left
    _gridPointIndex[i1+4] = _gridPointIndex[i1] + _nz *_nx;
    _weight[i1+4] = wz * wx * (1.0 - wy);

    // Bottom back left
    _gridPointIndex[i1+5] = _gridPointIndex[i1] + _nz *_nx + 1;
    _weight[i1+5] = (1.0 - wz) * wx * (1.0 - wy);

    // Top back right
    _gridPointIndex[i1+6] = _gridPointIndex[i1] + _nz *_nx + _nz;
    _weight[i1+6] = wz * (1.0 - wx) * (1.0 - wy);

    // Bottom back right
    _gridPointIndex[i1+7] = _gridPointIndex[i1] + _nz *_nx + _nz + 1;
    _weight[i1+7] = (1.0 - wz) * (1.0 - wx) * (1.0 - wy);

		// Case where we use a dipole or the seismic device
		if (_dipole == 1){

			// Find the 8 neighboring points for all devices dipole points and compute the weights for the spatial interpolation
			float wzDipole = ( (*_zCoord->_mat)[iDevice] + _zDipoleShift - _oz ) / _dz;
			float wxDipole = ( (*_xCoord->_mat)[iDevice] + _xDipoleShift - _ox ) / _dx;
      float wyDipole = ( (*_yCoord->_mat)[iDevice] + _yDipoleShift - _oy ) / _dz;
			long long zRegDipole = wzDipole; // z-coordinate on regular grid
			wzDipole = wzDipole - zRegDipole;
			wzDipole = 1.0 - wzDipole;
			long long xRegDipole = wxDipole; // x-coordinate on regular grid
			wxDipole = wxDipole - xRegDipole;
			wxDipole = 1.0 - wxDipole;
      long long yRegDipole = wyDipole; // y-coordinate on regular grid
			wyDipole = wyDipole - yRegDipole;
			wyDipole = 1.0 - wyDipole;

			// Check for the y-axis
			if ( (yRegDipole < _fat + _yPad) || (yRegDipole + 1 >= _ny - _fat - _yPad) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the y-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}
			// Check for the x-axis
			if ( (xRegDipole < _fat + _xPadMinus) || (xRegDipole + 1 >= _nx - _fat - _xPadPlus) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the x-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}
			// Check for the z-axis
			if ( (zRegDipole < _fat + _zPadMinus) || (zRegDipole + 1 >= _nz - _fat - _zPadPlus) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the z-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}

			// Top front left (dipole point)
			_gridPointIndex[i1+8] = yRegDipole * _nz * _nx + xRegDipole * _nz + zRegDipole; // Index of this point for a 1D array representation
			_weight[i1+8] = (-1.0) * wzDipole * wxDipole * wyDipole;

			// Bottom front left (dipole point)
			_gridPointIndex[i1+9] = _gridPointIndex[i1+8] + 1;
			_weight[i1+9] = (-1.0) * (1.0 - wzDipole) * wxDipole * wyDipole;

			// Top front right (dipole point)
			_gridPointIndex[i1+10] = _gridPointIndex[i1+8] + _nz;
			_weight[i1+10] = (-1.0) * wzDipole * (1.0 - wxDipole) * wyDipole;

			// Bottom right (dipole point)
			_gridPointIndex[i1+11] = _gridPointIndex[i1+8] + _nz + 1;
			_weight[i1+11] = (-1.0) * (1.0 - wzDipole) * (1.0 - wxDipole) * wyDipole;

      // Top back left (dipole point)
      _gridPointIndex[i1+12] = _gridPointIndex[i1+8] + _nz * _nx;
			_weight[i1+12] = (-1.0) * wzDipole * wxDipole * (1.0 - wyDipole);

			// Bottom back left (dipole point)
			_gridPointIndex[i1+13] = _gridPointIndex[i1+8] + _nz * _nx + 1;
			_weight[i1+13] = (-1.0) * (1.0 - wzDipole) * wxDipole * (1.0 - wyDipole);

			// Top back right (dipole point)
			_gridPointIndex[i1+14] = _gridPointIndex[i1+8] + _nz * _nx + _nz;
			_weight[i1+14] = (-1.0) * wzDipole * (1.0 - wxDipole) * (1.0 - wyDipole);

			// Bottom back right (dipole point)
			_gridPointIndex[i1+15] = _gridPointIndex[i1+8] + _nz * _nx + _nz + 1;
			_weight[i1+15] = (-1.0) * (1.0 - wzDipole) * (1.0 - wxDipole) * (1.0 - wyDipole);

		}
	}
}

// Compute weights for sinc interpolation
void spaceInterpGpu_3D::calcSincWeights(){

	for (long long iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {

		long long i1 = iDevice * _nFilter3d;

		// Compute position of the acquisition device [km]
		float zIrreg = (*_zCoord->_mat)[iDevice];
		float xIrreg = (*_xCoord->_mat)[iDevice];
		float yIrreg = (*_yCoord->_mat)[iDevice];
		float zReg = (zIrreg - _oz) / _dz;
		float xReg = (xIrreg - _ox) / _dx;
		float yReg = (yIrreg - _oy) / _dy;

		// Index of top left grid point closest to the acquisition device
		long long zRegInt = zReg;
		long long xRegInt = xReg;
		long long yRegInt = yReg;

		// Check that none of the points used in the interpolation are out of bounds
		if ( (yRegInt-_hFilter1d+1 < 0) || (yRegInt+_hFilter1d+1 < _ny) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the y-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}
		if ( (xRegInt-_hFilter1d+1 < 0) || (xRegInt+_hFilter1d+1 < _nx) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the x-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}
		if ( (zRegInt-_hFilter1d+1 < 0) || (zRegInt+_hFilter1d+1 < _nz) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the z-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}

		// Loop over grid points involved in the interpolation
		for (long long iy = 0; iy < _nFilter1d; iy++){
			for (long long ix = 0; ix < _nFilter1d; ix++){
				for (long long iz = 0; iz < _nFilter1d; iz++){

					// Compute grid point position
					float yCur = (yRegInt+iy-_hFilter1d+1) * _dy + _oy;
					float xCur = (xRegInt+ix-_hFilter1d+1) * _dx + _ox;
					float zCur = (zRegInt+iz-_hFilter1d+1) * _dz + _oz;

					// Compute argument for the sinc function
					float wy = (yIrreg-yCur)/_dy;
					float wx = (xIrreg-xCur)/_dx;
					float wz = (zIrreg-zCur)/_dz;

					// Compute global index of grid point used in the interpolation (on the main FD grid)
					long long iPointInterp = _nz*_nx*(yRegInt+iy-_hFilter1d+1) + _nz*(xRegInt+ix-_hFilter1d+1) + (zRegInt+iz-_hFilter1d+1);

					// Compute index in the array that contains the positions of the grid points involved in the interpolation
					long long iGridPointIndex = i1+iy*_nFilter1d*_nFilter1d+ix*_nFilter1d+iz;

					// Compute global index
					_gridPointIndex[iGridPointIndex] = iPointInterp;

					// Compute weight associated with that point
					_weight[iGridPointIndex] = boost::math::sinc_pi(M_PI*wz)*boost::math::sinc_pi(M_PI*wx)*boost::math::sinc_pi(M_PI*wy);

				}
			}
		}

		if (_dipole == 1){

			long long i1Dipole = i1 + _nFilter3dDipole;

			// Compute position of the acquisition device [km] for the second pole
			float zIrregDipole = (*_zCoord->_mat)[iDevice]+_zDipoleShift;
			float xIrregDipole = (*_xCoord->_mat)[iDevice]+_xDipoleShift;
			float yIrregDipole = (*_yCoord->_mat)[iDevice]+_yDipoleShift;
			float zRegDipole = (zIrregDipole - _oz) / _dz;
			float xRegDipole = (xIrregDipole - _ox) / _dx;
			float yRegDipole = (yIrregDipole - _oy) / _dy;

			// Index of top left grid point closest to the acquisition device (corner of the voxel where the device lies that has the smallest index)
			long long zRegDipoleInt = zRegDipole;
			long long xRegDipoleInt = xRegDipole;
			long long yRegDipoleInt = yRegDipole;

			// Check that none of the points used in the interpolation are out of bounds
			if ( (yRegDipoleInt-_hFilter1d+1 < 0) || (yRegDipoleInt+_hFilter1d+1 < _ny) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the y-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}
			if ( (xRegDipoleInt-_hFilter1d+1 < 0) || (xRegDipoleInt+_hFilter1d+1 < _nx) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the x-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}
			if ( (zRegDipoleInt-_hFilter1d+1 < 0) || (zRegDipoleInt+_hFilter1d+1 < _nz) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the z-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}

			// Loop over grid points involved in the interpolation
			for (long long iy = 0; iy < _nFilter1d; iy++){
				for (long long ix = 0; ix < _nFilter1d; ix++){
					for (long long iz = 0; iz < _nFilter1d; iz++){

						// Compute grid point position
						float yCurDipole = (yRegDipoleInt+iy-_hFilter1d+1) * _dy + _oy;
						float xCurDipole = (xRegDipoleInt+ix-_hFilter1d+1) * _dx + _ox;
						float zCurDipole = (zRegDipoleInt+iz-_hFilter1d+1) * _dz + _oz;

						// Compute argument for the sinc function
						float wyDipole = (yIrregDipole-yCurDipole)/_dy;
						float wxDipole = (xIrregDipole-xCurDipole)/_dx;
						float wzDipole = (zIrregDipole-zCurDipole)/_dz;

						// Compute global index of grid point used in the interpolation (on the main FD grid) for the other pole
						long long iPointInterpDipole = _nz*_nx*(yRegDipoleInt+iy-_hFilter1d+1) + _nz*(xRegDipoleInt+ix-_hFilter1d+1) + (zRegDipoleInt+iz-_hFilter1d+1);

						// Compute index in the array that contains the positions of the grid points (non-unique) involved in the interpolation
						long long iGridPointIndexDipole = i1Dipole+iy*_nFilter1d*_nFilter1d+ix*_nFilter1d+iz;

						// Compute global index
						_gridPointIndex[iGridPointIndexDipole] = iPointInterpDipole;

						// Compute weight associated with that point
						_weight[iGridPointIndexDipole] = boost::math::sinc_pi(M_PI*wzDipole)*boost::math::sinc_pi(M_PI*wxDipole)*boost::math::sinc_pi(M_PI*wyDipole);

					}
				}
			}

		}

	}
}

void spaceInterpGpu_3D::printRegPosUnique(){
	std::cout << "Size unique = " << getSizePosUnique() << std::endl;
	std::cout << "getNDeviceIrreg = " << getNDeviceIrreg() << std::endl;
	std::cout << "getSizePosUnique = " << getSizePosUnique() << std::endl;
	for (long long iDevice=0; iDevice<getSizePosUnique(); iDevice++){
		std::cout << "Position for device #" << iDevice << " = " << _gridPointIndexUnique[iDevice] << std::endl;
	}
}
