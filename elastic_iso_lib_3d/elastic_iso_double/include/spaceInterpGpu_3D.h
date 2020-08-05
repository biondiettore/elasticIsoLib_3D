#ifndef DEVICE_ELASTIC_GPU_3D_H
#define DEVICE_ELASTIC_GPU_3D_H 1

#include "ioModes.h"
#include "double1DReg.h"
#include "double2DReg.h"
#include "operator.h"
#include <vector>

using namespace SEP;

class spaceInterpGpu_3D : public Operator<SEP::double2DReg, SEP::double2DReg> {

	private:

		/* Spatial interpolation */
		std::shared_ptr<SEP::hypercube> _elasticParamHypercube;
		std::shared_ptr<double1DReg> _zCoord, _xCoord, _yCoord;
		std::vector<long long> _gridPointIndexUnique; // Array containing all the positions of the excited grid points - each grid point is unique
		std::map<int, int> _indexMap;
		std::map<int, int>::iterator _iteratorIndexMap;
		double _errorTolerance;
		double *_weight;
		double _oz, _dz, _ox, _dx, _oy, _dy;
		long long *_gridPointIndex;
		long long _nDeviceIrreg, _nDeviceReg;
		int _nt, _nz, _nx, _ny;
		int _nzSmall, _nxSmall, _nySmall;
		int _fat, _zPadMinus, _zPadPlus, _xPadMinus, _xPadPlus, _yPad;
		int _dipole, _zDipoleShift, _xDipoleShift, _yDipoleShift;
		int _hFilter1d, _nFilter1d, _nFilter3d, _nFilter3dDipole;
		std::string _interpMethod;
		std::shared_ptr<paramObj> _par;
		// Ginsu parameters
		std::shared_ptr<SEP::hypercube> _velHyperGinsu;

	public:

		// Constructor #1: Provide positions of acquisition devices in km
		// Acquisition devices do not need to be placed on grid points
		spaceInterpGpu_3D(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<double1DReg> yCoord, const std::shared_ptr<SEP::hypercube> elasticParamHypercube, int &nt, std::shared_ptr<paramObj> par, int dipole=0, double zDipoleShift=0, double xDipoleShift=0, double yDipoleShift=0, std::string interpMethod="linear", int hFilter1d=1);

		// Mutator: updates domain parameters for Ginsu
		void setspaceInterpGpu_3D(const std::shared_ptr<SEP::hypercube> elasticParamHypercubeGinsu, const int xPadMinusGinsu, const int xPadPlusGinsu);

		// FWD / ADJ
		void forward(const bool add, const std::shared_ptr<double2DReg> signalReg, std::shared_ptr<double2DReg> signalIrreg) const;
		void adjoint(const bool add, std::shared_ptr<double2DReg> signalReg, const std::shared_ptr<double2DReg> signalIrreg) const;

		// Destructor
		~spaceInterpGpu_3D(){};

		// Other functions
		void checkOutOfBounds(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<double1DReg> yCoord); // For constructor #1
		void convertIrregToReg();
		void calcLinearWeights();
		void calcSincWeights();

		// Accessors
		long long *getRegPosUnique(){ return _gridPointIndexUnique.data(); }
		long long *getRegPos(){ return _gridPointIndex; }
		int getNt(){ return _nt; }
		long long getNDeviceReg(){ return _nDeviceReg; }
		long long getNDeviceIrreg(){ return _nDeviceIrreg; }
		double * getWeights() { return _weight; }
		int getSizePosUnique(){ return _gridPointIndexUnique.size(); }
		std::shared_ptr<double1DReg> getZCoord() {return _zCoord;}
		std::shared_ptr<double1DReg> getXCoord() {return _xCoord;}
		std::shared_ptr<double1DReg> getYCoord() {return _yCoord;}

		// QC
		void printRegPosUnique();
};

#endif
