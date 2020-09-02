#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyElastic_iso_double_nl_3D
import pyElastic_iso_double_born_3D
import pyOperator as Op
import elasticParamConvertModule_3D as ElaConv_3D
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

from pyElastic_iso_double_nl_3D import spaceInterpGpu_3D

################################################################################
############################ Acquisition geometry ##############################
################################################################################
# Build sources geometry
def buildSourceGeometry_3D(parObject,elasticParam):


	#Dipole parameters
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getFloat("zDipoleShift",0.0)
	yDipoleShift = parObject.getFloat("yDipoleShift",0.0)
	xDipoleShift = parObject.getFloat("xDipoleShift",0.0)
	nts = parObject.getInt("nts")
	nExp = parObject.getInt("nExp",-1)
	info = parObject.getInt("info",0)
	spaceInterpMethod = parObject.getString("spaceInterpMethod","linear")
	sourceGeomFile = parObject.getString("sourceGeomFile")

	# Get total number of shots
	if (nExp==-1):
		raise ValueError("**** ERROR [buildSourceGeometry_3D]: User must provide the total number of shots ****\n")

	# Read parameters for spatial interpolation
	if (spaceInterpMethod == "linear"):
		hFilter1d = 1
	elif (spaceInterpMethod == "sinc"):
		hFilter1d = parObject.getInt("hFilter1d",1)
	else:
		raise ValueError("**** ERROR [buildSourceGeometry_3D]: Spatial interpolation method requested by user is not implemented ****\n")

	# Display information for user
	if (info == 1):
		print("**** [buildSourceGeometry_3D]: User has requested to display information ****\n")
		# Interpolation method
		if (spaceInterpMethod == "sinc"):
			print("**** [buildSourceGeometry_3D]: User has requested a sinc spatial interpolation method for the sources' signals injection/extraction ****\n")
		else:
			print("**** [buildSourceGeometry_3D]: User has requested a linear spatial interpolation method for the sources' signals injection/extraction ****\n")
		# Dipole injection/extraction
		if (dipole == 1):
			print("**** [buildSourceGeometry_3D]: User has requested a dipole source injection/extraction ****\n")
			print("**** [buildSourceGeometry_3D]: Dipole shift in z-direction: %f [km or m]  ****\n" %zDipoleShift)
			print("**** [buildSourceGeometry_3D]: Dipole shift in x-direction: %f [km or m]  ****\n" %xDipoleShift)
			print("**** [buildSourceGeometry_3D]: Dipole shift in y-direction: %f [km or m]  ****\n" %yDipoleShift)

	# Getting axes
	zAxis = elasticParam.getHyper().getAxis(1)
	xAxis = elasticParam.getHyper().getAxis(2)
	yAxis = elasticParam.getHyper().getAxis(3)
	paramAxis = elasticParam.getHyper().getAxis(4)
	# Axis parameters
	nz=zAxis.n
	dz=zAxis.d
	oz=zAxis.o

	nx=xAxis.n
	dx=xAxis.d
	ox=xAxis.o

	ny=yAxis.n
	dy=yAxis.d
	oy=yAxis.o

	zAxisShifted=Hypercube.axis(n=nz,o=oz-0.5*dz,d=dz)
	xAxisShifted=Hypercube.axis(n=nx,o=ox-0.5*dx,d=dx)
	yAxisShifted=Hypercube.axis(n=ny,o=oy-0.5*dy,d=dy)

	centerGridHyper=Hypercube.hypercube(axes=[zAxis,xAxis,yAxis,paramAxis])
	xGridHyper=Hypercube.hypercube(axes=[zAxis,xAxisShifted,yAxis,paramAxis])
	yGridHyper=Hypercube.hypercube(axes=[zAxis,xAxis,yAxisShifted,paramAxis])
	zGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxis,yAxis,paramAxis])
	xzGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxisShifted,yAxis,paramAxis])
	xyGridHyper=Hypercube.hypercube(axes=[zAxis,xAxisShifted,yAxisShifted,paramAxis])
	yzGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxis,yAxisShifted,paramAxis])

	# Create a source vector for centerGrid, x shifted, y shifted, z shifted,
	# xz shifted grid, xy shifted grid, and yz shifted grid
	sourcesVectorCenterGrid=[]
	sourcesVectorXGrid=[]
	sourcesVectorYGrid=[]
	sourcesVectorZGrid=[]
	sourcesVectorXZGrid=[]
	sourcesVectorXYGrid=[]
	sourcesVectorYZGrid=[]

	#check which source injection interp method

	# Read geometry file
	# 3 axes:
	# First (fastest) axis: experiment index
	# Second (slower) axis: simultaneous source points
	# Third (slowest) axis: spatial coordinates (x,y,z)
	sourceGeomVectorNd = genericIO.defaultIO.getVector(sourceGeomFile,ndims=3).getNdArray()
	nExpFile = sourceGeomVectorNd.shape[2]
	nSimSou = sourceGeomVectorNd.shape[1]

	if nExp != nExpFile:
		raise ValueError("ERROR! nExp (%d) not consistent with number of shots provided within sourceGeomFile (%d)"%(nExp,nExpFile))

	sourceAxis=Hypercube.axis(n=parObject.getInt("nExp"),o=0.0,d=1.0)

	for iExp in range(nExp):
		zCoord=SepVector.getSepVector(ns=[nSimSou],storage="dataDouble")
		xCoord=SepVector.getSepVector(ns=[nSimSou],storage="dataDouble")
		yCoord=SepVector.getSepVector(ns=[nSimSou],storage="dataDouble")

		#Setting z and x positions of the source for the given experiment
		zCoord.getNdArray()[:] = sourceGeomVectorNd[2,:,iExp]
		yCoord.getNdArray()[:] = sourceGeomVectorNd[1,:,iExp]
		xCoord.getNdArray()[:] = sourceGeomVectorNd[0,:,iExp]

		# Central grid
		sourcesVectorCenterGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),centerGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
		# Staggered grids for forces
		sourcesVectorXGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),xGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
		sourcesVectorYGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),yGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
		sourcesVectorZGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),zGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
		# Staggered grids for stresses
		sourcesVectorXZGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),xzGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
		sourcesVectorXYGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),xyGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
		sourcesVectorYZGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),yzGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))

	return sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,sourceAxis

# Build receivers geometry
def buildReceiversGeometry_3D(parObject,elasticParam):

	info = parObject.getInt("info",0)
	receiverGeomFile = parObject.getString("receiverGeomFile")
	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",0)
	xDipoleShift = parObject.getInt("xDipoleShift",0)
	yDipoleShift = parObject.getInt("yDipoleShift",0)
	receiversVector=[]
	spaceInterpMethod = parObject.getString("spaceInterpMethod","linear")

	# Get total number of shots
	nExp = parObject.getInt("nExp",-1)
	if (nExp==-1):
		raise ValueError("**** ERROR [buildReceiversGeometry_3D]: User must provide the total number of shots ****\n")

	# Check that user provides the number of receivers per shot (must be constant)
	nReceiverPerShot = parObject.getInt("nReceiverPerShot",-1)
	if (nReceiverPerShot == -1):
		raise ValueError("**** ERROR [buildReceiversGeometry_3D]: User must provide the total number of receivers per shot (this number must be the same for all shots [nReceiverPerShot]) ****\n")

	# Read parameters for spatial interpolation
	if (spaceInterpMethod == "linear"):
		hFilter1d = 1
	elif (spaceInterpMethod == "sinc"):
		hFilter1d = parObject.getInt("hFilter1d")
	else:
		raise ValueError("**** ERROR [buildReceiversGeometry_3D]: Spatial interpolation method requested by user is not implemented ****\n")

	# Display information for user
	if (info == 1):
		# print("**** [buildReceiversGeometry_3D]: User has requested to display information ****\n")
		if (spaceInterpMethod == "sinc"):
			print("**** [buildReceiversGeometry_3D]: User has requested a sinc spatial interpolation method for receivers' signals injection/extraction ****\n")
		else:
			print("**** [buildReceiversGeometry_3D]: User has requested a linear spatial interpolation method for receivers' signals injection/extraction ****\n")
		if (dipole == 1):
			print("**** [buildReceiversGeometry_3D]: User has requested a dipole source injection/extraction ****\n")
			print("**** [buildReceiversGeometry_3D]: Dipole shift in z-direction: %f [km or m]  ****\n" %zDipoleShift)
			print("**** [buildReceiversGeometry_3D]: Dipole shift in x-direction: %f [km or m]  ****\n" %xDipoleShift)
			print("**** [buildReceiversGeometry_3D]: Dipole shift in y-direction: %f [km or m]  ****\n" %yDipoleShift)

	# Getting axes
	zAxis = elasticParam.getHyper().getAxis(1)
	xAxis = elasticParam.getHyper().getAxis(2)
	yAxis = elasticParam.getHyper().getAxis(3)
	paramAxis = elasticParam.getHyper().getAxis(4)
	# Axis parameters
	nz=zAxis.n
	dz=zAxis.d
	oz=zAxis.o

	nx=xAxis.n
	dx=xAxis.d
	ox=xAxis.o

	ny=yAxis.n
	dy=yAxis.d
	oy=yAxis.o

	zAxisShifted=Hypercube.axis(n=nz,o=oz-0.5*dz,d=dz)
	xAxisShifted=Hypercube.axis(n=nx,o=ox-0.5*dx,d=dx)
	yAxisShifted=Hypercube.axis(n=ny,o=oy-0.5*dy,d=dy)

	centerGridHyper=Hypercube.hypercube(axes=[zAxis,xAxis,yAxis,paramAxis])
	xGridHyper=Hypercube.hypercube(axes=[zAxis,xAxisShifted,yAxis,paramAxis])
	yGridHyper=Hypercube.hypercube(axes=[zAxis,xAxis,yAxisShifted,paramAxis])
	zGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxis,yAxis,paramAxis])
	xzGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxisShifted,yAxis,paramAxis])
	xyGridHyper=Hypercube.hypercube(axes=[zAxis,xAxisShifted,yAxisShifted,paramAxis])
	yzGridHyper=Hypercube.hypercube(axes=[zAxisShifted,xAxis,yAxisShifted,paramAxis])

	# Create a receiver vector for centerGrid, x shifted, y shifted, z shifted,
	# xz shifted grid, xy shifted grid, and yz shifted grid
	recVectorCenterGrid=[]
	recVectorXGrid=[]
	recVectorYGrid=[]
	recVectorZGrid=[]
	recVectorXZGrid=[]
	recVectorXYGrid=[]
	recVectorYZGrid=[]

	# Read geometry file: 3 axes
	# 3 axes:
	# First (fastest) axis: experiment index [for now fixed for every source]
	# Second (slower) axis: receiver points
	# !!! The number of receivers per shot must be constant !!!
	# Third (slowest) axis: spatial coordinates [x,y,z]
	receiverGeomVectorNd = genericIO.defaultIO.getVector(receiverGeomFile,ndims=3).getNdArray()
	# Check consistency with total number of shots
	if (receiverGeomVectorNd.shape[2] != 1  and nExp != receiverGeomVectorNd.shape[2]):
		raise ValueError("**** ERROR [buildReceiversGeometry_3D]: Number of experiments from parfile (#shot=%s) not consistent with receivers' geometry file (#shots=%s) ****\n"%(nExp,receiverGeomVectorNd.shape[2]))

	# Read size of receivers' geometry file
	# Check consistency between the size of the receiver geometry file and the number of receivers per shot
	if(nReceiverPerShot != receiverGeomVectorNd.shape[1]):
		raise ValueError("**** ERROR [buildReceiversGeometry_3D]: Number of receivers from parfile (#receivers=%s) not consistent with receivers' geometry file (#receivers=%s) ****\n"%(nReceiverPerShot,receiverGeomVectorNd.shape[1]))

	nExp = receiverGeomVectorNd.shape[2]
	if (nExp==1 and info==1):
			print("**** [buildReceiversGeometry_3D]: User has requested a constant geometry (over shots) for receivers ****\n")

	receiverAxis=Hypercube.axis(n=nReceiverPerShot,o=0.0,d=1.0)

	for iExp in range(nExp):

		# Create inputs for devceiGpu_3D constructor
		zCoord=SepVector.getSepVector(ns=[nReceiverPerShot],storage="dataDouble")
		xCoord=SepVector.getSepVector(ns=[nReceiverPerShot],storage="dataDouble")
		yCoord=SepVector.getSepVector(ns=[nReceiverPerShot],storage="dataDouble")

		zCoordNd=zCoord.getNdArray()
		xCoordNd=xCoord.getNdArray()
		yCoordNd=yCoord.getNdArray()

		# Update the receiver's coordinates
		zCoordNd[:]=receiverGeomVectorNd[2,:,iExp]
		xCoordNd[:]=receiverGeomVectorNd[0,:,iExp]
		yCoordNd[:]=receiverGeomVectorNd[1,:,iExp]

		for iRec in range(nReceiverPerShot):
			# Central grid
			recVectorCenterGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),centerGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
			# Staggered grids for velocities
			recVectorXGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),xGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
			recVectorYGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),yGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
			recVectorZGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),zGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
			# Staggered grids for stresses
			recVectorXZGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),xzGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
			recVectorXYGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),xyGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))
			recVectorYZGrid.append(spaceInterpGpu_3D(zCoord.getCpp(),xCoord.getCpp(),yCoord.getCpp(),yzGridHyper.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,spaceInterpMethod,hFilter1d))


	return recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid,receiverAxis

################################################################################
############################### Nonlinear ######################################
################################################################################
def nonlinearOpInit_3D(args):
	"""
	   Function to initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# IO objects
	parObject=genericIO.io(params=sys.argv)

	# Time Axis
	nts=parObject.getInt("nts")
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts")
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	dummyAxis=Hypercube.axis(n=1)
	wavefieldAxis=Hypercube.axis(n=9)
	sourceGeomFile = parObject.getString("sourceGeomFile")

	# Allocate model
	sourceGeomVector = genericIO.defaultIO.getVector(sourceGeomFile,ndims=3)
	sourceSimAxis = sourceGeomVector.getHyper().getAxis(2)
	modelHyper=Hypercube.hypercube(axes=[timeAxis,sourceSimAxis,wavefieldAxis,dummyAxis])

	modelDouble=SepVector.getSepVector(modelHyper,storage="dataDouble")

	# elatic params
	elasticParam=parObject.getString("elasticParam", "noElasticParamFile")
	if (elasticParam == "noElasticParamFile"):
		print("**** ERROR: User did not provide elastic parameter file ****\n")
		sys.exit()
	elasticParamFloat=genericIO.defaultIO.getVector(elasticParam)
	elasticParamDouble=SepVector.getSepVector(elasticParamFloat.getHyper(),storage="dataDouble")
	#Converting model parameters to Rho|Lame|Mu if necessary [kg/m3|Pa|Pa]
	# 0 ==> correct parameterization
	# 1 ==> VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
	mod_par = parObject.getInt("mod_par",0)
	if(mod_par != 0):
		convOp = ElaConv_3D.ElasticConv_3D(elasticParamFloat,mod_par)
		elasticParamFloatTemp = elasticParamFloat.clone()
		convOp.forward(False,elasticParamFloatTemp,elasticParamFloat)
		del elasticParamFloatTemp

	#Conversion to double precision
	elasticParamDoubleNp=elasticParamDouble.getNdArray()
	elasticParamFloatNp=elasticParamFloat.getNdArray()
	elasticParamDoubleNp[:]=elasticParamFloatNp

	# Build sources/receivers geometry
	sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,sourceAxis=buildSourceGeometry_3D(parObject,elasticParamFloat)
	recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid,receiverAxis=buildReceiversGeometry_3D(parObject,elasticParamFloat)

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	# Outputs
	return  modelDouble,dataDouble,elasticParamDouble,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid

class nonlinearPropElasticShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear propagator"""

	def __init__(self,domain,range,elasticParam,paramP,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		self.pyOp = pyElastic_iso_double_nl_3D.nonlinearPropElasticShotsGpu_3D(elasticParam,paramP,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	# def forwardWavefield(self,add,model,data):
	# 	#Checking if getCpp is present
	# 	if("getCpp" in dir(model)):
	# 		model = model.getCpp()
	# 	if("getCpp" in dir(data)):
	# 		data = data.getCpp()
	# 	with pyElastic_iso_double_nl_3D.ostream_redirect():
	# 		self.pyOp.forwardWavefield(add,model,data)
	# 	return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	# def adjointWavefield(self,add,model,data):
	# 	#Checking if getCpp is present
	# 	if("getCpp" in dir(model)):
	# 		model = model.getCpp()
	# 	if("getCpp" in dir(data)):
	# 		data = data.getCpp()
	# 	with pyElastic_iso_double_nl_3D.ostream_redirect():
	# 		self.pyOp.adjointWavefield(add,model,data)
	# 	return

	# def getWavefield(self):
	# 	wavefield = self.pyOp.getWavefield()
	# 	return SepVector.doubleVector(fromCpp=wavefield)

	def setBackground(self,elasticParam):
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		with pyElastic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.setBackground(elasticParam)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyElastic_iso_double_nl_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result



################################################################################
################################### Born #######################################
################################################################################
def BornOpInitDouble_3D(args):
	"""
	   Function to correctly initialize Born operator
	   The function will return the necessary variables for operator construction
	"""
	# IO objects
	parObject=genericIO.io(params=sys.argv)

	# elatic params
	elasticParam=parObject.getString("elasticParam", "noElasticParamFile")
	if (elasticParam == "noElasticParamFile"):
		print("**** ERROR: User did not provide elastic parameter file ****\n")
		sys.exit()
	elasticParamFloat=genericIO.defaultIO.getVector(elasticParam)
	elasticParamDouble=SepVector.getSepVector(elasticParamFloat.getHyper(),storage="dataDouble")
	#Converting model parameters to Rho|Lame|Mu if necessary [kg/m3|Pa|Pa]
	# 0 ==> correct parameterization
	# 1 ==> VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
	mod_par = parObject.getInt("mod_par",0)
	if(mod_par != 0):
		convOp = ElaConv_3D.ElasticConv_3D(elasticParamFloat,mod_par)
		elasticParamFloatTemp = elasticParamFloat.clone()
		convOp.forward(False,elasticParamFloatTemp,elasticParamFloat)
		del elasticParamFloatTemp

	#Conversion to double precision
	elasticParamDoubleNp=elasticParamDouble.getNdArray()
	elasticParamFloatNp=elasticParamFloat.getNdArray()
	elasticParamDoubleNp[:]=elasticParamFloatNp


	# Time Axis
	nts=parObject.getInt("nts")
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts")
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	wavefieldAxis=Hypercube.axis(n=9)
	sourceGeomFile = parObject.getString("sourceGeomFile")

	# Read sources signals
	sourceGeomVector = genericIO.defaultIO.getVector(sourceGeomFile,ndims=3)
	sourceSimAxis = sourceGeomVector.getHyper().getAxis(2)
	sourceHyper=Hypercube.hypercube(axes=[timeAxis,sourceSimAxis,wavefieldAxis])

	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		raise IOError("**** ERROR: User did not provide seismic sources file ****")
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=3)
	sourcesSignalsDouble=SepVector.getSepVector(sourceHyper,storage="dataDouble")
	sourcesSignalsDoubleNp=sourcesSignalsDouble.getNdArray()
	sourcesSignalsFloatNp=sourcesSignalsFloat.getNdArray()
	sourcesSignalsDoubleNp[:]=sourcesSignalsFloatNp
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsDouble) # Create a vector of double3DReg slices

	# Build sources/receivers geometry
	sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,sourceAxis=buildSourceGeometry_3D(parObject,elasticParamFloat)
	recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid,receiverAxis=buildReceiversGeometry_3D(parObject,elasticParamFloat)

	# Allocate model
	modelDouble=SepVector.getSepVector(elasticParamDouble.getHyper(),storage="dataDouble")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,wavefieldAxis,sourceAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	# Outputs
	return
	modelDouble,dataDouble,elasticParamDouble,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid

class BornElasticShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for elastic Born propagator"""

	def __init__(self,domain,range,elasticParam,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		self.pyOp = pyElastic_iso_double_born_3D.BornElasticShotsGpu_3D(elasticParam,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def setBackground(self,elasticParam):
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		with pyElastic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.setBackground(elasticParam)
		return
