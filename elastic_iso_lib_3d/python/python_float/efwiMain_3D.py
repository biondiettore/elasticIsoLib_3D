#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Elastic_iso_float_3D
import elasticParamConvertModule_3D as ElaConv_3D
from dataCompModule_3D import ElasticDatComp_3D
import interpBSplineModule_3D
# import dataTaperModule
# import spatialDerivModule
# import maskGradientModule

# Solver library
import pyOperator as pyOp
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStepper as Stepper
import inversionUtils_3D
from sys_util import logger

############################ Bounds vectors ####################################
# Create bound vectors for FWI
def createBoundVectors(parObject,model,inv_log):

	# Get model dimensions
	nz=parObject.getInt("nz")
	nx=parObject.getInt("nx")

	# Min bound
	minBoundVectorFile=parObject.getString("minBoundVector","noMinBoundVectorFile")
	if (minBoundVectorFile=="noMinBoundVectorFile"):
		minBound1=parObject.getFloat("minBound_par1",-np.inf)
		minBound2=parObject.getFloat("minBound_par2",-np.inf)
		minBound3=parObject.getFloat("minBound_par3",-np.inf)
		if(minBound1 == minBound2 == minBound3 == -np.inf):
			minBoundVector = None
		else:
			if(pyinfo): print("--- User provided minimum bounds ---")
			inv_log.addToLog("--- User provided minimum bounds ---")
			minBoundVector=model.clone()
			minBoundVector.set(0.0)
			minBoundVectorNd=minBoundVector.getNdArray()
			minBoundVectorNd[0,:]=minBound1
			minBoundVectorNd[1,:]=minBound2
			minBoundVectorNd[2,:]=minBound3
	else:
		if(pyinfo): print("--- User provided a minimum-bound vector ---")
		inv_log.addToLog("--- User provided a minimum-bound vector ---")
		minBoundVector=genericIO.defaultIO.getVector(minBoundVectorFile)

	# Max bound
	maxBoundVectorFile=parObject.getString("maxBoundVector","noMaxBoundVectorFile")
	if (maxBoundVectorFile=="noMaxBoundVectorFile"):
		maxBound1=parObject.getFloat("maxBound_par1",np.inf)
		maxBound2=parObject.getFloat("maxBound_par2",np.inf)
		maxBound3=parObject.getFloat("maxBound_par3",np.inf)
		if(maxBound1 == maxBound2 == maxBound3 == np.inf):
			maxBoundVector = None
		else:
			if(pyinfo): print("--- User provided maximum bounds ---")
			inv_log.addToLog("--- User provided maximum bounds ---")
			maxBoundVector=model.clone()
			maxBoundVector.set(0.0)
			maxBoundVectorNd=maxBoundVector.getNdArray()
			maxBoundVectorNd[0,:]=maxBound1
			maxBoundVectorNd[1,:]=maxBound2
			maxBoundVectorNd[2,:]=maxBound3

	else:
		if(pyinfo): print("--- User provided a maximum-bound vector ---")
		inv_log.addToLog("--- User provided a maximum-bound vector ---")
		maxBoundVector=genericIO.defaultIO.getVector(maxBoundVectorFile)


	return minBoundVector,maxBoundVector
	
	
# Elastic FWI workflow script
if __name__ == '__main__':
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

    # IO object
	parObject = genericIO.io(params=sys.argv)

	# Checking if Dask was requested
	# client, nWrks = Elastic_iso_float_prop.create_client(parObject)

	pyinfo=parObject.getInt("pyinfo",1)
	spline=parObject.getInt("spline",0)
	dataTaper=parObject.getInt("dataTaper",0)
	# regType=parObject.getString("reg","None")
	# reg=0
	# if (regType != "None"): reg=1
	# epsilonEval=parObject.getInt("epsilonEval",0)

	# Nonlinear solver
	solverType=parObject.getString("solver")
	stepper=parObject.getString("stepper","default")

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils_3D.inversionInit(sys.argv)

	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("----------------------------------------------------------------------")
	if(pyinfo): print("------------------------ Elastic FWI logfile -------------------------")
	if(pyinfo): print("----------------------------------------------------------------------\n")
	inv_log.addToLog("------------------------ Elastic FWI logfile -------------------------")
	
	############################# Initialization ###############################

	# FWI nonlinear operator
	modelInit,modelInitConv,dataFloat,sourceFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid = Elastic_iso_float_3D.nonlinearFwiOpInitFloat_3D(sys.argv)

	# Born operator
	_,_,_,_,sourcesSignalsVector,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = Elastic_iso_float_3D.BornOpInitFloat_3D(sys.argv)
	
	############################# Read files ###################################
	# Seismic source
	# Read within nonlinearFwiOpInitFloat

	# Data
	dataFile=parObject.getString("data","noDataFile")
	if (dataFile == "noDataFile"):
		raise IOError("**** ERROR: User did not provide data file (data) ****\n")
	data=genericIO.defaultIO.getVector(dataFile,ndims=4)

	############################# Gradient mask ################################
	maskGradientFile=parObject.getString("maskGradient","NoMask")
	if (maskGradientFile=="NoMask"):
		maskGradientOp=None
	else:
		if(pyinfo): print("--- User provided a mask for the gradients ---")
		inv_log.addToLog("--- User provided a mask for the gradients ---")
		maskGradient=genericIO.defaultIO.getVector(maskGradientFile)
		maskGradientOp = pyOp.DiagonalOp(maskGradient)

	############################# Instantiation ################################
	
	# Nonlinear
	nonlinearElasticOp=Elastic_iso_float_3D.nonlinearFwiPropElasticShotsGpu_3D(modelInitConv,dataFloat,sourceFloat,parObject.param,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid)

	# Construct nonlinear operator object
	BornElasticOp=Elastic_iso_float_3D.BornElasticShotsGpu_3D(modelInit,dataFloat,modelInitConv,parObject.param,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid)

	#Born operator pointer for inversion
	BornElasticOpInv=BornElasticOp
	
	# Conventional FWI non-linear operator (with mask if requested)
	if maskGradientOp is not None:
		BornElasticOpInv=pyOp.ChainOperator(maskGradientOp,BornElasticOp)

	fwiInvOp=pyOp.NonLinearOperator(nonlinearElasticOp,BornElasticOpInv,BornElasticOp.setBackground)

	# Elastic parameter conversion if any
	mod_par = parObject.getInt("mod_par",0)
	if(mod_par != 0):
		convOp = ElaConv_3D.ElasticConv_3D(modelInit,mod_par)
		#Jacobian
		convOpJac = ElaConv_3D.ElasticConvJab_3D(modelInit,modelInit,mod_par)
		#Creating non-linear operator
		convOpNl=pyOp.NonLinearOperator(convOp,convOpJac,convOpJac.setBackground)
		#Chaining non-linear operators if not using Lame,Mu,Density parameterization
		#f(g(m)) where f is the non-linear modeling operator and g is the non-linear change of variables
		fwiInvOp=pyOp.CombNonlinearOp(convOpNl,fwiInvOp)
	
	########################### Data components ################################
	comp = parObject.getString("comp")
	if(comp != "vx,vy,vz,sxx,syy,szz,sxz,sxy,syz"):
		sampOp = ElasticDatComp_3D(comp,BornElasticOp.getRange())
		sampOpNl = pyOp.NonLinearOperator(sampOp,sampOp)
		#modeling operator = Sf(m)
		fwiInvOp=pyOp.CombNonlinearOp(fwiInvOp,sampOpNl)
	else:
		if(not dataFloat.checkSame(data)):
			raise ValueError("ERROR! The input data have different size of the expected inversion data! Check your arguments and paramater file")
	
	##################### Data muting with mask ################################
	dataMaskFile = parObject.getString("dataMaskFile","noDataMask")
	if (dataMaskFile!="noDataMask"):
		if(pyinfo): print("--- User provided a mask for the data ---")
		inv_log.addToLog("--- User provided a mask for the data ---")
		dataMask = genericIO.defaultIO.getVector(dataMaskFile,ndims=4)
		dataMaskOp = pyOp.DiagonalOp(dataMask)
		# Creating non-linear operator and concatenating with fwiInvOp
		dataMaskNl = pyOp.NonLinearOperator(dataMaskOp,dataMaskOp)
		fwiInvOp=pyOp.CombNonlinearOp(fwiInvOp,dataMaskNl)
		data_tmp = data.clone()
		dataMaskOp.forward(False,data,data_tmp)
		data = data_tmp
	
	############################# Spline operator ##############################
	if (spline==1):
		if(pyinfo): print("--- Using spline interpolation ---")
		inv_log.addToLog("--- Using spline interpolation ---")
		# Coarse-grid model
		modelCoarseInitFile=parObject.getString("modelCoarseInit")
		modelCoarseInit=genericIO.defaultIO.getVector(modelCoarseInitFile)
		modelFineInit=modelInit
		modelInit=modelCoarseInit
		# Parameter parsing
		_,_,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat=interpBSplineModule_3D.bSplineIter3dInit(sys.argv,zFat=0,xFat=0,yFat=0)
		splineOp=interpBSplineModule_3D.bSplineIter3d(modelCoarseInit,modelFineInit,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator
		fwiInvOp=pyOp.CombNonlinearOp(splineNlOp,fwiInvOp)
	
	############################### Bounds #####################################
	minBoundVector,maxBoundVector=createBoundVectors(parObject,modelInit,inv_log)

	########################### Inverse Problem ################################
	fwiProb=Prblm.ProblemL2NonLinear(modelInit,data,fwiInvOp,minBound=minBoundVector,maxBound=maxBoundVector)

	############################# Solver #######################################
	# Nonlinear conjugate gradient
	if (solverType=="nlcg"):
		nlSolver=NLCG(stop,logger=inv_log)
	# LBFGS
	elif (solverType=="lbfgs"):
		illumination_file=parObject.getString("illumination","noIllum")
		H0_Op = None
		if illumination_file != "noIllum":
			print("--- Using illumination as initial Hessian inverse ---")
			illumination=genericIO.defaultIO.getVector(illumination_file)
			H0_Op = pyOp.DiagonalOp(illumination)
		nlSolver = LBFGS(stop, H0=H0_Op, logger=inv_log)
	# Steepest descent
	elif (solverType=="sd"):
		nlSolver=NLCG(stop,beta_type="SD",logger=inv_log)
	else:
		raise ValueError("ERROR! Provided unknonw solver type: %s"%(solverType))

	############################# Stepper ######################################
	if (stepper == "parabolic"):
		nlSolver.stepper.eval_parab=True
	elif (stepper == "linear"):
		nlSolver.stepper.eval_parab=False
	elif (stepper == "parabolicNew"):
		nlSolver.stepper = Stepper.ParabolicStepConst()
	elif (stepper == "default"):
		pass
	else:
		raise ValueError("ERROR! Provided unknonw stepper type: %s"%(stepper))

	####################### Manual initial step length #########################
	initStep=parObject.getFloat("initStep",-1.0)
	if (initStep>0):
		nlSolver.stepper.alpha=initStep

	nlSolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	nlSolver.run(fwiProb,verbose=info)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")