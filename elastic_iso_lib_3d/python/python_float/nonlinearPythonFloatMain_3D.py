#!/usr/bin/env python3
import sys
import genericIO
import SepVector
import Hypercube
import Elastic_iso_float_3D
import numpy as np
import time

if __name__ == '__main__':
	# Initialize operator
	modelFloat,dataFloat,elasticParamFloat,parObject,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid = Elastic_iso_float_3D.nonlinearOpInit_3D(sys.argv)

	# Construct nonlinear operator object
	nonlinearElasticOp=Elastic_iso_float_3D.nonlinearPropElasticShotsGpu_3D(modelFloat,dataFloat,elasticParamFloat,parObject.param,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid)

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		nonlinearElasticOp.dotTest(True)
		quit()

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("------------------ Running Python nonlinear forward ---------------")
		print("-------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file [model] ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name [data] ****\n")
		modelInput=genericIO.defaultIO.getVector(modelFile,ndims=4)
		modelFloatNp=modelFloat.getNdArray()
		modelInputNp=modelInput.getNdArray()
		modelFloatNp.flat[:]=modelInputNp

		#check if we want to save wavefield
		if (parObject.getInt("saveWavefield",0) == 1):
			raise NotImplementError("ERROR! saveWavefield option not supported yet")
			# wfldFile=parObject.getString("wfldFile","noWfldFile")
			# if (wfldFile == "noWfldFile"):
			# 	raise IOError("**** ERROR: User specified saveWavefield=1 but did not provide wavefield file name (wfldFile)****")
			# #run Nonlinear forward with wavefield saving
			# nonlinearElasticOp.forwardWavefield(False,modelFloat,dataFloat)
			# #save wavefield to disk
			# wavefieldFloat = nonlinearElasticOp.getWavefield()
			# genericIO.defaultIO.writeVector(wfldFile,wavefieldFloat)
		else:
			#run Nonlinear forward without wavefield saving
			nonlinearElasticOp.forward(False,modelFloat,dataFloat)
		#write data to disk
		dataFloat.writeVec(dataFile)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("------------------ Running Python nonlinear adjoint ---------------")
		print("-------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name ****\n")

		#Reading model
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=4)

		#check if we want to save wavefield
		if (parObject.getInt("saveWavefield",0) == 1):
			raise NotImplementError("ERROR! saveWavefield option not supported yet")
			# wfldFile=parObject.getString("wfldFile","noWfldFile")
			# if (wfldFile == "noWfldFile"):
			# 	raise IOError("**** ERROR: User specified saveWavefield=1 but did not provide wavefield file name (wfldFile)****")
			# #run Nonlinear adjoint with wavefield saving
			# nonlinearElasticOp.adjointWavefield(False,modelFloat,dataFloat)
			# #save wavefield to disk
			# wavefieldFloat = nonlinearElasticOp.getWavefield()
			# genericIO.defaultIO.writeVector(wfldFile,wavefieldFloat)
		else:
			#run Nonlinear forward without wavefield saving
			nonlinearElasticOp.adjoint(False,modelFloat,dataFloat)
		#write data to disk
		modelFloat.writeVec(modelFile)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
