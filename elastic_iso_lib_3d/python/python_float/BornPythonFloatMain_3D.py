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
	modelFloat,dataFloat,elasticParamFloat,parObject,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid = Elastic_iso_float_3D.BornOpInitFloat_3D(sys.argv)

	# Construct nonlinear operator object
	BornElasticOp=Elastic_iso_float_3D.BornElasticShotsGpu_3D(modelFloat,dataFloat,elasticParamFloat,parObject.param,sourcesSignalsVector,sourcesVectorCenterGrid,sourcesVectorXGrid,sourcesVectorYGrid,sourcesVectorZGrid,sourcesVectorXZGrid,sourcesVectorXYGrid,sourcesVectorYZGrid,recVectorCenterGrid,recVectorXGrid,recVectorYGrid,recVectorZGrid,recVectorXZGrid,recVectorXYGrid,recVectorYZGrid)

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		BornElasticOp.dotTest(True)
		quit()

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("----------------------------------------------------------------------")
		print("------------------ Running Python Born Elastic forward ---------------")
		print("----------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name ****\n")

		#Reading model
		modelFloat=genericIO.defaultIO.getVector(modelFile)

		# Apply forward
		BornElasticOp.forward(False,modelFloat,dataFloat)

		# Write data
		dataFloat.writeVec(dataFile)


	# Adjoint
	else:
		print("----------------------------------------------------------------------")
		print("------------------ Running Python Born Elastic adjoint ---------------")
		print("----------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name ****\n")

		#Reading model
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=4)

		# Apply adjoint
		BornElasticOp.adjoint(False,modelFloat,dataFloat)

		# Write data
		modelFloat.writeVec(modelFile)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")
