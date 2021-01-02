#!/usr/bin/env python3
"""
Elastic data sampling operator

USAGE EXAMPLE:
	dataCompMain_3D.py input= output= comp=vx,vz,sxx,szz,sxz

INPUT PARAMETERS:
	input  = [no default] - string; Header file defining elastic data directly outputted by elastic propagator

	output = [no default] - string; Header file defining elastic data sampled according to user-defined comp

	comp = [no default] - string; Comma-separated list of the output/sampled components (e.g., 'vx,vz,sxx,szz,sxz' or 'p,vx,vz'; the order matters!).
	                              Currently, supported: vx,vy,vz,sxx,syy,szz,sxz,sxy,syz,p (i.e., p = 0.5*(sxx+syy+szz))

	dpTest = [False] - boolean; Dot-product test for the specified sampling operation

"""

import genericIO
import SepVector
from dataCompModule_3D import ElasticDatComp_3D
import numpy as np
import sys
import os.path


if __name__ == '__main__':
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	########################## PARSE COMMAND LINE ##############################
	# IO object
	parObject=genericIO.io(params=sys.argv)

	#check if verbose
	verbose = parObject.getInt("verbose",0)
	#check conversion kind
	comp = parObject.getString("comp")

	#read in file if it exists
	inputFile=parObject.getString("input","noFile")
	if (not os.path.isfile(inputFile)) or (inputFile == "noFile"):
		raise IOError("ERROR! no inputFile specified or given file does not exist!")
	model = genericIO.defaultIO.getVector(inputFile,ndims=4)

	#obtain output file
	outputFile=parObject.getString("output","noFile")
	if outputFile == "noFile":
		raise IOError("ERROR! no outputFile specified!")

	############################ Apply operator ################################

	sampOp = ElasticDatComp_3D(comp,model)
	data = sampOp.range.clone()

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		sampOp.dotTest(True)
		quit()

	#Applying sampling operator
	sampOp.forward(False,model,data)

	############################## Write output ################################
	genericIO.defaultIO.writeVector(outputFile, data)
