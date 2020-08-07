#!/usr/bin/env python3
"""
Synthetic geometry generator

USAGE EXAMPLE:
	generateGeom.py

INPUT PARAMETERS:
	geom  = [no default] - string; Type of acquisition geometry: [streamers, nodes]

	modelFile = [no default] - string; Subsurface model parameters to obtain domain information (e.g., nx, dx, ox, ...)

	sourceGeomFile = [no default] - string; Generated source geometry file

	receiverGeomFile = [no default] - string; Generated receiver geometry file

	######################################
	# PARAMETERS FOR STREAMER ACQUISITION#
	######################################

	# Source position
	firstShotPos = [ox,oy,oz] - floats; x,y,z coordinates of the first shot [same depth for all shot!]

	nShot_inline = [no default] - int; number of shots in the inline direction

	nShot_crossline = [no default] - int; number of shots in the crossline direction

	dShot_inline = [no default] - float; Shot sampling along the inline direction

	dShot_crossline = [no default] - float; Shot sampling along the crossline direction

	inline_dir = [X] - string; Inline axis

	# Receiver position (relative to a given shot)
	min_offset = [no default] - float; minimum offset along the inline direction; sign matters!

	max_offset = [no default] - float; maximum offset along the inline direction; sign matters!

	min_cross_off = [no default] - float; minimum offset along the crossline direction

	max_cross_off = [no default] - float; maximum offset along the crossline direction

	n_recs = [no default] - int; number of receivers along the cable

	n_cables = [no default] - int; number of cables

	depth_cables = [no default] - float; depth of the receivers/cables

	##################################
	# PARAMETERS FOR NODE ACQUISITION#
	##################################

	firstShotPos = [ox,oy,oz] - floats; x,y,z coordinates of the first shot [same depth for all shot!]

	dx_shot = [no default] - float; Sampling of shot carpet along x direction

	dy_shot = [no default] - float; Sampling of shot carpet along y direction

	nx_shot = [no default] - int; Number of shots along x direction

	ny_shot = [no default] - int; Number of shots along y direction

	firstRecPos = [ox,oy,oz] - floats; x,y,z coordinates of the first node [same depth for all nodes!]

	dx_rec = [no default] - float; Sampling of node carpet along x direction

	dy_rec = [no default] - float; Sampling of node carpet along y direction

	nx_rec = [no default] - int; Number of nodes along x direction

	ny_rec = [no default] - int; Number of nodes along y direction

"""

import genericIO
import SepVector
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

	geom = parObject.getString("geom")

	modelFile = parObject.getString("modelFile")
	modelHyper = genericIO.defaultIO.getRegFile(modelFile).getHyper()

	# Output file names
	sourceGeomFile = parObject.getString("sourceGeomFile")
	receiverGeomFile = parObject.getString("receiverGeomFile")

	########################## MODEL INFORMATION ###############################
	# Z axis
	zAxis = modelHyper.getAxis(1)
	nz = zAxis.n
	dz = zAxis.d
	oz = zAxis.o
	max_z = oz + (nz-1) * dz
	# X axis
	xAxis = modelHyper.getAxis(2)
	nx = xAxis.n
	dx = xAxis.d
	ox = xAxis.o
	max_x = ox + (nx-1) * dx
	# Y axis
	yAxis = modelHyper.getAxis(3)
	ny = yAxis.n
	dy = yAxis.d
	oy = yAxis.o
	max_y = oy + (ny-1) * dy

	############################ Streamer geometry #############################

	if geom == "streamers":
		print("---- Streamer geometry ----")

		# Source geometry generation
		firstShotx, firstShoty, firstShotz = parObject.getFloats("firstShotPos", [ox,oy,oz])
		ns_inline = parObject.getInt("nShot_inline")
		ns_crossline = parObject.getInt("nShot_crossline")
		ds_inline = parObject.getInt("dShot_inline")
		ds_crossline = parObject.getInt("dShot_crossline")
		inline_dir = parObject.getString("inline_dir","X")
		Nshots = ns_inline*ns_crossline

		# Getting inline direction
		if inline_dir in "xX":
			x_idx = 0
			y_idx = 1
		else:
			x_idx = 1
			y_idx = 0

		sourceGeom = SepVector.getSepVector(ns=[Nshots,1,3])
		sourceGeomNd = sourceGeom.getNdArray()
		# Set sources' depth
		sourceGeomNd[2,:,:] = firstShotz

		xPos = np.linspace(firstShotx, firstShotx+(ns_inline-1)*ds_inline, ns_inline)

		start = 0
		for ii_cross in range(ns_crossline):
			yPos = firstShoty + ii_cross*ds_crossline
			sourceGeomNd[x_idx,:,start:start+ns_inline] = xPos
			sourceGeomNd[y_idx,:,start:start+ns_inline] = yPos
			start += ns_inline

		# Receiver geometry
		n_recs = parObject.getInt("n_recs")
		n_cables = parObject.getInt("n_cables")
		min_offset = parObject.getFloat("min_offset")
		max_offset = parObject.getFloat("max_offset")
		min_cross_off = parObject.getFloat("min_cross_off")
		max_cross_off = parObject.getFloat("max_cross_off")
		depth_cables = parObject.getFloat("depth_cables")

		Nrecs = n_recs*n_cables

		receiverGeom = SepVector.getSepVector(ns=[Nshots,Nrecs,3])
		receiverGeomNd = receiverGeom.getNdArray()

		inlineCable = np.linspace(min_offset,max_offset,n_recs)
		crossCable = np.linspace(min_cross_off,max_cross_off,n_cables)

		xPos = np.zeros(Nrecs)
		yPos = np.zeros(Nrecs)

		# Setting relative positions of the receivers
		start = 0
		for ii in range(n_cables):
			xPos[start:start+n_recs] = inlineCable
			yPos[start:start+n_recs] = crossCable[ii]
			start += n_recs

		# Setting receiver positions with respect to give source
		for iShot in range(Nshots):
			receiverGeomNd[x_idx,:,iShot] = xPos[:] + sourceGeomNd[x_idx,0,iShot]
			receiverGeomNd[y_idx,:,iShot] = yPos[:] + sourceGeomNd[y_idx,0,iShot]

		# Setting depth of the receivers
		receiverGeomNd[2,:,:] = depth_cables



	############################ Node geometry #################################
	elif geom == "nodes":
		print("---- Node geometry ----")

		# Shot geometry
		firstShotx, firstShoty, firstShotz = parObject.getFloats("firstShotPos", [ox,oy,oz])
		dx_shot = parObject.getFloat("dx_shot")
		nx_shot = parObject.getInt("nx_shot")
		dy_shot = parObject.getFloat("dy_shot")
		ny_shot = parObject.getInt("ny_shot")

		Nshots = nx_shot*ny_shot

		sourceGeom = SepVector.getSepVector(ns=[Nshots,1,3])
		sourceGeomNd = sourceGeom.getNdArray()
		# Set sources' depth
		sourceGeomNd[2,:,:] = firstShotz

		xPos = np.linspace(firstShotx, firstShotx+(nx_shot-1)*dx_shot, nx_shot)

		start = 0
		for ii_y in range(ny_shot):
			yPos = firstShoty + ii_y*dy_shot
			sourceGeomNd[0,:,start:start+nx_shot] = xPos
			sourceGeomNd[1,:,start:start+nx_shot] = yPos
			start += nx_shot


		# Node geometry
		firstRecx, firstRecy, firstRecz = parObject.getFloats("firstRecPos", [ox,oy,oz])
		dx_rec = parObject.getFloat("dx_rec")
		nx_rec = parObject.getInt("nx_rec")
		dy_rec = parObject.getFloat("dy_rec")
		ny_rec = parObject.getInt("ny_rec")

		Nrecs = nx_rec*ny_rec

		receiverGeom = SepVector.getSepVector(ns=[1,Nrecs,3])
		receiverGeomNd = receiverGeom.getNdArray()
		# Set sources' depth
		receiverGeomNd[2,:,:] = firstRecz

		xPos = np.linspace(firstRecx, firstRecx+(nx_rec-1)*dx_rec, nx_rec)

		start = 0
		for ii_y in range(ny_rec):
			yPos = firstRecy + ii_y*dy_rec
			receiverGeomNd[0,start:start+nx_rec,0] = xPos
			receiverGeomNd[1,start:start+nx_rec,0] = yPos
			start += nx_rec

	else:
		raise ValueError("ERROR! Unknonw geometry type: %s"%geom)

	sourceGeom.writeVec(sourceGeomFile)
	receiverGeom.writeVec(receiverGeomFile)




























#
