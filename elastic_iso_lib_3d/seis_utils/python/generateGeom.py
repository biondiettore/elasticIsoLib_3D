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
	min_offset = [no default] - float; minimum offset along the inline direction; (off = Xreceiver - XSource)!

	max_offset = [no default] - float; maximum offset along the inline direction; (off = Xreceiver - XSource)!

	min_cross_off = [no default] - float; minimum offset along the crossline direction; (off = Xreceiver - XSource)!

	max_cross_off = [no default] - float; maximum offset along the crossline direction; (off = Xreceiver - XSource)!

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

	# Error allowed for the soruce/receiver positions due to rounding error
	zTolerance=parObject.getFloat("zTolerance",0.1)
	xTolerance=parObject.getFloat("xTolerance",0.1)
	yTolerance=parObject.getFloat("yTolerance",0.1)
	zError=dz*zTolerance
	xError=dx*xTolerance
	yError=dy*yTolerance

	############################ Streamer geometry #############################

	if geom == "streamers":
		print("---- Streamer geometry ----\n")

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
		depth_source = firstShotz

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
		depth_rec = depth_cables

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
		print("---- Node geometry ----\n")

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
		depth_source = firstShotz

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
		depth_rec = firstRecz

		xPos = np.linspace(firstRecx, firstRecx+(nx_rec-1)*dx_rec, nx_rec)

		start = 0
		for ii_y in range(ny_rec):
			yPos = firstRecy + ii_y*dy_rec
			receiverGeomNd[0,start:start+nx_rec,0] = xPos
			receiverGeomNd[1,start:start+nx_rec,0] = yPos
			start += nx_rec

	else:
		raise ValueError("ERROR! Unknonw geometry type: %s"%geom)

	# General acquisition information
	print("Propagation domain size:")
	print("Extent in the x axis = %s [km or m]"%(max_x-ox))
	print("Extent in the y axis = %s [km or m]"%(max_y-oy))
	print("Extent in the z axis = %s [km or m]"%(max_z-oz))

	print("\nAcquisition geometry info:")
	max_shot_x = np.max(sourceGeomNd[0,:,:].ravel())
	max_shot_y = np.max(sourceGeomNd[1,:,:].ravel())
	min_shot_x = np.min(sourceGeomNd[0,:,:].ravel())
	min_shot_y = np.min(sourceGeomNd[1,:,:].ravel())
	print("Maximum x shot coordinate = %s"%max_shot_x)
	print("Maximum y shot coordinate = %s"%max_shot_y)
	print("Minimum x shot coordinate = %s"%min_shot_x)
	print("Minimum y shot coordinate = %s"%min_shot_y)

	max_rec_x = np.max(receiverGeomNd[0,:,:].ravel())
	max_rec_y = np.max(receiverGeomNd[1,:,:].ravel())
	min_rec_x = np.min(receiverGeomNd[0,:,:].ravel())
	min_rec_y = np.min(receiverGeomNd[1,:,:].ravel())
	print("Maximum x receiver coordinate = %s"%max_rec_x)
	print("Maximum y receiver coordinate = %s"%max_rec_y)
	print("Minimum x receiver coordinate = %s"%min_rec_x)
	print("Minimum y receiver coordinate = %s"%min_rec_y)

	# Checking if any receiver or shot is outside of the propagation boundaries
	if ox-min_shot_x > xError or max_shot_x-max_x > xError or oy-min_shot_y > yError or max_shot_y-max_y > yError or not oz-zError < depth_source < max_z+zError:
		raise ValueError("ERROR! One or more shots are outside of the propagation domain")
	if ox-min_rec_x > xError or max_rec_x-max_x > xError or oy-min_rec_y > yError or max_rec_y-max_y > yError or not oz-zError < depth_rec < max_z+zError:
		raise ValueError("ERROR! One or more receivers is outside of the propagation domain")

	sourceGeom.writeVec(sourceGeomFile)
	receiverGeom.writeVec(receiverGeomFile)




























#
