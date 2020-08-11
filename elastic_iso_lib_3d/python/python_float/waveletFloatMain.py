#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

if __name__ == '__main__':

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Ali's wavelet
	if (parObject.getString("type", "ali") == "ali"):

		# Time parameters
		nts=parObject.getInt("nts")
		dts=parObject.getFloat("dts",-1.0)
		ots=0.0
		timeDelay=parObject.getFloat("timeDelay",0.0)

		# Time signal
		timeAxis=Hypercube.axis(n=nts,o=ots,d=dts,label="Time")
		waveletHyper=Hypercube.hypercube(axes=[timeAxis])
		wavelet=SepVector.getSepVector(waveletHyper)
		waveletNd=wavelet.getNdArray();
		odd = False
		if(nts%2 != 0):
			nts+=1
			odd=True
		waveletFftNd=np.zeros(nts,dtype=np.complex64)

		# Frequency parameters
		f1=parObject.getFloat("f1",-1.0)
		f2=parObject.getFloat("f2",-1.0)
		f3=parObject.getFloat("f3",-1.0)
		f4=parObject.getFloat("f4",-1.0)

		if (not (f1 < f2 <= f3 < f4) ):
			raise ValueError("**** ERROR: Corner frequencies values must be increasing ****\n")

		# Check if f4 < fNyquist
		fNyquist=1/(2*dts)
		if (f4 > fNyquist):
			raise ValueError("**** ERROR: f4 > fNyquist ****\n")

		df=1.0/((nts)*dts)

		for iFreq in range(nts//2):
			f=iFreq*df # Loop over frequencies
			if (f < f1):
				waveletFftNd[iFreq]=0
			elif (f1 <= f < f2):
				waveletFftNd[iFreq]=np.cos(np.pi/2.0*(f2-f)/(f2-f1))*np.cos(np.pi/2.0*(f2-f)/(f2-f1))
				waveletFftNd[iFreq]=waveletFftNd[iFreq]*np.exp(-1j*2.0*np.pi*f*timeDelay)
			elif (f2 <= f < f3):
				waveletFftNd[iFreq]=1.0
				waveletFftNd[iFreq]=waveletFftNd[iFreq]*np.exp(-1j*2.0*np.pi*f*timeDelay)
			elif (f3 <= f < f4):
				waveletFftNd[iFreq]=np.cos(np.pi/2.0*(f-f3)/(f4-f3))*np.cos(np.pi/2.0*(f-f3)/(f4-f3))
				waveletFftNd[iFreq]=waveletFftNd[iFreq]*np.exp(-1j*2.0*np.pi*f*timeDelay)
			elif(f >= f4):
				waveletFftNd[iFreq]=0

		# Duplicate, flip spectrum and take the complex conjugate
		waveletFftNd[nts//2+1:] = np.flip(waveletFftNd[1:nts//2].conj(),axis=0)

		# Apply inverse FFT
		if(odd):
			waveletNd[:]=np.fft.ifft(waveletFftNd[:-1]).real #*2.0/np.sqrt(nts)
		else:
			waveletNd[:]=np.fft.ifft(waveletFftNd[:]).real #*2.0/np.sqrt(nts)

		# Write wavelet to disk
		waveletFile=parObject.getString("wavelet")
		genericIO.defaultIO.writeVector(waveletFile,wavelet)


	# Ricker wavelet
	elif (parObject.getString("type","ali") == "ricker"):

		# Time parameters
		nts=parObject.getInt("nts")
		dts=parObject.getFloat("dts",0.0)
		ots=0.0
		timeDelay=parObject.getFloat("timeDelay",0.0)

		# Allocate wavelet
		timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
		waveletHyper=Hypercube.hypercube(axes=[timeAxis])
		wavelet=SepVector.getSepVector(waveletHyper)
		waveletNd=wavelet.getNdArray();

		fDom=parObject.getFloat("fDom",0.0)
		alpha=(np.pi*fDom)*(np.pi*fDom)
		for its in range(nts):
			t=ots+its*dts
			t=t-timeDelay
			waveletNd[its]=(1-2.0*alpha*t*t)*np.exp(-1.0*alpha*t*t)

		# Write wavelet to disk
		waveletFile=parObject.getString("wavelet")
		genericIO.defaultIO.writeVector(waveletFile,wavelet)

	else:
		raise ValueError("**** ERROR: Wavelet type not supported (iz bazicly minz) ****")
