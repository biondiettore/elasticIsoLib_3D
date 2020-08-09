#Module containing definition for elastic parameter conversion
import pyOperator
import numpy as np

class ElasticConv_3D(pyOperator.Operator):
	"""
	   Operator class to convert elastic parameters
	"""

	def __init__(self,domain,conv_type):
		"""
		   Constructor for elastic parameter conversion:
		   domain    	= [no default] - vector class; Vector for defining domain of the transformation
		   conv_type    = [no default] - int; Conversion kind
			1 = VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
			2 = VpVsRho to RhoLameMu (km/s|km/s|g/cm3 -> kg/m3|Pa|Pa)
		   -1 = RhoLameMu to VpVsRho (m/s|m/s|kg/m3 <- kg/m3|Pa|Pa)
		"""
		self.setDomainRange(domain,domain)
		self.conv_type=conv_type
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		#Getting pointers to input arrays
		modelNd = model.getNdArray()
		dataNd = data.getNdArray()

		if(not add): data.zero()
		if(self.conv_type == 1):
			#VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
			dataNd[0,:] += modelNd[2,:] #rho
			dataNd[1,:] += modelNd[2,:]*(modelNd[0,:]*modelNd[0,:]-2.0*modelNd[1,:]*modelNd[1,:]) #lame
			dataNd[2,:] += modelNd[2,:]*modelNd[1,:]*modelNd[1,:] #mu
		elif(self.conv_type == 2):
			#VpVsRho to RhoLameMu (km/s|km/s|g/m3 -> kg/m3|Pa|Pa)
			dataNd[0,:] += modelNd[2,:]*1e3 #rho
			dataNd[1,:] += modelNd[2,:]*(modelNd[0,:]*modelNd[0,:]-2.0*modelNd[1,:]*modelNd[1,:])*1e9 #lame
			dataNd[2,:] += modelNd[2,:]*modelNd[1,:]*modelNd[1,:]*1e9 #mu
		elif(self.conv_type == -1):
			#RhoLameMu to VpVsRho (kg/m3|Pa|Pa -> m/s|m/s|kg/m3)
			dataNd[0,:] += np.sqrt(np.divide((modelNd[1,:]+2*modelNd[2,:]),modelNd[0,:])) #vp
			dataNd[1,:] += np.sqrt(np.divide(modelNd[2,:],modelNd[0,:])) #vs
			dataNd[2,:] += modelNd[0,:] #rho
		else:
			raise ValueError("ERROR! Unsupported conv_type (current value: %s). Please redefine it!"%(self.conv_type))
		return



class ElasticConvJab_3D(pyOperator.Operator):
	"""
	   Operator class of the Jacobian of elastic parameters conversion
	"""


	def __init__(self,domain,background,conv_type):
		"""
		   Constructor for elastic parameter conversion Jacobian matrix:
		   domain    	= [no default] - vector class; Vector for defining domain of the transformation
		   background  	= [no default] - vector class; Vector defining the background elastic parameters on which the Jacobian is computed
		   conv_type    = [no default] - int; Conversion kind
		    1 = VpVsRho to RhoLameMu (m/s|m/s|kg/m3 -> kg/m3|Pa|Pa)
			2 = VpVsRho to RhoLameMu (km/s|km/s|g/cm3 -> kg/m3|Pa|Pa)
		"""
		self.setDomainRange(domain,domain)
		self.background=domain.clone() #Background elastic model
		self.conv_type=conv_type
		return


	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		#Getting pointers to input arrays
		modelNd = model.getNdArray()
		backgroundNd = self.background.getNdArray()
		dataNd = data.getNdArray()

		if(not add): data.zero()
		if(self.conv_type == 1 or self.conv_type == 2):
			#rho' = rho => drho'/drho = 1
			dataNd[0,:] += modelNd[2,:]
			#lame = (Vp*Vp-2*Vs*Vs)*rho => dlame/drho = (Vp*Vp-2*Vs*Vs), dlame/dVs = -4*Vs*Vs*rho, dlame/dVp = 2*Vp*Vp*rho
			dataNd[1,:] += (backgroundNd[0,:]*backgroundNd[0,:]-2.0*backgroundNd[1,:]*backgroundNd[1,:])*modelNd[2,:] - 4.0*backgroundNd[1,:]*backgroundNd[1,:]*modelNd[1,:] + 2.0*backgroundNd[0,:]*backgroundNd[0,:]*modelNd[0,:]
			#mu = Vs*Vs*rho => dmu/drho = Vs*Vs, dmu/dVs = 2*Vs*rho
			dataNd[2,:] += backgroundNd[1,:]*backgroundNd[1,:]*modelNd[2,:] + 2.0*backgroundNd[1,:]*backgroundNd[2,:]*modelNd[1,:]
		else:
			raise ValueError("ERROR! Unsupported conv_type (current value: %s). Please redefine it!"%(self.conv_type))
		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		#Getting pointers to input arrays
		modelNd = model.getNdArray()
		backgroundNd = self.background.getNdArray()
		dataNd = data.getNdArray()

		if(not add): model.zero()
		if(self.conv_type == 1 or self.conv_type == 2):
			#Vp
			modelNd[0,:] += 2.0*backgroundNd[0,:]*backgroundNd[0,:]*dataNd[1,:]
			#Vs
			modelNd[1,:] += 2.0*backgroundNd[1,:]*backgroundNd[2,:]*dataNd[2,:] - 4.0*backgroundNd[1,:]*backgroundNd[1,:]*dataNd[1,:]
			#Rho
			modelNd[2,:] += dataNd[0,:] + (backgroundNd[0,:]*backgroundNd[0,:]-2.0*backgroundNd[1,:]*backgroundNd[1,:])*dataNd[1,:] + backgroundNd[1,:]*backgroundNd[1,:]*dataNd[2,:]
		else:
			raise ValueError("ERROR! Unsupported conv_type (current value: %s). Please redefine it!"%(self.conv_type))
		return

	def setBackground(self,background):
		"""
		   Function to set the background elastic parameters
		"""
		self.background.copy(background)
		if(self.conv_type == 2):
			self.background.getNdArray()[:] *= 1e3
		return
