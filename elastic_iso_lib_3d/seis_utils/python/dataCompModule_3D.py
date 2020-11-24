#Module containing definition for the elastic data sampling
import pyOperator
import Hypercube, SepVector
#Other modules
from collections import Counter
import numpy as np


class ElasticDatComp_3D(pyOperator.Operator):
    """
       Operator for sampling elastic data components
    """

    def __init__(self,components,domain):
        """
           Constructor for resampling elastic data components
           components = [no default] - string; Comma-separated list of the output/sampled components (e.g., 'vx,vy,vz,sxx,syy,szz,sxz,sxy,syz' or 'p,vx,vz'; the order matters!). Currently, supported: vx,vy,vz,sxx,syy,szz,sxz,sxy,syz,p (i.e., p = 0.5*(sxx+syy+szz))
           domain     = [no default] - vector class; Elastic data directly outputted from the elastic modeling operator (i.e., [time,receivers,component(vx,vz,sxx,szz,sxz),shots])
        """
        #Getting axes from domain vector
        timeAxis = domain.getHyper().getAxis(1)
        recAxis = domain.getHyper().getAxis(2)
        shotAxis = domain.getHyper().getAxis(4)
        #Getting number of output components
        self.comp_list = components.split(",")
        if any(elem > 1 for elem in Counter(self.comp_list).values()):
            raise ValueError("ERROR! A component was provided multiple times! Check your input arguments")
        #Creating component axis for output vector
        compAxis=Hypercube.axis(n=len(self.comp_list),label="Component %s"%(components))
        range = SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,recAxis,compAxis,shotAxis]))
        #Setting domain and range
        self.setDomainRange(domain,range)
        #Making the list lower case
        self.comp_list = [elem.lower() for elem in self.comp_list]
        if not any(["vx" in self.comp_list,"vy" in self.comp_list,"vz" in self.comp_list,"sxx" in self.comp_list,"syy" in self.comp_list,"szz" in self.comp_list,"sxz" in self.comp_list,"sxy" in self.comp_list,"syz" in self.comp_list,"p" in self.comp_list]):
            raise ValueError("ERROR! Provided unknown data components: %s"%(components))
        return


    def forward(self,add,model,data):
        """
           Forward operator: elastic data -> sampled data
        """
        self.checkDomainRange(model,data)
        if(not add): data.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()
        #Checking if vx was requested to be sampled
        if("vx" in self.comp_list):
            idx = self.comp_list.index("vx")
            dataNd[:,idx,:,:] += modelNd[:,0,:,:]
		#Checking if vy was requested to be sampled
        if("vy" in self.comp_list):
            idx = self.comp_list.index("vy")
            dataNd[:,idx,:,:] += modelNd[:,1,:,:]
        #Checking if vz was requested to be sampled
        if("vz" in self.comp_list):
            idx = self.comp_list.index("vz")
            dataNd[:,idx,:,:] += modelNd[:,2,:,:]
        #Checking if sxx (normal stress) was requested to be sampled
        if("sxx" in self.comp_list):
            idx = self.comp_list.index("sxx")
            dataNd[:,idx,:,:] += modelNd[:,3,:,:]
		#Checking if syy (normal stress) was requested to be sampled
        if("syy" in self.comp_list):
            idx = self.comp_list.index("syy")
            dataNd[:,idx,:,:] += modelNd[:,4,:,:]
        #Checking if szz (normal stress) was requested to be sampled
        if("szz" in self.comp_list):
            idx = self.comp_list.index("szz")
            dataNd[:,idx,:,:] += modelNd[:,5,:,:]
        #Checking if sxz (shear stress) was requested to be sampled
        if("sxz" in self.comp_list):
            idx = self.comp_list.index("sxz")
            dataNd[:,idx,:,:] += modelNd[:,6,:,:]
		#Checking if sxy (shear stress) was requested to be sampled
        if("sxy" in self.comp_list):
            idx = self.comp_list.index("sxy")
            dataNd[:,idx,:,:] += modelNd[:,7,:,:]
		#Checking if syz (shear stress) was requested to be sampled
        if("syz" in self.comp_list):
            idx = self.comp_list.index("syz")
            dataNd[:,idx,:,:] += modelNd[:,8,:,:]
        #Checking if pressure was requested to be sampled
        if("p" in self.comp_list):
            idx = self.comp_list.index("p")
            dataNd[:,idx,:,:] += 0.5*(modelNd[:,3,:,:]+modelNd[:,4,:,:]+modelNd[:,5,:,:])
        return

    def adjoint(self,add,model,data):
        """
           Adjoint operator: sampled data -> elastic data
        """
        self.checkDomainRange(model,data)
        if(not add): model.zero()
        modelNd = model.getNdArray()
        dataNd = data.getNdArray()
        #Checking if vx was requested to be sampled
        if("vx" in self.comp_list):
            idx = self.comp_list.index("vx")
            modelNd[:,0,:,:] += dataNd[:,idx,:,:]
		#Checking if vy was requested to be sampled
        if("vy" in self.comp_list):
            idx = self.comp_list.index("vy")
            modelNd[:,1,:,:] += dataNd[:,idx,:,:]
        #Checking if vz was requested to be sampled
        if("vz" in self.comp_list):
            idx = self.comp_list.index("vz")
            modelNd[:,2,:,:] += dataNd[:,idx,:,:]
        #Checking if sxx (normal stress) was requested to be sampled
        if("sxx" in self.comp_list):
            idx = self.comp_list.index("sxx")
            modelNd[:,3,:,:] += dataNd[:,idx,:,:]
		#Checking if syy (normal stress) was requested to be sampled
        if("syy" in self.comp_list):
            idx = self.comp_list.index("syy")
            modelNd[:,4,:,:] += dataNd[:,idx,:,:]
        #Checking if szz (normal stress) was requested to be sampled
        if("szz" in self.comp_list):
            idx = self.comp_list.index("szz")
            modelNd[:,5,:,:] += dataNd[:,idx,:,:]
        #Checking if sxz (shear stress) was requested to be sampled
        if("sxz" in self.comp_list):
            idx = self.comp_list.index("sxz")
            modelNd[:,6,:,:] += dataNd[:,idx,:,:]
		#Checking if sxy (shear stress) was requested to be sampled
        if("sxy" in self.comp_list):
            idx = self.comp_list.index("sxy")
            modelNd[:,7,:,:] += dataNd[:,idx,:,:]
		#Checking if syz (shear stress) was requested to be sampled
        if("syz" in self.comp_list):
            idx = self.comp_list.index("syz")
            modelNd[:,8,:,:] += dataNd[:,idx,:,:]
        #Checking if pressure was requested to be sampled
        if("p" in self.comp_list):
            idx = self.comp_list.index("p")
            modelNd[:,3,:,:] += 0.5*dataNd[:,idx,:,:]
            modelNd[:,4,:,:] += 0.5*dataNd[:,idx,:,:]
			modelNd[:,5,:,:] += 0.5*dataNd[:,idx,:,:]
        return
