from Analysis.src.Voxel import Voxel
import numpy as np





class ActiveVolume:

    def __init__(self, center, size, NVoxels):
        
        self.center = np.asarray(center)
        self.size = np.asarray(size)
        self.NVoxels = np.asarray(NVoxels, dtype=np.uint32)
        self.Linit = self.center - self.size/2.0
        self.step = self.size/self.NVoxels
        self.voxels = []
        ################
        z = [] 
        for iz in range(NVoxels[2]):
            z.append(self.Linit[2] + iz * self.step[2])
        self.z = np.asarray(z)
        #################
        for ix in range(NVoxels[0]):
            voxelsYZ = []
            for iy in range(NVoxels[1]):
                voxelsZ = []
                for iz in range(NVoxels[2]):
                    disp = np.asarray([ix * self.step[0], iy * self.step[1], iz * self.step[2]])
                    center = self.Linit + disp
                    voxel = Voxel(center, self.step)
                    voxelsZ.append(voxel)
                voxelsYZ.append(voxelsZ)
            self.voxels.append(voxelsYZ)    


    def voxelList(self, rays):
        z = np.tile(self.z, (rays.shape[0], 1))
        z = z.T
        z = np.reshape(z, (z.shape[0]*z.shape[1], 1))
        raysl = np.tile(rays, (self.z.shape[0], 1))  
        x0 = np.asmatrix(raysl[:,0]).T
        y0 = np.asmatrix(raysl[:,1]).T
        z0 = np.asmatrix(raysl[:,2]).T
        vx = np.asmatrix(raysl[:,3]).T
        vy = np.asmatrix(raysl[:,4]).T
        vz = np.asmatrix(raysl[:,5]).T
        l = (z - z0)/vz
        x = x0 + np.multiply(l,vx)
        y = y0 + np.multiply(l,vy)
        z = np.reshape(z, (self.z.shape[0], rays.shape[0])).T
        x = np.reshape(x, (self.z.shape[0], rays.shape[0])).T
        y = np.reshape(y, (self.z.shape[0], rays.shape[0])).T

        

    
