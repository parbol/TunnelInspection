from Analysis.src.Voxel import Voxel
import numpy as np





class ActiveVolume:

    def __init__(self, center, size, NVoxels):
        
        self.center = np.asarray(center)
        self.size = np.asarray(size)
        self.NVoxels = np.asarray(NVoxels, dtype=np.uint32)
        self.Lmin = self.center - self.size/2.0
        self.Lmax = self.center + self.size/2.0
        self.step = self.size/self.NVoxels
        self.voxels = []
        ################
        x = [] 
        for ix in range(NVoxels[0]):
            x.append(self.Linit[0] + ix * self.step[0])
        self.x = np.asarray(x)
        y = [] 
        for iy in range(NVoxels[1]):
            y.append(self.Linit[1] + iy * self.step[1])
        self.y = np.asarray(y)
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


    def voxelList(self, ray):

        epsilon = 1e-6
        x0 = ray[0]
        y0 = ray[1]
        z0 = ray[2]
        vx = ray[3]
        vy = ray[4]
        vz = ray[5]
        lx = (self.x - x0)/vx 
        ly = (self.y - y0)/vy  
        lz = (self.z - z0)/vz 
        l = np.concatenate((lx,ly,lz))
        l = np.sort(l)
        x = x0 + np.multiply(l,vx)
        y = y0 + np.multiply(l,vy)
        z = z0 + np.multiply(l,vz)
        r = np.concatenate((x, y, z), axis=1)
        r = r[(r[0] > self.Lmin[0]) & (r[0] < self.Lmax[0]) & 
              (r[1] > self.Lmin[1]) & (r[1] < self.Lmax[1]) & 
              (r[2] > self.Lmin[2]) & (r[2] < self.Lmax[2])]
        nx = 
        z = np.reshape(z, (self.z.shape[0], rays.shape[0])).T
        x = np.reshape(x, (self.z.shape[0], rays.shape[0])).T
        y = np.reshape(y, (self.z.shape[0], rays.shape[0])).T

        

    
