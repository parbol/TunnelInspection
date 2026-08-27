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
            x.append(self.Lmin[0] + ix * self.step[0])
        self.x = np.asarray(x)
        y = [] 
        for iy in range(NVoxels[1]):
            y.append(self.Lmin[1] + iy * self.step[1])
        self.y = np.asarray(y)
        z = [] 
        for iz in range(NVoxels[2]):
            z.append(self.Lmin[2] + iz * self.step[2])
        self.z = np.asarray(z)
        #################
        for ix in range(NVoxels[0]):
            voxelsYZ = []
            for iy in range(NVoxels[1]):
                voxelsZ = []
                for iz in range(NVoxels[2]):
                    disp = np.asarray([ix * self.step[0], iy * self.step[1], iz * self.step[2]])
                    center = self.Lmin + disp
                    voxel = Voxel(center, self.step)
                    voxelsZ.append(voxel)
                voxelsYZ.append(voxelsZ)
            self.voxels.append(voxelsYZ)    


    def update(self, n, l, t):
        
        for i in range(len(n)):
            ix = int(n[i][0])
            iy = int(n[i][1])
            iz = int(n[i][2])
            self.voxels[ix][iy][iz].N += 1
            #self.voxels[ix][iy][iz].Lrho += l[i]
            self.voxels[ix][iy][iz].rho += t


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
        l = np.reshape(l, ((len(l),1)))
        xi = x0 + np.multiply(l+epsilon,vx)
        yi = y0 + np.multiply(l+epsilon,vy)
        zi = z0 + np.multiply(l+epsilon,vz)
        ri = np.concatenate((xi, yi, zi), axis=1)
        ri = ri[(ri[:,0] >= self.Lmin[0]) & (ri[:,0] < self.Lmax[0]+self.step[0]) & 
                (ri[:,1] >= self.Lmin[1]) & (ri[:,1] < self.Lmax[1]+self.step[1]) & 
                (ri[:,2] >= self.Lmin[2]) & (ri[:,2] < self.Lmax[2]+self.step[2])]
        nx = np.floor((ri[:,0] - self.Lmin[0])/self.step[0])
        ny = np.floor((ri[:,1] - self.Lmin[1])/self.step[1])
        nz = np.floor((ri[:,2] - self.Lmin[2])/self.step[2])
        nx = np.reshape(nx, ((len(nx),1)))
        ny = np.reshape(ny, ((len(ny),1)))
        nz = np.reshape(nz, ((len(nz),1)))
        n = np.concatenate((nx, ny, nz), axis=1)
        ri1 = np.copy(ri)
        ri2 = np.copy(ri)
        ri1 = np.delete(ri1, (len(ri)-1), axis=0)
        ri2 = np.delete(ri2, (0), axis=0)
        distance = np.sqrt(np.sum((ri2 - ri1)**2, axis=1))

        return n, distance
    
