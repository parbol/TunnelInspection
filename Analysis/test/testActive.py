from Analysis.src.ActiveVolume import ActiveVolume
import numpy as np





if __name__=='__main__':

    center = [0.0, 0.0, 0.0]
    size = [100.0, 100.0, 100.0]
    nvoxel = [10, 10, 10]
    active = ActiveVolume(center, size, nvoxel)

    ray = np.asarray([0.05,0,0,0,0,1])
    
    active.voxelList(ray)

    

   
