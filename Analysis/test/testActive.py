from Analysis.src.ActiveVolume import ActiveVolume
import numpy as np





if __name__=='__main__':

    center = [0.0, 0.0, 0.0]
    size = [100.0, 100.0, 100.0]
    nvoxel = [10, 10, 10]
    active = ActiveVolume(center, size, nvoxel)

    rays = np.asarray([[1,2,3,4,5,6], [1.1,2.1,3.1,4.1,5.1,6.1], [1.2,2.2,3.2,4.2,5.2,6.2]])
    
    active.voxelList(rays)

    

   
