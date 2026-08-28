from Analysis.src.FilteredPropagation import FilteredPropagation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import logging
import napari

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


    
if __name__=='__main__':


    openskyFile = '/home/pablo/Documentos/softwareProjects/TunnelInspection/MuonGeneration/datasets/NewConfiguration/Opensky/1000/outputOpensky_conf2_1-1000_100M.h5'
    tunnelFile = '/home/pablo/Documentos/softwareProjects/TunnelInspection/MuonGeneration/datasets/NewConfiguration/Tunnel/1000/outputTunnel_conf2_1-1000-100M.h5'

    openskyData = pd.read_hdf(openskyFile)
    tunnelData = pd.read_hdf(tunnelFile)
    openskyfull = openskyData.to_numpy()
    tunnelfull = tunnelData.to_numpy()

    #Detasets to be used in the FilteredPropagation
    datasets = [openskyfull, tunnelfull]

    #Configuration of the active volume
    center = [0, 0, 200]
    size = [1200.0, 1200.0, 400.0]
    Nvoxels = [20, 20, 10]
    
    #Configuration of the transmission map
    tmap = [-np.pi, np.pi, 36, 3.0*np.pi/4.0, np.pi, 20]
    
    #Configuration of the partition scheme
    partitionScheme = [[-350, -250, 3, -350, -250, 3, 40],
                       [-50, 50, 3, -50, 50, 3, 40],
                       [-50, 50, 3, -150, -50, 3, 40]]
    
    fProp = FilteredPropagation(center=center, size=size, Nvoxels=Nvoxels, tmap=tmap, partitionScheme=partitionScheme, datasets=datasets)


    fProp.run()
    fProp.endRun()

    fig, ax = plt.subplots(2, int(Nvoxels[2]/2), figsize=(16, 16))
    rho3D = np.zeros((Nvoxels[0], Nvoxels[1], Nvoxels[2]))
    for i in range(int(Nvoxels[2]/2)):
        for j in range(2):
            zindex = i*2+j       
            rho = np.zeros((Nvoxels[0], Nvoxels[1]))
            #for ix in range(Nvoxels[0]):
            #    for iy in range(Nvoxels[1]):
            for ix in range(1, 10):
                for iy in range(1,10):
                    rho[ix, iy] = fProp.active.voxels[ix][iy][zindex].rho
                    rho3D[ix, iy, zindex] = fProp.active.voxels[ix][iy][zindex].rho
            #im = ax[j,i].imshow(rho.T, origin='lower', cmap='inferno', extent=(fProp.active.x[0], fProp.active.x[-1], fProp.active.y[0], fProp.active.y[-1]), aspect='auto', norm=colors.LogNorm(vmin=0.3, vmax=1.0))
            im = ax[j,i].imshow(rho.T, origin='lower', cmap='inferno', extent=(fProp.active.x[0], fProp.active.x[-1], fProp.active.y[0], fProp.active.y[-1]), aspect='auto', vmin=0.3, vmax=1.0)
            print('Rho:', np.max(rho))
            ax[j,i].set_title(f'Z = {fProp.active.z[zindex]:.2f} cm')
            ax[j,i].set_xlabel('X (cm)')
            ax[j,i].set_ylabel('Y (cm)')
   
    
    #plt.show()
    
    # create an empty viewer
    viewer = napari.Viewer()

    # add the xarray
    layer = viewer.add_image(rho3D, name="blobs")

    napari.run()