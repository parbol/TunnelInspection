import numpy as np
from Analysis.src.DatasetSelector import DatasetSelector
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def drawProjection(dataSky, dataTunnel, Z):
        
    fig,ax = plt.subplots(len(dataSky), 3, figsize = (16, 16))

    for i in range(len(dataSky)):
        lsky = (Z-dataSky[i][:,2])/dataSky[i][:,5]
        xsky = dataSky[i][:,0] + lsky * dataSky[i][:,3]
        ysky = dataSky[i][:,1] + lsky * dataSky[i][:,4]
    
        ltunnel = (Z-dataTunnel[i][:,2])/dataTunnel[i][:,5]
        xtunnel = dataTunnel[i][:,0] + ltunnel * dataTunnel[i][:,3]
        ytunnel = dataTunnel[i][:,1] + ltunnel * dataTunnel[i][:,4]

        ax[i][0].hist2d(xsky, ysky, bins=[50,50], range=[[-2000, 2000],[-2000, 2000]])
        ax[i][1].hist2d(xtunnel, ytunnel, bins=[50,50], range=[[-2000, 2000],[-2000, 2000]])
        h1, xedges, yedges = np.histogram2d(xtunnel, ytunnel, bins=[50,50], range=[[-2000, 2000],[-2000, 2000]])
        h2, xedges, yedges = np.histogram2d(xsky, ysky, bins=[50,50], range=[[-2000, 2000],[-2000, 2000]])
        h = np.divide(h1, h2, where=(h2!= 0))
        pc = ax[i][2].pcolorfast(xedges, yedges, h.T)


if __name__=='__main__':

    openskyFile = '/home/pablo/Documentos/softwareProjects/TunnelInspection/MuonGeneration/datasets/opensky_30p3Mv2.h5'
    tunnelFile = '/home/pablo/Documentos/softwareProjects/TunnelInspection/MuonGeneration/datasets/tunnel_30p8Mv2.h5'

    openskyData = pd.read_hdf(openskyFile)
    tunnelData = pd.read_hdf(tunnelFile)

    openskyfull = openskyData.to_numpy()
    tunnelfull = tunnelData.to_numpy()
    
    dsel1 = DatasetSelector(limits=((-50, 50),(-50,50)), detsize=40.0)  
    dsel2 = DatasetSelector(limits=((-125, -25),(-125,-25)), detsize=40.0)  
    opensky = []
    tunnel = []
    opensky1 = dsel1.get(openskyfull)
    tunnel1 = dsel1.get(tunnelfull)
    opensky2 = dsel2.get(openskyfull)
    tunnel2 = dsel2.get(tunnelfull)
    opensky.append(opensky2)
    opensky.append(opensky1)
    tunnel.append(tunnel2)
    tunnel.append(tunnel1)

    drawProjection(opensky, tunnel, 1000)
    plt.savefig('fig.png')

