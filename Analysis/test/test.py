import numpy as np
from Analysis.src.DatasetSelector import DatasetSelector
import pandas as pd
import matplotlib.pyplot as plt


def drawProjection(dataSky, dataTunnel, Z):

    lsky = (Z-dataSky[:,2])/dataSky[:,5]
    xsky = dataSky[:,0] + lsky * dataSky[:,3]
    ysky = dataSky[:,1] + lsky * dataSky[:,4]
    
    ltunnel = (Z-dataTunnel[:,2])/dataTunnel[:,5]
    xtunnel = dataTunnel[:,0] + ltunnel * dataTunnel[:,3]
    ytunnel = dataTunnel[:,1] + ltunnel * dataTunnel[:,4]

    

    fig,ax = plt.subplots(1, 3, figsize = (16, 8))
    ax[0].hist2d(xsky, ysky, bins=[100,100], range=[[-2000, 2000],[-2000, 2000]])
    ax[1].hist2d(xtunnel, ytunnel, bins=[100,100], range=[[-2000, 2000],[-2000, 2000]])

    h1, xedges, yedges = np.histogram2d(xtunnel, ytunnel, bins=[100,100], range=[[-2000, 2000],[-2000, 2000]])
    h2, xedges, yedges = np.histogram2d(xsky, ysky, bins=[100,100], range=[[-2000, 2000],[-2000, 2000]])
    h = np.divide(h1, h2, where=(h2 != 0))
    pc = ax[2].pcolorfast(xedges, yedges, h.T)


if __name__=='__main__':

    openskyFile = '/home/pablo/Documentos/softwareProjects/TunnelInspection/MuonGeneration/datasets/opensky_30p3Mv2.h5'
    tunnelFile = '/home/pablo/Documentos/softwareProjects/TunnelInspection/MuonGeneration/datasets/tunnel_30p8Mv2.h5'

    openskyData = pd.read_hdf(openskyFile)
    tunnelData = pd.read_hdf(tunnelFile)

    openskyfull = openskyData.to_numpy()
    tunnelfull = tunnelData.to_numpy()
    
    dsel = DatasetSelector(((-50, 50),(-50,50)))  
    opensky = dsel.get(openskyfull)
    tunnel = dsel.get(tunnelfull)

    drawProjection(opensky, tunnel, 1000)
    plt.savefig('fig.png')

