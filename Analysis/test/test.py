import numpy as np
from Analysis.src.DatasetSelector import DatasetSelector
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def drawProjection(dataSky, dataTunnel, Z):
        
    fig,ax = plt.subplots(len(dataSky), 3, figsize = (16, 16))
    maxx = 2000
    maxy = 2000
    minx = -2000
    miny = -2000
    binsx = 50
    binsy = 50
    for i in range(len(dataSky)):
        lsky = (Z-dataSky[i][:,2])/dataSky[i][:,5]
        xsky = dataSky[i][:,0] + lsky * dataSky[i][:,3]
        ysky = dataSky[i][:,1] + lsky * dataSky[i][:,4]
    
        ltunnel = (Z-dataTunnel[i][:,2])/dataTunnel[i][:,5]
        xtunnel = dataTunnel[i][:,0] + ltunnel * dataTunnel[i][:,3]
        ytunnel = dataTunnel[i][:,1] + ltunnel * dataTunnel[i][:,4]
        ax[i][0].hist2d(xsky, ysky, bins=[binsx,binsy], range=[[minx, maxx],[miny, maxy]])
        ax[i][1].hist2d(xtunnel, ytunnel, bins=[binsx,binsy], range=[[minx, maxx],[miny, maxy]])
        h1, xedges, yedges = np.histogram2d(xtunnel, ytunnel, bins=[binsx,binsy], range=[[minx, maxx],[miny, maxy]])
        h2, xedges, yedges = np.histogram2d(xsky, ysky, bins=[binsx,binsy], range=[[minx, maxx],[miny, maxy]])
        h = np.divide(h1, h2, where=(h2!= 0), out=None)
        pc = ax[i][2].pcolorfast(xedges, yedges, h.T, vmin=0, vmax=0.5)


if __name__=='__main__':

    openskyFile = '/home/pablo/Documentos/softwareProjects/TunnelInspection/MuonGeneration/datasets/opensky_30p3Mv2.h5'
    tunnelFile = '/home/pablo/Documentos/softwareProjects/TunnelInspection/MuonGeneration/datasets/tunnel_30p8Mv2.h5'

    openskyData = pd.read_hdf(openskyFile)
    tunnelData = pd.read_hdf(tunnelFile)

    openskyfull = openskyData.to_numpy()
    tunnelfull = tunnelData.to_numpy()
    
    dsel1 = DatasetSelector(limits=((-500, 500),(-500,500)), detsize=40.0)  
    dsel2 = DatasetSelector(limits=((-125, -25),(-125,-25)), detsize=40.0)  
    opensky = []
    tunnel = []
    #opensky1 = dsel1.get(openskyfull)
    #tunnel1 = dsel1.get(tunnelfull)
    #opensky2 = dsel2.get(openskyfull)
    #tunnel2 = dsel2.get(tunnelfull)
    opensky.append(openskyfull)
    opensky.append(openskyfull)
    tunnel.append(tunnelfull)
    tunnel.append(tunnelfull)
    
    fig,ax = plt.subplots(2, 1, figsize = (16, 16))
    ax[0].hist(openskyfull[:,0], bins=100)
    ax[1].hist(openskyfull[:,1], bins=100)
    #drawProjection(opensky, tunnel, -729)
    plt.savefig('fig.png')

