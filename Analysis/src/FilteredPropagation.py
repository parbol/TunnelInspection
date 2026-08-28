from Analysis.src.DatasetSelector import DatasetSelector
from Analysis.src.ActiveVolume import ActiveVolume
import numpy as np
import sys
import logging
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')



class FilteredPropagation:

    def __init__(self, center, size, Nvoxels, tmap, partitionScheme, datasets):

        logging.info('----------------- Initializing FilteredPropagation algorithm -----------------')
        logging.info('Active Volume information:')
        logging.info(f'    Center: ({center[0]} cm, {center[1]} cm, {center[2]} cm)')
        logging.info(f'    Size: ({size[0]} cm, {size[1]} cm, {size[2]} cm')
        logging.info(f'    Nvoxels: ({Nvoxels[0]}, {Nvoxels[1]}, {Nvoxels[2]})')
        self.active = ActiveVolume(center, size, Nvoxels)
        
        logging.info('Transmission map information:')
        logging.info(f'    Phi range: ({tmap[0]} to {tmap[1]}) and {tmap[2]} bins')
        logging.info(f'    Theta range: ({tmap[3]} to {tmap[4]}) and {tmap[5]} bins')

        logging.info('Partition scheme information:')
        logging.info(f'    Number of chambers: {len(partitionScheme)}')
        for i, p in enumerate(partitionScheme):
            logging.info(f'    Chamber {i}: xmin: {p[0]}, xmax: {p[1]}, Nx: {p[2]}, ymin: {p[3]}, ymax: {p[4]}, Ny: {p[5]}, detzsize: {p[6]}')

        logging.info('Data set information:')
        logging.info(f'    Opensky dataset has {datasets[0].shape[0]} events')
        logging.info(f'    Object dataset has {datasets[1].shape[0]} events')
        logging.info('------------------------------------------------------------------------------')
 
        #Partition definition
        self.partitions = []
        self.hs = []
        self.xedges = []
        self.yedges = []
        for data in datasets:
            chamberpartitiondataset = []
            for p in partitionScheme:
                xdetmin = p[0]
                xdetmax = p[1]
                xdetN = p[2]
                stepx = (xdetmax-xdetmin)/xdetN
                ydetmin = p[3]
                ydetmax = p[4]
                ydetN = p[5]
                detzsize = p[6]
                stepy = (ydetmax-ydetmin)/ydetN
                logging.info(f'Starting with detector with xmin:{xdetmin}, xmax:{xdetmax}, ymin:{ydetmin}, ymax:{ydetmax} and Nx:{xdetN}, Ny:{ydetN} partitions')
                partitiondataset = []
                for jx in range(xdetN):
                    for jy in range(ydetN):
                        xmin = xdetmin + jx * stepx
                        xmax = xdetmin + (jx + 1) * stepx
                        ymin = ydetmin + jy * stepx
                        ymax = ydetmin + (jy + 1) * stepx            
                        dsel = DatasetSelector([[xmin, xmax],[ymin, ymax]], detzsize)
                        selection = dsel.get(data)
                        logging.info(f'    Partition with xmin:{xmin}, xmax:{xmax}, ymin:{ymin}, ymax:{ymax}, number of events: {selection.shape[0]}')
                        partitiondataset.append(selection)
                chamberpartitiondataset.append(partitiondataset)
            self.partitions.append(chamberpartitiondataset)
        

        #Estimation of the tranmission map for each chamber and partition
        for chamber, c in enumerate(self.partitions[0]):
            hchamber = []
            xedgeschamber = []
            yedgeschamber = []
            for partition, p in enumerate(c):
                phiOpensky = np.atan2(self.partitions[0][chamber][partition][:,4], self.partitions[0][chamber][partition][:,3])
                phiObject = np.atan2(self.partitions[1][chamber][partition][:,4], self.partitions[1][chamber][partition][:,3])
                thetaOpensky = np.atan2(np.sqrt(self.partitions[0][chamber][partition][:,3]**2+self.partitions[0][chamber][partition][:,4]**2), self.partitions[0][chamber][partition][:,5])
                thetaObject = np.atan2(np.sqrt(self.partitions[1][chamber][partition][:,3]**2+self.partitions[1][chamber][partition][:,4]**2), self.partitions[1][chamber][partition][:,5])
                hopen, xedgesopen, yedgesopen = np.histogram2d(phiOpensky, thetaOpensky, bins=[tmap[2],tmap[5]], range=[[tmap[0], tmap[1]],[tmap[3], tmap[4]]])
                hobject, xedgesobject, yedgesobject = np.histogram2d(phiObject, thetaObject, bins=[tmap[2],tmap[5]], range=[[tmap[0], tmap[1]],[tmap[3], tmap[4]]])
                h = np.divide(hobject, hopen, where=(hopen!= 0), out=None)
                hchamber.append(h)
                xedgeschamber.append(xedgesopen)
                yedgeschamber.append(yedgesopen)
            self.hs.append(hchamber)
            self.xedges.append(xedgeschamber)
            self.yedges.append(yedgeschamber)
      

    
    def getTransmission(self, det, partition, ray):

        phi = np.atan2(ray[4],ray[3])
        theta = np.atan2(np.sqrt(ray[3]**2+ray[4]**2), ray[5])
        #logging.info(f'Accesing transmission map for detector {det} and partition {partition} for ray with phi {phi}, theta {theta}')
        xmin = self.xedges[det][partition]
        stepx = xmin[1]-xmin[0]
        ymin = self.yedges[det][partition]
        stepy = ymin[1]-ymin[0]
        indexX = int(np.floor((phi-xmin)/stepx)[0])
        indexY = int(np.floor((theta-ymin)/stepy)[0])
        t = self.hs[det][partition][indexX, indexY]
        #logging.info(f'Indexes for transmission map: Phi {indexX}, Theta {indexY}. Transmission value: {t}')
        return t


    def run(self):

        logging.info('----------------- Starting algorithm -----------------')
        for det, chamberDataset in enumerate(self.partitions[1]):
            logging.info(f'    Processing chamber {det} with {len(chamberDataset)} partitions')
            for partition, partitionDataset in enumerate(chamberDataset):
                logging.info(f'        Processing partition {partition} with {len(partitionDataset)} rays')
                for ray in partitionDataset:
                    #logging.info(f'Ray info: ({ray[0]} cm, {ray[1]} cm, {ray[2]} cm) + l ({ray[3]}, {ray[4]}, {ray[5]})')
                    n, d = self.active.voxelList(ray)
                    if len(n) == 0:
                        continue
                    #logging.info(f'Ray passing through {len(n)} voxels')               
                    self.active.update(n, d, self.getTransmission(det=det, partition=partition, ray=ray))
            

    def endRun(self):

        for ix in range(self.active.NVoxels[0]):
            for iy in range(self.active.NVoxels[1]):
                for iz in range(self.active.NVoxels[2]):
                    if self.active.voxels[ix][iy][iz].N > 0:
                        self.active.voxels[ix][iy][iz].rho /= self.active.voxels[ix][iy][iz].N
                        if self.active.voxels[ix][iy][iz].rho > 1.0:
                            self.active.voxels[ix][iy][iz].rho = 1.0