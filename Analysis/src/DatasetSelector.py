import numpy as np


class DatasetSelector:

    def __init__(self, limits):

        self.zval = 2.5
        self.xmin = limits[0][0]
        self.xmax = limits[0][1]
        self.ymin = limits[1][0]
        self.ymax = limits[1][1]


    def get(self, data):
        
        preselect = data[(data[:, 0] > self.xmin) & (data[:, 0] < self.xmax) & (data[:,1] > self.ymin) & (data[:,1] < self.ymax)]
        select = preselect[(preselect[:,0] + (self.zval - preselect[:,2])/preselect[:, 5] * preselect[:,3] > self.xmin) &
                           (preselect[:,0] + (self.zval - preselect[:,2])/preselect[:, 5] * preselect[:,3] < self.xmax) &
                           (preselect[:,1] + (self.zval - preselect[:,2])/preselect[:, 5] * preselect[:,4] > self.ymin) &
                           (preselect[:,1] + (self.zval - preselect[:,2])/preselect[:, 5] * preselect[:,4] < self.ymax)]
        return select





