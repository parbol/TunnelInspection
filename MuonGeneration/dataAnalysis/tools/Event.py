import json, sys, optparse
import ROOT as r
from array import array

class Event:

    def __init__(self, nEvent, ndet, nlayer):
        self.nEvent = nEvent
        self.ndet = ndet
        self.nlayer = nlayer
        self.det = []
        self.layer = []
        self.Energy = []
        self.localx = []
        self.localy = []
        self.localz = []
        self.x = []
        self.y = []
        self.z = []
        self.localvx = []
        self.localvy = []
        self.localvz = []
        self.vx = []
        self.vy = []
        self.vz = []

    def add(self, det_, layer_, Energy_, x_, y_, z_, vx_, vy_, vz_, localx_, localy_, localz_, localvx_, localvy_, localvz_):
        
        self.det.append(det_)
        self.layer.append(layer_)
        self.Energy.append(Energy_)
        self.localx.append(localx_)
        self.localy.append(localy_)
        self.localz.append(localz_)
        self.localvx.append(localvx_)
        self.localvy.append(localvy_)
        self.localvz.append(localvz_)
        self.x.append(x_)
        self.y.append(y_)
        self.z.append(z_)
        self.vx.append(vx_)
        self.vy.append(vy_)
        self.vz.append(vz_)

    def nHits(self):

        return len(self.det)
    
    def validEvent(self):

        if self.nHits() < self.nlayer*self.ndet:
            return False
        
        lista = [i for i in range(0, self.ndet*self.nlayer)]
        for i in range(0, self.nHits()):
            id = self.det[i] * self.nlayer + self.layer[i]
            if id in lista:
                lista.remove(id)
        
        if len(lista) != 0:
            return False
        return True
    
  
