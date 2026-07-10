import json, sys, optparse
import ROOT as r
from array import array
from tools.Event import Event
from tools.EventLoader import EventLoader

if __name__ == '__main__':

    parser = optparse.OptionParser(usage='usage: %prog [options] path', version='%prog 1.0')
    parser.add_option('-i', '--input', action='store', type='string', dest='inputFile', default='input.root', help='Input ROOT File')
    parser.add_option('-c', '--conf', action='store', type='string', dest='configurationFile', default='conf.json', help='Configuration file')
    parser.add_option('-o', '--output', action='store', type='string', dest='outputFile', default='output.root', help='Output ROOT file')
    (opts, args) = parser.parse_args()
 
    loader = EventLoader(opts.inputFile, opts.configurationFile)
    events = loader.loadEvents()

    print('Number of events', len(events))


    try:
        output = r.TFile(opts.outputFile, 'RECREATE')
    except:
        print('Cannot open output file')
        sys.exit()


    t = r.TTree('events', 'events')
    nevent = array('i', [0])
    x = array('f', [0])
    y = array('f', [0])
    z = array('f', [0])
    vx = array('f', [0])
    vy = array('f', [0])
    vz = array('f', [0])
    energy = array('f', [0])

    
    t.Branch('nevent', 'nevent', 'nevent/I')
    t.Branch('x', x, 'x/F')
    t.Branch('y', y, 'y/F')
    t.Branch('z', z, 'z/F')
    t.Branch('vx', vx, 'vx/F')
    t.Branch('vy', vy, 'vy/F')
    t.Branch('vz', vz, 'vz/F')
    t.Branch('energy', energy, 'energy/F')

   
    for ev in events:
        if not ev.validEvent():
            continue    
        insertEvent = False
        for i in range(0, len(ev.x)):
            if ev.layer[i] == 0 and ev.det[i] == 0:
                nevent[0] = ev.nEvent
                x[0] = ev.x[i]
                y[0] = ev.y[i]
                z[0] = ev.z[i]
                vx[0] = ev.vx[i]
                vy[0] = ev.vy[i]
                vz[0] = ev.vz[i]
                energy[0] = ev.Energy[i]
                insertEvent = True
                break
        if insertEvent:   
            t.Fill()

    output.Write()
    output.Close()
