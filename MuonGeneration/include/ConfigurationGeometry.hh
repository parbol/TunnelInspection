//------------------------------------------------------------//
// |__   __/ __ \|  \/  | |  | | |    / ____|   /\   |  __ \  //
//    | | | |  | | \  / | |  | | |   | |  __   /  \  | |  | | //
//    | | | |  | | |\/| | |  | | |   | | |_ | / /\ \ | |  | | //
//    | | | |__| | |  | | |__| | |___| |__| |/ ____ \| |__| | //
//    |_|  \____/|_|  |_|\____/|______\_____/_/    \_\_____/  //
//------------------------------------------------------------//
// ConfigurationGeometry class:                               //                                                           
//                                                            //
// Parses json files with the configuration of the detectors. //
//                                                            //
//------------------------------------------------------------//

#ifndef ConfigurationGeometry_h
#define ConfigurationGeometry_h 1

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "assert.h"

#include "globals.hh"
#include "Detector.hh"
#include "Layer.hh"

class ConfigurationGeometry {

public:

    ConfigurationGeometry(G4String);
    
    bool isGood();

    // Information about the GEANT4 universe 
    G4double getSizeX();
    G4double getSizeY();
    G4double getSizeZ();
   
    // Detector information
    Detector *getDetector(G4int);
    G4int getNDetectors();

    G4double getZCeiling();
    G4double getXPlanePos();
    G4double getYPlanePos();
    G4double getZPlanePos();
    G4double getXPlaneSize();
    G4double getYPlaneSize();
    G4double getRockSizeX();
    G4double getRockSizeY();
    G4double getRockSizeZ();
    G4double getTunnelInner();
    G4double getTunnelOuter();
    G4double getTunnelSizeZ();

    //Creating the geometry
    void createG4objects(G4LogicalVolume *, 
                         std::map<G4String, G4Material*> &,
                         G4SDManager *);
    // Detector collections
    void registerCollection(G4String);
    std::vector <G4String> collections;
    
    void Print();

private:
    G4double uniSizeX, uniSizeY, uniSizeZ;
    G4double ZCeiling;
    G4double xPlanePos, yPlanePos, zPlanePos;
    G4double xPlaneSize, yPlaneSize, zPlaneSize;
    G4double rockSizeX, rockSizeY, rockSizeZ;
    G4double tunnelInner, tunnelOuter, tunnelSizeZ;
    std::vector <Detector *> detectors;
    bool goodGeometry;
    
};



#endif

