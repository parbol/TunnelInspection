#ifndef PrimaryGeneratorAction_h
#define PrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ThreeVector.hh"
#include "G4DataVector.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleGun.hh"
#include "Randomize.hh"
#include "globals.hh"
#include "ConfigurationGeometry.hh"
#include "vector"
#include "RNGWrapper.hh"
#include "PrimaryGeneratorMessenger.hh"
#include "EcoMug.h"

class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    PrimaryGeneratorAction(ConfigurationGeometry *, G4long, G4double);
    ~PrimaryGeneratorAction();

public:
    void GeneratePrimaries(G4Event* anEvent);
    G4double timeSimulated;
    std::array<G4double, 6> getProjection(std::array<G4double, 3>, G4double, G4double);
private:
    EcoMug *gen;
    G4ParticleTable* particleTable;
    G4ParticleGun* particleGun;
    G4ParticleDefinition *fMuonMinus;
    G4ParticleDefinition *fMuonPlus;
    PrimaryGeneratorMessenger* gunMessenger;
    ConfigurationGeometry *myGeom;
    G4long randomSeed;
    G4double pt;
    G4double zceiling;
};

#endif
