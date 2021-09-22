// ----------------------------------------------------------------------------
// nexus | B10Generator.cc
//
// This class is the primary generator for the decay chain
// of a source containing B10.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "B10Generator.h"

#include "DetectorConstruction.h"
#include "GeometryBase.h"
#include "FactoryBase.h"

#include <G4Event.hh>
#include <G4GenericMessenger.hh>
#include <G4RunManager.hh>
#include <G4ParticleTable.hh>
#include <G4RandomDirection.hh>
#include <Randomize.hh>

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

using namespace nexus;

REGISTER_CLASS(B10Generator, G4VPrimaryGenerator)

namespace nexus {

  using namespace CLHEP;

  B10Generator::B10Generator() :
  msg_ (nullptr),
  geom_(nullptr),
  simulate_xrays_(false),
  region_("")
  {
     msg_ = new G4GenericMessenger(this, "/Generator/B10Generator/",
                                         "Control commands of B10 generator.");

     msg_->DeclareProperty("region", region_,
			   "Set the region of the geometry where the vertex will be generated.");

     msg_->DeclareProperty("simulate_xrays", simulate_xrays_,
			   "Whether to simulate Am241 60 keV gamma.");

     // Set particle type searching in particle table by name
    gammadef_ = G4ParticleTable::GetParticleTable()->FindParticle("gamma");

    DetectorConstruction* detconst = (DetectorConstruction*)
      G4RunManager::GetRunManager()->GetUserDetectorConstruction();
    geom_ = detconst->GetGeometry();
  }

  B10Generator::~B10Generator()
  {
  }

  void B10Generator::GeneratePrimaryVertex(G4Event* evt)
  {
    std::vector<double> energies;

    double p = G4UniformRand();

         if (p<0.229454956) { // 3854->0
        energies.push_back(3854 * keV);
    }
    else if (p<0.345947472) { // 3854->3685->0
      energies.push_back( 169 * keV);
      energies.push_back(3685 * keV);
    }
    else if (p<0.353007625) { // 3854->3090->0
      energies.push_back( 764 * keV);
      energies.push_back(3090 * keV);
    }
    else if (p<0.901934482) { // 3685->0
      // Doppler broadening. Not calculated, estimated from spectrum
      energies.push_back(G4RandGauss::shoot(3685 * keV, 25 * keV));
    }
    else            { // 3090->0
      // Doppler broadening. Not calculated, completely made up
      energies.push_back(G4RandGauss::shoot(3090 * keV, 100 * keV));
    }

    G4ThreeVector position = geom_->GenerateVertex(region_);
    G4double      time     = 0;

    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);

    for (double e : energies) {
      G4PrimaryParticle* gamma = new G4PrimaryParticle(gammadef_);

      gamma->SetMomentumDirection(G4RandomDirection());
      gamma->SetTotalEnergy(e);
      gamma->SetPolarization(0.,0.,0.);
      gamma->SetProperTime(time);
      vertex->SetPrimary(gamma);
    }

  if (simulate_xrays_ && G4UniformRand() < 0.3636) {
    G4PrimaryParticle* gamma = new G4PrimaryParticle(gammadef_);

    gamma->SetMomentumDirection(G4RandomDirection());
    gamma->SetTotalEnergy(59.54 * keV);
    gamma->SetPolarization(0.,0.,0.);
    gamma->SetProperTime(time);
    vertex->SetPrimary(gamma);
  }

   evt->AddPrimaryVertex(vertex);
  }

} // Namespace nexus
