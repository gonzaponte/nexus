// ----------------------------------------------------------------------------
// nexus | LineELGenerator.cc
//
// This class is the primary generator for events consisting of
// a single particle. The user must specify via configuration
// parameters the particle type, a kinetic energy interval and, optionally,
// a momentum direction.
// Particle energy is generated with flat random probability
// between E_min and E_max.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "LineELGenerator.h"

#include "DetectorConstruction.h"
#include "GeometryBase.h"
#include "RandomUtils.h"
#include "FactoryBase.h"

#include <G4GenericMessenger.hh>
#include <G4ParticleDefinition.hh>
#include <G4RunManager.hh>
#include <G4ParticleTable.hh>
#include <G4PrimaryVertex.hh>
#include <G4Event.hh>
#include <G4RandomDirection.hh>
#include <Randomize.hh>
#include <G4OpticalPhoton.hh>

#include "CLHEP/Units/SystemOfUnits.h"

using namespace nexus;
using namespace CLHEP;

REGISTER_CLASS(LineELGenerator, G4VPrimaryGenerator)


LineELGenerator::LineELGenerator():
  G4VPrimaryGenerator(),
  msg_(0),
  photon_(0),
  geom_(nullptr),
  energy_min_(0),
  energy_max_(0),
  sigma_x_(0),
  sigma_y_(0),
  costheta_min_(-1.),
  costheta_max_(1.),
  n_photons_(0)
{
  msg_ = new G4GenericMessenger(this, "/Generator/LineEL/",
    "Control commands of Line EL generator.");

  G4GenericMessenger::Command& min_energy =
  msg_->DeclareProperty("min_energy", energy_min_, "Minimum kinetic energy of the particle.");
  min_energy.SetUnitCategory("Energy");
  min_energy.SetParameterName("min_energy", false);
  min_energy.SetRange("min_energy>0.");

  G4GenericMessenger::Command& max_energy =
    msg_->DeclareProperty("max_energy", energy_max_, "Maximum kinetic energy of the particle");
  max_energy.SetUnitCategory("Energy");
  max_energy.SetParameterName("max_energy", false);
  max_energy.SetRange("max_energy>0.");

  G4GenericMessenger::Command& sigma_x =
    msg_->DeclareProperty("sigma_x", sigma_x_, "Diffusion sigma along x axis.");
  sigma_x.SetUnitCategory("Length");
  sigma_x.SetParameterName("sigma_x", false);
  sigma_x.SetRange("sigma_x>=0.");

  G4GenericMessenger::Command& sigma_y =
    msg_->DeclareProperty("sigma_y", sigma_y_, "Diffusion sigma along y axis.");
  sigma_y.SetUnitCategory("Length");
  sigma_y.SetParameterName("sigma_y", false);
  sigma_y.SetRange("sigma_y>=0.");

  msg_->DeclareProperty("region", region_,
                        "Region of the geometry where the vertex will be generated.");

  msg_->DeclareProperty("min_costheta", costheta_min_,
			"Minimum cosTheta for the direction of the particle.");
  msg_->DeclareProperty("max_costheta", costheta_max_,
			"Maximum cosTheta for the direction of the particle.");
  msg_->DeclareProperty("n_photons", n_photons_, "Number of photons per event");

  DetectorConstruction* detconst =
    (DetectorConstruction*) G4RunManager::GetRunManager()->GetUserDetectorConstruction();
  geom_ = detconst->GetGeometry();
}



LineELGenerator::~LineELGenerator()
{
  delete msg_;
}



void LineELGenerator::SetParticleDefinition()
{
  photon_ = G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");

  if (!photon_)
    G4Exception("[LineELGenerator]", "SetParticleDefinition()",
      FatalException, "Could not load optical photon definition");
}


void LineELGenerator::GeneratePrimaryVertex(G4Event* event)
{
  if (!photon_) SetParticleDefinition();

  auto pos0   = geom_->GenerateVertex(region_);
  auto x0     = pos0.x();
  auto y0     = pos0.y();
  auto z0     = geom_ -> GetELzCoord();
  auto t0     = 0.;
  auto dz     = geom_ -> GetELWidth();

  for (int i=0; i<n_photons_; i++) {
    auto x = x0 + G4RandGauss::shoot() * sigma_x_;
    auto y = y0 + G4RandGauss::shoot() * sigma_y_;
    auto z = z0 + G4UniformRand     () * dz;
    auto energy = energy_min_ + G4UniformRand() * (energy_max_ - energy_min_);
    auto p_dir  = RandomDirectionInRange(costheta_min_, costheta_max_, 0, 2*M_PI);
    auto photon = new G4PrimaryParticle(photon_);
         photon -> SetTotalEnergy(energy);
         photon -> SetMomentumDirection(p_dir);
         photon -> SetPolarization(G4RandomDirection());

    auto vertex = new G4PrimaryVertex({x, y, z}, t0);
    vertex->SetPrimary(photon);
    event->AddPrimaryVertex(vertex);
  }
}
