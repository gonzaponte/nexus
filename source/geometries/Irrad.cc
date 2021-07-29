// ----------------------------------------------------------------------------
// nexus | Irrad.cc
//
// Co irrad at WIS
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "Irrad.h"
#include "IonizationSD.h"
#include "Visibilities.h"

#include <G4GenericMessenger.hh>
#include <G4Box.hh>
#include <G4Tubs.hh>
#include <G4Material.hh>
#include <G4NistManager.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4UserLimits.hh>
#include <G4SDManager.hh>
#include <G4VisAttributes.hh>
#include <G4ThreeVector.hh>
#include <Randomize.hh>

namespace nexus {

using namespace CLHEP;

Irrad::Irrad():
  visibility_    (false)
, source_length_ (10.00 * mm)
, source_diam_   ( 3.00 * mm)
, ss_diam_       ( 6.35 * mm)
, gap_to_crystal_(15    * mm)
, crystal_size_  ( 5    * mm)
, crystal_type_  ("G4_LITHIUM_FLUORIDE")
{
  msg_ = new G4GenericMessenger(this, "/Geometry/Irrad/",
                                "Control commands of the Co irrad geometry.");
  msg_->DeclareProperty("visibility", visibility_,
                        "Visibility of this geometry");

  msg_->DeclareProperty("source_length" , source_length_ , "");
  msg_->DeclareProperty("source_diam"   , source_diam_   , "");
  msg_->DeclareProperty("ss_diam"       , ss_diam_       , "");
  msg_->DeclareProperty("gap_to_crystal", gap_to_crystal_, "");
  msg_->DeclareProperty("crystal_size"  , crystal_size_  , "");
  msg_->DeclareProperty("crystal_type"  , crystal_type_  , "");

}

void Irrad::Construct()
{
  G4Tubs*          lab_solid = new G4Tubs("lab", 0., 1*m, 1*m, 0., twopi);
  G4Material*      vacuum    = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4LogicalVolume* lab_logic = new G4LogicalVolume(lab_solid, vacuum, "lab");
  new G4PVPlacement(nullptr, G4ThreeVector{}, lab_logic, "lab", nullptr, false, 0, false);

  G4Tubs*          source_solid = new G4Tubs("source", 0., source_diam_/2, source_length_/2., 0., twopi);
  G4Material*      source_mat   = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  G4LogicalVolume* source_logic = new G4LogicalVolume(source_solid, source_mat, "source");
  new G4PVPlacement(nullptr, G4ThreeVector{}, source_logic, "source", lab_logic, false, 0, false);

  G4Tubs*          ss_shield_solid = new G4Tubs("ss_shield", source_diam_/2, ss_diam_/2, source_length_/2, 0., twopi);
  G4Material*      ss              = G4NistManager::Instance()->FindOrBuildMaterial("G4_STAINLESS-STEEL");
  G4LogicalVolume* ss_shield_logic = new G4LogicalVolume(ss_shield_solid, ss, "ss_shield");
  new G4PVPlacement(nullptr, G4ThreeVector{}, ss_shield_logic, "ss_shield", lab_logic, false, 0, false);

  G4double         lead_length       = 2 * gap_to_crystal_;
  G4Tubs*          lead_shield_solid = new G4Tubs("lead_shield", ss_diam_/2, 5*cm, lead_length/2, 0., twopi);
  G4Material*      lead              = G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb");
  G4LogicalVolume* lead_shield_logic = new G4LogicalVolume(lead_shield_solid, lead, "lead_shield");
  new G4PVPlacement(nullptr, G4ThreeVector{}, lead_shield_logic, "lead", lab_logic, false, 0, false);

  G4double         c_height      = source_length_/2 + gap_to_crystal_ + crystal_size_/2;
  G4Box*           crystal_solid = new G4Box("crystal", crystal_size_/2, crystal_size_/2, crystal_size_/2);
  G4Material*      crystal_mat   = G4NistManager::Instance()->FindOrBuildMaterial(crystal_type_);
  G4LogicalVolume* crystal_logic = new G4LogicalVolume(crystal_solid, crystal_mat, "crystal");
  new G4PVPlacement(nullptr, G4ThreeVector{0, 0, c_height}, crystal_logic, "crystal", lab_logic, false, 0, false);


  this->SetLogicalVolume(lab_logic);

  // crystal is defined as an ionization sensitive volume.
  G4SDManager*  sdmgr   = G4SDManager::GetSDMpointer();
  G4String      detname = "/CC/crystal" ;
  IonizationSD* ionisd  = new IonizationSD(detname);
  ionisd       ->IncludeInTotalEnergyDeposit(true);
  sdmgr        ->AddNewDetector             (ionisd);
  crystal_logic->SetSensitiveDetector       (ionisd);

   if (visibility_) {
    G4VisAttributes yellow = nexus::Yellow();
    yellow.SetForceSolid(true);
    crystal_logic->SetVisAttributes(yellow);

    G4VisAttributes red = nexus::Red();
    red.SetForceSolid(true);
    source_logic->SetVisAttributes(red);

    G4VisAttributes grey = nexus::DarkGrey();
    grey.SetForceSolid(true);
    lead_shield_logic->SetVisAttributes(grey);

    G4VisAttributes white = nexus::White();
    white.SetForceSolid(true);
    ss_shield_logic->SetVisAttributes(white);

  }
}

G4ThreeVector Irrad::GenerateVertex(const G4String& region) const{
  G4double x=0, y=0, z=0;
  if (region == "source") {
    G4double r = 100*m;
    while (r < source_diam_/2){
      x = G4UniformRand() * source_diam_ - source_diam_/2;
      y = G4UniformRand() * source_diam_ - source_diam_/2;
      r = std::sqrt(x*x + y*y);
    }
    z = G4UniformRand() * source_length_ - source_length_/2;
  }
  else {
    G4cerr << "Invalid region: " << region << G4endl;
  }
  return G4ThreeVector{x, y, z};
}


} // namespace nexus
