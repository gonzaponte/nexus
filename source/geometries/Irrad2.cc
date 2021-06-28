// ----------------------------------------------------------------------------
// nexus | Irrad2.cc
//
// Co irrad at WIS
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "Irrad2.h"
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

Irrad2::Irrad2():
  visibility_    (false)
, source_length_ ( 15   * mm)
, source_diam_   ( 2    * mm)
, capsule_diam_  (6.35  * mm)
, gap_to_crystal_(50    * mm)
, crystal_size_  ( 5    * mm)
, crystal_type_  ("G4_LITHIUM_FLUORIDE")
{
  msg_ = new G4GenericMessenger(this, "/Geometry/Irrad2/",
                                "Control commands of the Co irrad geometry.");
  msg_->DeclareProperty("visibility", visibility_,
                        "Visibility of this geometry");

  msg_->DeclareProperty("source_length" , source_length_ , "");
  msg_->DeclareProperty("source_diam"   , source_diam_   , "");
  msg_->DeclareProperty("capsule_diam"  , capsule_diam_  , "");
  msg_->DeclareProperty("gap_to_crystal", gap_to_crystal_, "");
  msg_->DeclareProperty("crystal_size"  , crystal_size_  , "");
  msg_->DeclareProperty("crystal_type"  , crystal_type_  , "");

}

void Irrad2::Construct()
{
  G4Tubs*          lab_solid = new G4Tubs("lab", 0., 1*m, 1*m, 0., twopi);
  G4Material*      vacuum    = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4LogicalVolume* lab_logic = new G4LogicalVolume(lab_solid, vacuum, "lab");
  new G4PVPlacement(nullptr, G4ThreeVector{}, lab_logic, "lab", nullptr, false, 0, false);

  G4Tubs*          source_solid = new G4Tubs("source", 0., source_diam_/2, source_length_/2., 0., twopi);
  G4Material*      source_mat   = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
  G4LogicalVolume* source_logic = new G4LogicalVolume(source_solid, source_mat, "source");
  new G4PVPlacement(nullptr, G4ThreeVector{}, source_logic, "source", lab_logic, false, 0, false);

  G4Tubs*          capsule_solid = new G4Tubs("ss_shield", source_diam_/2., capsule_diam_/2, source_length_/2., 0., twopi);
  G4Material*      ss            = G4NistManager::Instance()->FindOrBuildMaterial("G4_STAINLESS-STEEL");
  G4LogicalVolume* capsule_logic = new G4LogicalVolume(capsule_solid, ss, "ss_shield");
  new G4PVPlacement(nullptr, G4ThreeVector{}, capsule_logic, "capsule", lab_logic, false, 0, false);

  G4Box*           crystal_solid = new G4Box("crystal", crystal_size_/2, crystal_size_/2, crystal_size_/2);
  G4Material*      crystal_mat   = G4NistManager::Instance()->FindOrBuildMaterial(crystal_type_);
  G4LogicalVolume* crystal_logic = new G4LogicalVolume(crystal_solid, crystal_mat, "crystal");
  new G4PVPlacement(nullptr, G4ThreeVector{gap_to_crystal_, 0, 0}, crystal_logic, "crystal", lab_logic, false, 0, false);


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

    G4VisAttributes grey = nexus::LightGrey();
    grey.SetForceSolid(true);
    capsule_logic->SetVisAttributes(grey);
  }
}

G4ThreeVector Irrad2::GenerateVertex(const G4String& region) const{
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
