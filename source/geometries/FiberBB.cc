// -----------------------------------------------------------------------------
// nexus | FiberBB.cc
//
//
//
// The NEXT Collaboration
// -----------------------------------------------------------------------------

#include "FiberBB.h"
#include "CylinderPointSampler2020.h"
#include "MaterialsList.h"
#include "OpticalMaterialProperties.h"
#include "Visibilities.h"
#include "FactoryBase.h"

#include <G4GenericMessenger.hh>
#include <G4LogicalVolume.hh>
#include <G4NistManager.hh>
#include <G4ProductionCuts.hh>
#include <G4PVPlacement.hh>
#include <G4Region.hh>
#include <G4SubtractionSolid.hh>
#include <G4OpticalSurface.hh>
#include <G4LogicalSkinSurface.hh>
#include <G4LogicalBorderSurface.hh>
#include <G4Box.hh>
#include <G4Tubs.hh>
#include <G4VisAttributes.hh>


namespace nexus {

  REGISTER_CLASS(FiberBB, GeometryBase)

  using namespace CLHEP;

  FiberBB::FiberBB():
    GeometryBase(),
    scint_size_ (20.0 * mm),
    scint_thick_( 2.0 * mm),
      pmt_size_ (20.5 * mm),
      air_gap_  (1    * mm)
  {
  }

  FiberBB::~FiberBB()
  {
  }

  void FiberBB::Construct()
  {

    G4Box * world_solid = new G4Box ("world", 1 * m, 1 * m, 1 * m);
    G4Box *   pmt_solid = new G4Box ("pmt"  , pmt_size_/2, pmt_size_/2, pmt_size_/2);
    G4Box * scint_solid = new G4Box ("scint", scint_size_/2, scint_size_/2, scint_thick_/2);

    auto air     = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
    auto plastic = G4NistManager::Instance()->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
    auto lead    = G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb" );
    plastic->SetMaterialPropertiesTable(opticalprops::AdHoc(1.0, 1.0*m, 1.58));

    G4LogicalVolume* world_logic = new G4LogicalVolume( world_solid
                                                      , air
                                                      , "world");

    G4LogicalVolume*   pmt_logic = new G4LogicalVolume( pmt_solid
                                                      , lead
                                                      , "pmt");

    G4LogicalVolume* scint_logic = new G4LogicalVolume( scint_solid
                                                      , plastic
                                                      , "scint");

    G4OpticalSurface* interface = new G4OpticalSurface("interface"
                                                      , unified
                                                      , polished
                                                      , dielectric_dielectric
                                                      );
//    interface->SetMaterialPropertiesTable(plastic->GetMaterialPropertiesTable());
    // G4LogicalBorderSurface* WaterSurface = new G4LogicalBorderSurface( "interface"
    //                                                                  , scint_logic
    //                                                                  , world_logic
    //                                                                  , interface);

    world_logic->SetVisAttributes(G4VisAttributes::GetInvisible());
      pmt_logic->SetVisAttributes(nexus::DarkGrey());
    scint_logic->SetVisAttributes(nexus::Blue());

    this->SetLogicalVolume(world_logic);

    auto zero       = G4ThreeVector(0., 0., 0.);
    auto z          = G4ThreeVector(0., 0., 1.);
    auto  scint_pos = zero + scint_thick_/2 * z;
    auto    pmt_pos = zero + (scint_thick_ + air_gap_  + 0.5 * pmt_size_) * z;

    new G4PVPlacement(0, scint_pos, scint_logic, "scint", world_logic, false, 0, false);
    new G4PVPlacement(0,   pmt_pos,   pmt_logic, "pmt"  , world_logic, false, 0, false);

    new G4LogicalSkinSurface("interface", scint_logic, interface);

  }

  G4ThreeVector FiberBB::GenerateVertex(const G4String& region) const
  {
    return G4ThreeVector{0., 0., 0.};
  }

}
