// -----------------------------------------------------------------------------
// nexus | WLSFiber.cc
//
//
//
// The NEXT Collaboration
// -----------------------------------------------------------------------------

#include "WLSFiber.h"
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
#include <G4Box.hh>
#include <G4Tubs.hh>
#include <G4VisAttributes.hh>


namespace nexus {

  using namespace CLHEP;

  WLSFiber::WLSFiber(G4double length, G4double diameter):
    GeometryBase(),
    length_  (length),
    diameter_(diameter)
  {
  }

  WLSFiber::~WLSFiber()
  {
  }

  void WLSFiber::Construct()
  {
    G4double clad_fraction = 0.02;
    G4double core_diam = diameter_ * (1. - 2.*clad_fraction);
    G4double clad_diam = diameter_ * (1. -    clad_fraction);

    G4double alum_thickness = 1.0 * um;

    G4Tubs* core_solid = new G4Tubs( "fiber_core"
                                   , 0, diameter_ / 2
                                   , length_ / 2.
                                   , 0., 360. * deg);

    G4Tubs* inner_solid = new G4Tubs( "inner_clad"
                                    , core_diam / 2., clad_diam / 2.
                                    , length_ / 2.
                                    , 0., 360. * deg);

    G4Tubs* outer_solid = new G4Tubs( "outer_clad"
                                    , clad_diam / 2., diameter_ / 2.
                                    , length_ / 2.
                                    , 0., 360. * deg);

    G4Tubs*  alum_solid = new G4Tubs( "alum_reflector"
                                    , 0., diameter_ / 2.
                                    , alum_thickness
                                    , 0., 360. * deg);

    G4Material* pvt = materials::EJ280();
    pvt->SetMaterialPropertiesTable(opticalprops::EJ280());

    G4Material* pmma = materials::PMMA();
    pmma->SetMaterialPropertiesTable(opticalprops::PMMA());

    G4Material* fp = materials::FPethylene();
    fp->SetMaterialPropertiesTable(opticalprops::FPethylene());

    G4Material* aluminium = G4NistManager::Instance()->FindOrBuildMaterial("G4_Al");
    aluminium->SetMaterialPropertiesTable(opticalprops::Aluminium());

    G4LogicalVolume*  core_logic = new G4LogicalVolume( core_solid
                                                      , pvt
                                                      , "fiber_core");

    G4LogicalVolume* inner_logic = new G4LogicalVolume( inner_solid
                                                      , pmma
                                                      , "inner_clad");

    G4LogicalVolume* outer_logic = new G4LogicalVolume( outer_solid
                                                      , fp
                                                      , "outer_clad");

    G4LogicalVolume*  alum_logic = new G4LogicalVolume( alum_solid
                                                      , aluminium
                                                      , "alum_reflector");

    G4OpticalSurface* alum_surf = new G4OpticalSurface( "alum_surface"
                                                      , unified
                                                      , polished
                                                      , dielectric_metal
                                                      );
    alum_surf->SetMaterialPropertiesTable(opticalprops::Aluminium());

    G4ThreeVector zero  {0., 0., 0.};
    G4ThreeVector bottom{0., 0., -(length_ + alum_thickness)/2.};
    new G4PVPlacement(nullptr,   zero, inner_logic, "inner_clad"    , core_logic, false, 0, false);
    new G4PVPlacement(nullptr,   zero, outer_logic, "outer_clad"    , core_logic, false, 0, false);
    new G4PVPlacement(nullptr, bottom,  alum_logic, "alum_reflector", core_logic, false, 0, false);

    new G4LogicalSkinSurface("alum_surface", alum_logic, alum_surf);

    this->SetLogicalVolume(core_logic);

     core_logic->SetVisAttributes(nexus::DarkGreen());
    inner_logic->SetVisAttributes(nexus::DarkGreenAlpha());
    outer_logic->SetVisAttributes(nexus::LightGreenAlpha());
     alum_logic->SetVisAttributes(nexus::DarkGrey());
  }

  G4ThreeVector WLSFiber::GenerateVertex(const G4String& region) const {
    return G4ThreeVector{0., 0., 0.};
  }

}
