// ----------------------------------------------------------------------------
// nexus | FrogXe.cc

// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "FrogXe.hh"
#include "FactoryBase.h"

#include "MaterialsList.h"
#include "OpticalMaterialProperties.h"
#include <G4Box.hh>
#include <G4NistManager.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4Material.hh>
#include <G4ThreeVector.hh>
#include <G4VisAttributes.hh>
#include <G4SystemOfUnits.hh>

namespace nexus {

  REGISTER_CLASS(FrogXe, GeometryBase)

  FrogXe::FrogXe():
    GeometryBase()
  {
  }

  FrogXe::~FrogXe()
  {
  }

  void FrogXe::Construct()
  {
    G4String world_name = "WORLD";

    G4Material* world_mat = materials::GXe(1*bar, 290*kelvin);
    world_mat->SetMaterialPropertiesTable(opticalprops::GXe(1*bar, 290*kelvin, 1, 1));

    G4Box* world_solid_vol = new G4Box(world_name, 1*m, 1*m, 1*m);

    G4LogicalVolume* world_logic_vol = new G4LogicalVolume(world_solid_vol, world_mat, world_name);
    GeometryBase::SetLogicalVolume(world_logic_vol);
  }

    G4ThreeVector FrogXe::GenerateVertex(const G4String& region) const
  {
    G4ThreeVector vertex(0.,0.,0.);
    return vertex;
  }

} // end namespace nexus
