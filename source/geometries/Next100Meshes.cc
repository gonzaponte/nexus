// ----------------------------------------------------------------------------
// nexus | Next100Meshes.cc
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "Next100Meshes.h"
#include "MaterialsList.h"
#include "Visibilities.h"
#include "OpticalMaterialProperties.h"
#include "XenonProperties.h"
#include "CylinderPointSampler.h"
#include "HexagonMeshTools.h"
#include "FactoryBase.h"

#include <G4SystemOfUnits.hh>
#include <G4GenericMessenger.hh>
#include <G4PVPlacement.hh>
#include <G4VisAttributes.hh>
#include <G4Material.hh>
#include <G4MaterialPropertiesTable.hh>
#include <G4LogicalVolume.hh>
#include <G4VPhysicalVolume.hh>
#include <G4Tubs.hh>
#include <G4OpticalSurface.hh>
#include <G4LogicalSkinSurface.hh>
#include <G4NistManager.hh>
#include <G4UnitsTable.hh>

namespace nexus {

REGISTER_CLASS(Next100Meshes, GeometryBase)

Next100Meshes::Next100Meshes() :
  GeometryBase(),
  el_diam_(1*m),
  el_thick_(0.13*mm),
  el_gap_length_(10*mm),
  hex_indiam_(2.5*mm),
  rotation_(0. * deg),
  reflectivity_(0.)
{
  msg_ = new G4GenericMessenger(this, "/Geometry/Meshes/", "Control commands of this class.");
  msg_->DeclareProperty("reflectivity", reflectivity_, "Mesh reflectivity");
  msg_->DeclareProperty("rotation", rotation_, "Mesh rotation");
}


void Next100Meshes::Construct()
{
  auto gas = materials::GXe(4 * bar, 300 * kelvin); gas -> SetMaterialPropertiesTable(opticalprops::GXe(4 * bar, 300 * kelvin, 0, 1));
  auto steel = materials::Steel316Ti(); steel -> SetMaterialPropertiesTable(new G4MaterialPropertiesTable());

  // Dist from centre of hex to hex vertex, excluding the land width (circumradius)
  auto hex_circumradius = hex_indiam_/std::sqrt(3);
  auto n_hex = (G4int) (el_diam_ / 2.0 / hex_circumradius);

  // Define the disk to punch hexagon holes through for the mesh
  auto grid_solid = new G4Tubs("", 0., el_diam_/2.0 , el_thick_/2., 0., twopi);
  auto grid_logic = new G4LogicalVolume(grid_solid, steel, "EL_GRID");

  // Define a hexagonal prism
  auto hex_prism = CreateHexagon(el_thick_/2.0, hex_circumradius);
  auto hex_logic = new G4LogicalVolume(hex_prism, gas, "MESH_HEX_GAS");

  // Place GXe hexagons in the disk to make the mesh
  PlaceHexagons(n_hex, hex_indiam_, el_thick_, grid_logic, hex_logic, el_diam_);

  // Add optical surface
  auto gas_mesh_opsur = new G4OpticalSurface("GAS_EL_MESH_OPSURF");
  gas_mesh_opsur -> SetType(dielectric_metal);
  gas_mesh_opsur -> SetModel(unified);
  gas_mesh_opsur -> SetFinish(ground);
  gas_mesh_opsur -> SetSigmaAlpha(0.0);
  gas_mesh_opsur -> SetMaterialPropertiesTable(opticalprops::Steel(reflectivity_ * perCent));
  new G4LogicalSkinSurface("GAS_EL_MESH_OPSURF", grid_logic, gas_mesh_opsur);

  auto gas_solid = new G4Tubs("GAS", 0., el_diam_/2., el_gap_length_, 0., twopi);
  auto gas_logic = new G4LogicalVolume(gas_solid, gas, "EL_GAP");

  // Create a rotation vector to change the orientation of the EL mesh
  CLHEP::HepRotationZ Roty(rotation_);
  G4RotationMatrix* pRot = new G4RotationMatrix();
  pRot->set(Roty);

  new G4PVPlacement(0, {0., 0., -el_gap_length_/2.}, grid_logic,  "GATE", gas_logic, false, 0, false);
  new G4PVPlacement(0, {0., 0., +el_gap_length_/2.}, grid_logic, "ANODE", gas_logic, false, 0, false);

  sampler_ = new CylinderPointSampler(0., hex_indiam_/2., el_gap_length_/2., 0., twopi, nullptr, {});

  G4VisAttributes red  = nexus::Red();
  red.SetForceSolid(true);
  grid_logic->SetVisAttributes(red);

  new G4PVPlacement(0, {}, gas_logic,  "GAS", nullptr, false, 0, false);
  this -> SetLogicalVolume(gas_logic);
  this -> SetSpan(std::max(el_diam_, el_gap_length_));
  this -> SetDimensions({el_diam_/2., el_diam_/2., el_gap_length_/2.});

}

Next100Meshes::~Next100Meshes()
{
  delete sampler_;
  delete msg_;
}

G4ThreeVector Next100Meshes::GenerateVertex(const G4String& region) const
{
  G4ThreeVector vertex(0., 0., 0.);

  if (region == "CENTER") {}
  else if (region == "SAMPLER") {
    vertex = sampler_ -> GenerateVertex(VOLUME);
  }
  else {
    G4Exception("[Next100Meshes]", "GenerateVertex()", FatalException,
    "Unknown vertex generation region!");
  }
  return vertex;
}

} // namespace nexus
