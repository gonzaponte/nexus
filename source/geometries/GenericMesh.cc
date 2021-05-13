// ----------------------------------------------------------------------------
// nexus | GenericMesh.cc
//
// Co GenericMesh at WIS
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "GenericMesh.h"
#include "Visibilities.h"

#include <G4GenericMessenger.hh>
#include <G4Box.hh>
#include <G4Tubs.hh>
#include <G4Material.hh>
#include <G4NistManager.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4MultiUnion.hh>
#include <G4Transform3D.hh>
#include <G4SubtractionSolid.hh>
#include <G4VisAttributes.hh>
#include <G4ThreeVector.hh>
#include <Randomize.hh>

namespace nexus {

using namespace CLHEP;

GenericMesh::GenericMesh():
  visibility_   (false)
, create_lab_   (true)
, wire_material_("")
, wire_shape_   ("")
, mesh_shape_   ("")

, mesh_diam_ (0)
, mesh_width_(0)
, mesh_angle_(0)

, wire_pitch_    (0)
, wire_diam_     (0)
, wire_thickness_(0)
, wire_width_    (0)

{
  msg_ = new G4GenericMessenger{this, "/Geometry/GenericMesh/",
                                "Control commands of a generic mesh geometry."};
  msg_->DeclareProperty("visibility", visibility_,
                        "Visibility of this geometry");

  msg_->DeclareProperty("wire_material" , wire_material_ , "");
  msg_->DeclareProperty("wire_shape"    , wire_shape_    , "");
  msg_->DeclareProperty("mesh_shape"    , mesh_shape_    , "");
  msg_->DeclareProperty("mesh_diam"     , mesh_diam_     , "").SetUnitCategory("Length");
  msg_->DeclareProperty("mesh_width"    , mesh_width_    , "").SetUnitCategory("Length");
  msg_->DeclareProperty("mesh_angle"    , mesh_angle_    , "").SetUnitCategory("Angle");
  msg_->DeclareProperty("wire_pitch"    , wire_pitch_    , "").SetUnitCategory("Length");
  msg_->DeclareProperty("wire_diam"     , wire_diam_     , "").SetUnitCategory("Length");
  msg_->DeclareProperty("wire_thickness", wire_thickness_, "").SetUnitCategory("Length");
  msg_->DeclareProperty("wire_width"    , wire_width_    , "").SetUnitCategory("Length");

}

void GenericMesh::Construct()
{
  G4cout << "wire_material"  << "\t" << wire_material_ << G4endl;
  G4cout << "wire_shape"     << "\t" << wire_shape_    << G4endl;
  G4cout << "mesh_shape"     << "\t" << mesh_shape_    << G4endl;
  G4cout << "mesh_diam"      << "\t" << mesh_diam_     << G4endl;
  G4cout << "mesh_width"     << "\t" << mesh_width_    << G4endl;
  G4cout << "mesh_angle"     << "\t" << mesh_angle_    << G4endl;
  G4cout << "wire_pitch"     << "\t" << wire_pitch_    << G4endl;
  G4cout << "wire_diam"      << "\t" << wire_diam_     << G4endl;
  G4cout << "wire_thickness" << "\t" << wire_thickness_<< G4endl;
  G4cout << "wire_width"     << "\t" << wire_width_    << G4endl;

  if (create_lab_) BuildLab();

  mesh_shape_.toLower();
  wire_shape_.toLower();

  G4double  length = 0;
  G4VSolid* wire   = nullptr;

       if (mesh_shape_ == "rectangular") length = mesh_width_;
  else if (mesh_shape_ == "circular"   ) length = mesh_diam_ ;
  else     G4cerr << "Invalid mesh shape: " << mesh_shape_ << G4endl;

       if (wire_shape_ == "rectangular") wire = new G4Box {"wire", wire_thickness_/2, wire_width_/2, length/2};
  else if (wire_shape_ == "circular"   ) wire = new G4Tubs{"wire", 0, wire_diam_/2, length/2, 0, 2*pi};
  else     G4cerr << "Invalid wire shape: " << wire_shape_ << G4endl;

  G4MultiUnion* all_wires = new G4MultiUnion{"wires"};

  G4int    n_wires = 2 * std::ceil( (length - wire_pitch_) / (2 * wire_pitch_) );
  G4double start   = -wire_pitch_/2 - (n_wires - 1)/2 * wire_pitch_;

  G4RotationMatrix along_x_axis{}; along_x_axis.rotateY(90 * deg);
  G4RotationMatrix along_y_axis{}; along_y_axis.rotateX(90 * deg); along_y_axis.rotateY(90 * deg);

  // HORIZONTAL WIRES
  for (G4int i=0; i<n_wires; i++)
  {
    G4ThreeVector pos       = G4ThreeVector{0,  start + wire_pitch_ * i, 0};
    G4Transform3D transform = G4Transform3D(along_x_axis, pos);
    all_wires->AddNode(*wire, transform);
  }

  // VERTICAL WIRES
  for (G4int i=0; i<n_wires; i++)
  {
    G4ThreeVector pos       = G4ThreeVector{start + wire_pitch_ * i, 0, 0};
    G4Transform3D transform = G4Transform3D(along_y_axis, pos);
    all_wires->AddNode(*wire, transform);
  }
  all_wires->Voxelize();

  G4VSolid* wires = all_wires;

  if (mesh_shape_ == "circular") {
    G4double thickness = std::max(wire_thickness_, wire_diam_);
    G4Tubs* disk = new G4Tubs{"disk", mesh_diam_/2, mesh_diam_, thickness/2, 0, 2*pi};

    G4SubtractionSolid* framed_wires = new G4SubtractionSolid{ "framed_wires"
                                                             , all_wires
                                                             , disk
                                                             };
    wires = framed_wires;
  }

  G4Material*      wire_material = G4NistManager::Instance()->FindOrBuildMaterial(wire_material_);
  G4LogicalVolume* wire_logic    = new G4LogicalVolume{wires, wire_material, "wires"};

  G4RotationMatrix* rotation = new G4RotationMatrix{}; rotation->rotateX(mesh_angle_);
  new G4PVPlacement( rotation
                   , G4ThreeVector{}
                   , wire_logic
                   , "WIRES"
                   , GetLogicalVolume()
                   , false
                   , 0
                   , false
                   );

  if (!create_lab_) this->SetLogicalVolume(wire_logic);

  if (visibility_) {
    G4VisAttributes white = nexus::White();
    white.SetForceSolid(true);
    wire_logic->SetVisAttributes(white);
  }
}

void GenericMesh::BuildLab()
{
  std::vector<G4double> dims{mesh_diam_, mesh_width_, wire_thickness_, wire_diam_};
  G4double         dim = 100 * (*std::max_element(dims.begin(), dims.end()));
  G4Box*           box = new G4Box{"lab", dim, dim, dim};
  G4Material*      mat = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4LogicalVolume* lab = new G4LogicalVolume{box, mat, "LAB"};

  new G4PVPlacement(nullptr, {}, lab, "LAB", nullptr, false, 0, false);

  this->SetLogicalVolume(lab);
}


G4ThreeVector GenericMesh::GenerateVertex(const G4String& region) const{
  G4double x=0, y=0, z=0;
       if (region == "ONAXIS") {
    z = G4UniformRand() * std::max(mesh_diam_, mesh_width_) / 2;
  }
  else if (region == "AROUNDAXIS") {
    x = G4UniformRand() * 2 * wire_pitch_ - wire_pitch_;
    y = G4UniformRand() * 2 * wire_pitch_ - wire_pitch_;
    z = G4UniformRand() * std::max(mesh_diam_, mesh_width_) / 2;
  }
  else {
    G4cerr << "Invalid vertex region: " << region << G4endl;
  }
  z += std::max(wire_thickness_, wire_diam_);
  return G4ThreeVector{x, y, z};
}


} // namespace nexus
