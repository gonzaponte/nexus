#include "XeTrap.hh"

#include "FactoryBase.h"
#include "OpticalMaterialProperties.h"
#include "MaterialsList.h"
#include "SquareOpticalFiber.hh"
#include "SquareFiberSD.h"
#include "G4Box.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4GenericMessenger.hh"
#include "G4NistManager.hh"
#include "G4OpticalSurface.hh"
#include "G4PVPlacement.hh"
#include "G4SDManager.hh"
#include "G4SubtractionSolid.hh"
#include "G4Tubs.hh"
#include "G4VisAttributes.hh"
#include "Randomize.hh"

#include <vector>
#include <iomanip>
#include <optional>
#include <cassert>

#define CHECK_OVLP false
#define TWO_PI 2*M_PI
#define PLACE(X, Y, Z, LOGIC, NAME, MOTHER, COPY) new G4PVPlacement(nullptr, {X,Y,Z}, LOGIC, NAME, MOTHER, false, COPY, CHECK_OVLP)
#define PLACE_Z(Z, LOGIC, NAME, MOTHER) PLACE(0, 0, Z, LOGIC, NAME, MOTHER, 0)
#define PLACE_ORG(LOGIC, NAME, MOTHER) PLACE_Z(0, LOGIC, NAME, MOTHER)

namespace nexus{

REGISTER_CLASS(XeTrap, GeometryBase);

XeTrap::XeTrap() :
  GeometryBase(),
  msg_(nullptr),
  sipm_size_(5*mm),
  sipm_thickness_(1*mm),
  pmma_size_(50*mm),
  pmma_thickness_(5*mm),
  tpb_thickness_(5 * um),
  tpb_surface_roughness_(0),
  reflector_reflectivity_(1)
{
  msg_ = new G4GenericMessenger(this, "/Geometry/XeTrap/", "Control commands of geometry XeTrap.");
  msg_ -> DeclarePropertyWithUnit("sipm_size"     , "mm", sipm_size_     , "Set SiPM and fiber size.");
  msg_ -> DeclarePropertyWithUnit("pmma_size"     , "mm", pmma_size_     , "Set PMMA size.");
  msg_ -> DeclarePropertyWithUnit("pmma_thickness", "mm", pmma_thickness_, "Set PMMA thickness.");
  msg_ -> DeclarePropertyWithUnit( "tpb_thickness", "um",  tpb_thickness_, "Set TPB thickness.");

  msg_ -> DeclareProperty("tpb_surface_roughness", tpb_surface_roughness_, "Set the roughness of the TPB layer.");
  msg_ -> DeclareProperty("reflector_reflectivity" , reflector_reflectivity_ , "Set reflectivity of walls.");
}


XeTrap::~XeTrap(){
  delete msg_;
}

void XeTrap::Construct() {
  auto temperature = 298*kelvin;
  auto pressure    =  10*atmosphere;

  // Materials and properties
  auto ptfe       = G4NistManager::Instance() -> FindOrBuildMaterial("G4_TEFLON");
  auto si         = G4NistManager::Instance() -> FindOrBuildMaterial("G4_Si");
  auto xe         = materials::GXe(pressure, temperature);
  auto pmma       = materials::PMMA();
  auto tpb        = materials::TPB();

  ptfe       -> SetMaterialPropertiesTable(opticalprops::PTFE(reflector_reflectivity_));
  si         -> SetMaterialPropertiesTable(opticalprops::Si());
  xe         -> SetMaterialPropertiesTable(opticalprops::GXe(pressure, temperature, 1, 1)); // ones represent dummy values
  pmma       -> SetMaterialPropertiesTable(opticalprops::PMMA());
  tpb        -> SetMaterialPropertiesTable(opticalprops::TPB());

  // Optical surfaces - The same as in Nexus
  auto ptfe_surface    = new G4OpticalSurface(   "ptfe_surface", unified,   ground, dielectric_metal);
  auto tpb_surface     = new G4OpticalSurface(    "tpb_surface",  glisur,   ground, dielectric_dielectric, tpb_surface_roughness_);
  auto pmma_surface    = new G4OpticalSurface(   "pmma_surface", unified, polished, dielectric_dielectric, 0.0);

  ptfe_surface    -> SetMaterialPropertiesTable(opticalprops::PTFE(reflector_reflectivity_));
  tpb_surface     -> SetMaterialPropertiesTable(opticalprops::TPB());
  pmma_surface    -> SetMaterialPropertiesTable(opticalprops::PMMA());

  // GAS
  auto gas_solid = new G4Box("gas", pmma_size_, pmma_size_, pmma_thickness_*2);
  auto gas_logic = new G4LogicalVolume(gas_solid, xe, "gas"); this->SetLogicalVolume(gas_logic);
  auto gas_phys  = PLACE_ORG(gas_logic, "gas", nullptr);

  // SiPM
  auto sipm_solid = new G4Box("SiPM", sipm_size_/2, sipm_size_/2, sipm_thickness_/2);
  auto sipm_logic = new G4LogicalVolume(sipm_solid, si, "sipm");
  PLACE_Z(sipm_thickness_/2, sipm_logic, "sipm", gas_logic);

  // Reflector
  auto refl_solid = new G4Box("reflector", pmma_size_*3/4, pmma_size_*3/4, pmma_thickness_/2);
  auto refl_logic = new G4LogicalVolume(refl_solid, ptfe, "reflector");
  auto refl_phys  = PLACE_Z(-pmma_thickness_/2, refl_logic, "reflector", gas_logic);
  auto pmma_solid = new G4Box("pmma"     , pmma_size_/2     , pmma_size_/2     , pmma_thickness_/2);
  auto pmma_logic = new G4LogicalVolume(pmma_solid, pmma, "pmma");
  auto pmma_phys  = PLACE_Z(0, pmma_logic, "pmma", refl_logic);
  new G4LogicalSkinSurface("reflector", refl_logic, ptfe_surface);

  auto tpb_solid = new G4Box("tpb", pmma_size_/2, pmma_size_/2, tpb_thickness_/2);
  auto tpb_logic = new G4LogicalVolume(tpb_solid, tpb, "tpb");
  auto tpb_phys  = PLACE_Z(-pmma_thickness_ - tpb_thickness_/2, tpb_logic, "tpb", gas_logic);

  new G4LogicalBorderSurface("gas_tpb", gas_phys, tpb_phys, tpb_surface);
  new G4LogicalBorderSurface("tpb_gas", tpb_phys, gas_phys, tpb_surface);

  new G4LogicalBorderSurface("tpb_pmma", tpb_phys, pmma_phys, tpb_surface);
  new G4LogicalBorderSurface("pmma_tpb", pmma_phys, tpb_phys, tpb_surface);

//  gas_logic                              -> SetVisAttributes(G4VisAttributes::GetInvisible());
  pmma_logic                             -> SetVisAttributes(G4Color::Green());
  refl_logic                             -> SetVisAttributes(G4Color::White());
   tpb_logic                             -> SetVisAttributes(G4Color::Blue ());
  sipm_logic                             -> SetVisAttributes(G4Color::Red  ());

  // SENSITIVE DETECTORS
  auto sipm_sd = new SquareFiberSD("sipm", "sipm.txt", "tpb.txt");
  G4SDManager::GetSDMpointer() -> AddNewDetector(sipm_sd);

  sipm_logic -> SetSensitiveDetector(sipm_sd);

  this -> SetELzCoord(0.);
  this -> SetELWidth(0.);
} // Construct()



G4ThreeVector XeTrap::GenerateVertex(const G4String& region) const {
  auto x = (G4UniformRand() - 0.5) * pmma_size_;
  auto y = (G4UniformRand() - 0.5) * pmma_size_;
  auto z = pmma_thickness_ + tpb_thickness_ + 1 * mm;
  return {x, y, -z};
}

} // close namespace

#undef TWO_PI
#undef CHECK_OVLP
#undef PLACE
#undef PLACE_Z
#undef PLACE_ORG
