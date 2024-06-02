#include "SquareOpticalFiber.hh"
#include "SquareFiberSD.h"

#include "FactoryBase.h"
#include "OpticalMaterialProperties.h"
#include "MaterialsList.h"

#include "CLHEP/Random/Random.h"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4SDManager.hh"
#include "G4SubtractionSolid.hh"
#include "G4VSensitiveDetector.hh"
#include "G4VSolid.hh"
#include "G4VisAttributes.hh"

#include <vector>
#include <iomanip>
#include <optional>
#include <cassert>

#define CHECK_OVLP false
#define TWO_PI 2*M_PI

namespace nexus{

REGISTER_CLASS(SquareOpticalFiber, GeometryBase)

SquareOpticalFiber::SquareOpticalFiber() :
  GeometryBase(),
  msg_(nullptr),
  specific_vertex_({0., 0., 0.}),
  el_gap_length_(0.),
  pitch_(0.),
  sipm_size_(0.),
  fiber_length_(0.),
  d_fiber_holder_(0.),
  d_anode_holder_(0.),
  holder_thickness_(0.),
  tpb_thickness_(0.),
  sipm_output_file_(""),
  tpb_output_file_(""),
  diff_sigma_(0.),
  n_sipms_(0),
  with_light_tube_(false),
  with_cladding_  ( true),
  with_walls_     (false),
  with_holder_    (false),
  with_fiber_tpb_ ( true),
  with_holder_tpb_(false)
{
  msg_ = new G4GenericMessenger(this, "/Geometry/SquareOpticalFiber/", "Control commands of geometry SquareOpticalFiber.");
  msg_->DeclarePropertyWithUnit("specific_vertex" , "mm",  specific_vertex_ , "Set generation vertex.");
  msg_->DeclarePropertyWithUnit("el_gap_length"   , "mm",  el_gap_length_   , "Set EL gap size.");
  msg_->DeclarePropertyWithUnit("pitch"           , "mm",  pitch_           , "Set pitch/spacing size.");
  msg_->DeclarePropertyWithUnit("d_fiber_holder"  , "mm",  d_fiber_holder_  , "Set depth of fiber in holder.");
  msg_->DeclarePropertyWithUnit("d_anode_holder"  , "mm",  d_anode_holder_  , "Set distance between anode and holder.");
  msg_->DeclarePropertyWithUnit("holder_thickness", "mm",  holder_thickness_, "Teflon holder thickness.");
  msg_->DeclarePropertyWithUnit("tpb_thickness"   , "um",  tpb_thickness_   , "TPB thickness.");

  msg_->DeclareProperty("sipm_path", sipm_output_file_);
  msg_->DeclareProperty( "tpb_path" , tpb_output_file_);
}


SquareOpticalFiber::~SquareOpticalFiber(){
  delete msg;
}

void SquareOpticalFiber::Construct() {
  assert(el_gap_length_    >  0);
  assert(pitch_            >  0);
  assert(sipm_size_        >  0);
  assert(fiber_length_     >  0);
  assert(d_fiber_holder_   >= 0);
  assert(d_anode_holder_   >  0);
  assert(holder_thickness_ >  0);
  assert(tpb_thickness_    >= 0);
  assert(n_sipms_          >  0);
  assert(n_sipms_ % 2      != 0);
  assert(sipm_output_file_ != "");
  assert(tpb_output_file_  != "");

  auto temperature = 298*kelvin;
  auto pressure    =  10*atmosphere;

  // Materials and properties
  auto ptfe       = G4NistManager::Instance() -> FindOrBuildMaterial("G4_TEFLON");
  auto si         = G4NistManager::Instance() -> FindOrBuildMaterial("G4_Si");
  auto xe         = materials::GXe(pressure, temperature);
  auto pmma       = materials::PMMA();
  auto tpb        = materials::TPB();
  auto fpethylene = materials::FPethylene();

  ptfe       -> SetMaterialPropertiesTable(opticalprops::PTFE());
  si         -> SetMaterialPropertiesTable(opticalprops::Si());
  xe         -> SetMaterialPropertiesTable(opticalprops::GXe(pressure, temperature, 1, 1)); // ones represent dummy values
  pmma       -> SetMaterialPropertiesTable(opticalprops::PMMA());
  tpb        -> SetMaterialPropertiesTable(opticalprops::TPB());
  fpethylene -> SetMaterialPropertiesTable(opticalprops::FPethylene());

  // Optical surfaces - The same as in Nexus
  auto ptfe_surface      = new G4OpticalSurface(   "ptfe_surface", unified,   ground, dielectric_metal);
  auto tpb_fiber_surface = new G4OpticalSurface(    "tpb_surface",  glisur,   ground, dielectric_dielectric, 0.01);
  auto vikuiti_coating   = new G4OpticalSurface("vikuiti_surface", unified, polished, dielectric_metal);

  ptfe_surface      -> SetMaterialPropertiesTable(opticalprops::PTFE());
  tpb_fiber_surface -> SetMaterialPropertiesTable(opticalprops::TPB());
  vikuiti_coating   -> SetMaterialPropertiesTable(opticalprops::Vikuiti());

  // WORLD
  auto world_hsize = 2*m;
  auto world_solid = new G4Box("world", world_hsize,  world_hsize,  world_hsize);
  auto world_logic = new G4LogicalVolume(world_solid, xe, "world"); this->SetLogicalVolume(world_logic);
  auto world_phys  = new G4PVPlacement( nullptr       // no rotation
                                      ,  {}           // at origin
                                      ,  world_logic  // its logical volume
                                      ,  "world"      // its name
                                      ,  nullptr      // its mother  volume
                                      ,  false        // no boolean operation
                                      ,  0            // copy number
                                      ,  CHECK_OVLP); // checking overlaps

  // light tube
  auto   sipm_thick   =  1*mm;
  auto barrel_thick   =  5*cm;
  auto barrel_outer_r = 50*cm;
  auto barrel_inner_r = barrel_outer_r - barrel_thick;
  auto barrel_length  = 2*sipm_thick;

  if (light_tube_) {
    auto barrel_solid = new G4Tubs("Barrel"
                                   , barrel_inner_r
                                   , barrel_outer_r
                                   , barrel_length/2
                                   , 0., TWO_PI);
    auto barrel_logic = new G4LogicalVolume(barrel_solid, ptfe, "barrel");
    new G4PVPlacement( nullptr      // no rotation
                     , {0, 0, barrel_length/2}
                     , barrel_logic // its logical volume
                     , "barrel"     // its name
                     , world_logic  // its mother  volume
                     , false        // no boolean operation
                     , 0            // copy number
                     , CHECK_OVLP); // checking overlaps

    new G4LogicalSkinSurface("barrel_surface", barrel_logic, ptfe_surface);
  }

  // SiPM holder
  auto sipm_holder_r     = barrel_outer_r;
  auto sipm_holder_thick = 5*cm;
  auto sipm_holder_z     = fiber_length_ + sipm_thick + sipm_holder_thick/2;
  auto sipm_holder_solid = new G4Tubs( "sipm_holder"
                                     , 0, sipm_holder_r
                                     , cap_thick/2
                                     , 0, TWO_PI);
  auto sipm_holder_logic = new G4LogicalVolume(sipm_holder_solid, ptfe, "sipm_holder");
  new G4PVPlacement( nullptr           // no rotation
                   , {0, 0, sipm_holder_z}
                   , sipm_holder_logic // its logical volume
                   , "sipm_holder"     // name
                   , world_logic       // mother volume
                   , false             // no boolean operation
                   , 0                 // copy number
                   , CHECK_OVLP);      // checking overlaps

  new G4LogicalSkinSurface("sipm_holder_surface", sipm_holder_logic, ptfe_surface);


  // SIPM
  auto sipm_thick = sipm_size_/2;
  auto sipm_solid = new G4Box("SiPM", sipm_size_/2, sipm_size_/2, sipm_thick/2);
  auto sipm_logic = new G4LogicalVolume(sipm_solid, si, "sipm");
  auto max_pos = (n_sipms_ - 1) / 2.0 * pitch_;
  auto sipm_z  = fiber_length + sipm_thick/2;

  // FIBER
  //     CORE
  auto core_solid = new G4Box("fiber_core", sipm_size_/2, sipm_size_/2, fiber_length_/2);
  auto core_logic = new G4LogicalVolume(core_solid, pmma, "fiber_core");
  new G4LogicalSkinSurface("fiber_vikuiti_surface", core_logic, vikuiti_coating);
  //  new G4LogicalBorderSurface("Fiber-Cladding", fiber_core_solid, fiber_clad_solid, vikuiti_coating);

  //     CLADDING
  G4LogicalVolume* fiber_logic = core_logic; // overriden if with_cladding_
  if (with_cladding_) {
    auto cladding_thick = 0.01 * sipm_size_;
    auto cladding_outer = sipm_size_ + cladding_thick;
    auto fiber_solid    = new G4Box("fiber", cladding_outer/2, cladding_outer/2, fiber_length_/2);
    /**/ fiber_logic    = new G4LogicalVolume(fiber_clad_solid, fpethylene, "fiber");
    new G4PVPlacement(nullptr, {}, core_logic, "fiber", fiber_logic, false, 0, CHECK_OVLP);
  }

  //     TPB COATING ON ENTRANCE
  auto fiber_tpb_solid = new G4Box("fiber_tpb", sipm_size_/2, sipm_size_/2, tpb_thickness_/2);
  auto fiber_tpb_logic = new G4LogicalVolume(fiber_tpb_solid, tpb, "fiber_tpb");

  // Fiber holder. Holder hole size depends on cladding
  auto holder_hole_size = ((G4Box*) fiber_logic -> GetSolid()) -> GetXHalfLength() * 2;
  auto holder_full      = new G4Tubs("fiber_holder", 0, barrel_outer_r, holder_thickness_/2, 0, TWO_PI);
  auto holder_hole      = new G4Box ("holder_hole", holder_hole_size/2, holder_hole_size/2, (holder_thickness_+2*tpb_thickness_)/2);


  // CREATE ARRAY
  auto fiber_z     = fiber_length_/2;
  auto fiber_tpb_z = -tpb_thickness_/2;

  std::vector<G4ThreeVector> sipm_poss;
  auto holder_holes = new G4MultiUnion("holes");
  auto max_idx      = (n_sipms_ - 1) / 2;
  auto copy_no      = 0;

  for   (auto i=-max_idx; i<=max_idx, ++i) {
    for (auto j=-max_idx; j<=max_idx, ++j) {
      auto x = pitch_ * i;
      auto y = pitch_ * j;
      sipm_poss.emplace_back(x, y, sipm_z);

      new G4PVPlacement( nullptr,     // no rotation
                       , {x, y, sipm_z}
                       , sipm_logic   // its logical volume
                       , "sipm"       // its name
                       , world_logic  // its mother volume
                       , false        // no boolean operation
                       , copy_no      // copy number, this time each SiPM has a unique copy number
                       , false);      // checking overlaps

      auto fiber_phys = new G4PVPlacement( nullptr     // no rotation
                                         , {x, y, fiber_z}
                                         , fiber_logic // its logical volume
                                         , "fiber"     // its name
                                         , world_logic // its mother volume
                                         , false       // no boolean operation
                                         , copy_no     // copy number
                                         , false);     // checking overlaps

      if (with_fiber_tpb_) {
        auto fiber_tpb_phys = new G4PVPlacement( nullptr         // no rotation
                                               , {x, y, fiber_tpb_z}
                                               , fiber_tpb_logic // its logical volume
                                               , "fiber_tpb"     // its name
                                               , world_logic     // its mother volume
                                               , false           // no boolean operation
                                               , copy_no         // copy number
                                               , false);         // checking overlaps

        new G4LogicalBorderSurface("tpb_fiber", fiber_tpb_phys, fiber_phys    , fiber_tpb_surface);
        new G4LogicalBorderSurface("fiber_tpb", fiber_phys    , fiber_tpb_phys, fiber_tpb_surface);
      }

      holder_holes -> AddNode(*holder_hole, G4Translate3D(x, y, 0));

      copy_no++;
    }
  }
  holder_holes -> Voxelize();

  if (with_holder_) {
    auto fiber_holder_z = d_fiber_holder_/2;
    auto fiber_holder_solid = new G4SubtractionSolid("fiber_holder", holder, holder_holes);
    auto fiber_holder_logic = new G4LogicalVolume(fiber_holder_solid, ptfe, "fiber_holder");
    new G4LogicalSkinSurface("holder_surface", fiber_holder_logic, ptfe_surface);
    new G4PVPlacement( nullptr
                     , {0, 0, fiber_holder_z}
                     , fiber_holder_logic
                     , "fiber_holder"
                     , world_logic
                     , false
                     , 0
                     , CHECK_OVLP);

    if (with_holder_tpb_) {
      auto holder_tpb_z     = d_fiber_holder_ + tpb_thickness_/2;
      auto holder_tpb_full  = new G4Tubs("fibers_holder_full", 0, barrel_outer_r, tpb_thickness_/2, 0, TWO_PI);
      auto holder_tpb_solid = new G4SubtractionSolid("fibers_holder_tpb", holder_tpb_full, holder_holes);
      auto holder_tpb_logic = new G4LogicalVolume(holder_tpb_solid, tpb, "fiber_holder_tpb");
      new G4PVPlacement( nullptr
                       , {0, 0, holder_tpb_z}
                       , holder_tpb_logic
                       , "fiber_holder_tpb"
                       , world_logic
                       , false
                       , 0
                       , CHECK_OVLP);
    }
  }

  sipm_logic                               -> SetVisAttributes(new G4VisAttributes(G4Color::Cyan  ()));
  core_logic                               -> SetVisAttributes(new G4VisAttributes(G4Color::White ()));
  fiber_tpb_logic                          -> SetVisAttributes(new G4VisAttributes(G4Color::Green ()));
  if (with_cladding_)   fiber_logic        -> SetVisAttributes(new G4VisAttributes(G4Color::Brown ()));
  if (with_holder_)     fiber_holder_logic -> SetVisAttributes(new G4VisAttributes(G4Color::Yellow()));
  if (with_holder_ &&
      with_holder_tpb_) holder_tpb_logic   -> SetVisAttributes(new G4VisAttributes(G4Color::Green()));


  // SENSITIVE DETECTORS
  auto square_fiber_sd = new SquareFiberSD("square_fiber", sipm_output_file_, tpb_output_file_);
  G4SDManager::GetSDMpointer() -> AddNewDetector(square_fiber_sd);

  fiber_tpb_logic -> SetSensitiveDetector(square_fiber_sd);

  if (with_holder_tpb_ && holder_tpb_logic) {
    holder_tpb_logic -> SetSensitiveDetector(square_fiber_sd);
  }
  sipm_logic -> SetSensitiveDetector(square_fiber_sd);

} // Construct()


G4ThreeVector SquareOpticalFiber::GenerateVertex(const G4String& region) const {
    if (region == "CENTER") { return {}; }

    if (region == "AD_HOC") { return specific_vertex_; }

    if (region == "FACE_RANDOM_DIST") {
      auto x = (G4UniformRand() - 0.5) * sipm_size_;
      auto y = (G4UniformRand() - 0.5) * sipm_size_;
      auto z = specific_vertex_.z();
      return {x, y, z};
    }

    if (region == "LINE_SOURCE_EL") {
      auto el_gap_end = -d_fiber_holder_ - d_anode_holder_;
      auto x = specific_vertex_.x();
      auto y = specific_vertex_.y();
      auto z = el_gap_end - G4UniformRand() * el_gap_length_ ;
      return {x, y, z};
    }

    if (region == "LINE_SOURCE_EL_TRANSVERSE_DIFFUSION"){
      auto el_gap_end = -d_fiber_holder_ - d_anode_holder_;
      auto x = G4RandGauss::shoot() * diff_sigma_ + specific_vertex_.x();
      auto y = G4RandGauss::shoot() * diff_sigma_ + specific_vertex_.y();
      auto z = el_gap_end - G4UniformRand() * el_gap_length_ ;
      return {x, y, z};
    }

    G4Exception("[SquareOpticalFiber]", "GenerateVertex()", FatalException, "Unknown vertex generation region!");
    return G4ThreeVector();

} // GenerateVertex


} // close namespace

#undef TWO_PI
#undef CHECK_OVLP
