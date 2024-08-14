#include "SquareOpticalFiber.hh"
#include "SquareFiberSD.h"

#include "FactoryBase.h"
#include "OpticalMaterialProperties.h"
#include "MaterialsList.h"

#include "G4Box.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4MultiUnion.hh"
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

REGISTER_CLASS(SquareOpticalFiber, GeometryBase)

SquareOpticalFiber::SquareOpticalFiber() :
  GeometryBase(),
  msg_(nullptr),
  specific_vertex_({0., 0., 0.}),
  sipm_size_(0.),
  fiber_length_(0.),
  el_gap_length_(0.),
  pitch_(0.),
  d_fiber_holder_(0.),
  d_anode_holder_(0.),
  tpb_thickness_(0.),
  tpb_surface_roughness_(0),
  coating_reflectivity_(0),
  diff_sigma_(0.),
  n_sipms_(0),
  sipm_output_file_(""),
  tpb_output_file_(""),
  with_holder_    ( true),
  with_fiber_tpb_ ( true),
  with_holder_tpb_(false)
{
  msg_ = new G4GenericMessenger(this, "/Geometry/SquareOpticalFiber/", "Control commands of geometry SquareOpticalFiber.");
  msg_ -> DeclarePropertyWithUnit("specific_vertex" , "mm", specific_vertex_, "Set generation vertex.");
  msg_ -> DeclarePropertyWithUnit("sipm_size"       , "mm", sipm_size_      , "Set SiPM and fiber size.");
  msg_ -> DeclarePropertyWithUnit("fiber_length"    , "mm", fiber_length_   , "Set fiber length.");
  msg_ -> DeclarePropertyWithUnit("el_gap_length"   , "mm", el_gap_length_  , "Set EL gap length.");
  msg_ -> DeclarePropertyWithUnit("pitch"           , "mm", pitch_          , "Set sensor pitch.");
  msg_ -> DeclarePropertyWithUnit("d_fiber_holder"  , "mm", d_fiber_holder_ , "Set depth of fiber in holder.");
  msg_ -> DeclarePropertyWithUnit("d_anode_holder"  , "mm", d_anode_holder_ , "Set distance anode-holder.");
  msg_ -> DeclarePropertyWithUnit("tpb_thickness"   , "um", tpb_thickness_   , "Set TPB thickness.");

  msg_ -> DeclareProperty("tpb_surface_roughness", tpb_surface_roughness_, "Set the roughness of the TPB layer.");
  msg_ -> DeclareProperty("coating_reflectivity" , coating_reflectivity_ , "Set reflectivity of Vikuiti coating.");
  msg_ -> DeclareProperty(   "n_sipms", n_sipms_        , "Set Number of SiPMs per axis.");
  msg_ -> DeclareProperty(    "holder", with_holder_    , "Add fiber holder to geometry.");
  msg_ -> DeclareProperty( "fiber_tpb", with_fiber_tpb_ , "Add fiber tpb coating to geometry.");
  msg_ -> DeclareProperty("holder_tpb", with_holder_tpb_, "Add holder tpb coating to geometry.");

  msg_ -> DeclareProperty("sipm_path", sipm_output_file_);
  msg_ -> DeclareProperty( "tpb_path" , tpb_output_file_);
}


SquareOpticalFiber::~SquareOpticalFiber(){
  delete msg_;
}

void SquareOpticalFiber::Construct() {
  assert(el_gap_length_         >  0);
  assert(pitch_                 >  0);
  assert(sipm_size_             >  0);
  assert(fiber_length_          >  0);
  //  assert(d_fiber_holder_        >= 0);
  assert(d_anode_holder_        >  0);
  assert(tpb_thickness_         >  0);
  assert(tpb_surface_roughness_ >  0);
  assert(tpb_surface_roughness_ <= 1);
  assert(coating_reflectivity_  >  0);
  assert(coating_reflectivity_  <= 1);
  assert(n_sipms_               >  0);
  assert(n_sipms_ % 2           != 0);
  //  assert(sipm_output_file_      != "");
  assert(tpb_output_file_       != "");

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
  auto ptfe_surface    = new G4OpticalSurface(   "ptfe_surface", unified,   ground, dielectric_metal);
  auto tpb_surface     = new G4OpticalSurface(    "tpb_surface",  glisur,   ground, dielectric_dielectric, tpb_surface_roughness_);
  auto pmma_surface    = new G4OpticalSurface(   "pmma_surface", unified, polished, dielectric_dielectric, 0.0);
  auto vikuiti_coating = new G4OpticalSurface("vikuiti_surface", unified, polished, dielectric_metal);

  ptfe_surface    -> SetMaterialPropertiesTable(opticalprops::PTFE());
  tpb_surface     -> SetMaterialPropertiesTable(opticalprops::TPB());
  pmma_surface    -> SetMaterialPropertiesTable(opticalprops::PMMA());
  vikuiti_coating -> SetMaterialPropertiesTable(opticalprops::Vikuiti(coating_reflectivity_));

  /// Fibers entry at (x, y, 0)
  /// Fibers exit  at (x, y, +fibers_length_)
  /// SiPMs stick out from tracking plane
  auto   sipm_thick = 1*mm;
  auto     tp_thick = 1*cm;
  auto holder_thick = d_fiber_holder_ > 0 ? d_fiber_holder_ : 10*mm;

  // GAS
  auto tracking_plane_r = (n_sipms_ / 2 * pitch_ + sipm_size_) * std::sqrt(2); // just a bit bigger than needed
  auto gas_length       = std::max( fiber_length_ + sipm_thick + tp_thick
                                  , holder_thick + d_anode_holder_ + el_gap_length_);
  auto gas_solid = new G4Tubs("gas", 0, tracking_plane_r * 1.1, gas_length * 1.1, 0, TWO_PI);
  auto gas_logic = new G4LogicalVolume(gas_solid, xe, "gas"); this->SetLogicalVolume(gas_logic);
  auto gas_phys  = PLACE_ORG(gas_logic, "gas", nullptr);
  std::cerr << "WORLD LIMS " << tracking_plane_r / mm << " " << gas_length * 1.1 / 2. / mm << std::endl;

  // SiPM
  auto sipm_solid = new G4Box("SiPM", sipm_size_/2, sipm_size_/2, sipm_thick/2);
  auto sipm_logic = new G4LogicalVolume(sipm_solid, si, "sipm");

  // Tracking plane (sipm holder)
  auto tp_z     = fiber_length_ + sipm_thick + tp_thick/2;
  auto tp_solid = new G4Tubs( "sipm_holder", 0, tracking_plane_r, tp_thick/2, 0, TWO_PI);
  auto tp_logic = new G4LogicalVolume(tp_solid, ptfe, "sipm_holder");
  PLACE_Z(tp_z, tp_logic, "sipm_holder", gas_logic);

  new G4LogicalSkinSurface("sipm_holder_surface", tp_logic, ptfe_surface);

  // FIBER CORE
  auto core_solid = new G4Box("fiber_core", sipm_size_/2, sipm_size_/2, fiber_length_/2);
  auto core_logic = new G4LogicalVolume(core_solid, pmma, "fiber_core");

  // Vikuiti coating
  auto refl_thick = 0.01 * sipm_size_;
  auto refl_outer = sipm_size_ + 2*refl_thick;
  auto refl_solid = new G4Box("reflector", refl_outer/2, refl_outer/2, fiber_length_/2);
  auto refl_logic = new G4LogicalVolume(refl_solid, fpethylene, "reflector");

  auto fiber_solid = new G4Box("fiber", refl_outer/2, refl_outer/2, fiber_length_/2 + tpb_thickness_);
  auto fiber_logic = new G4LogicalVolume(fiber_solid, xe, "fiber");
  auto core_phys   = PLACE_ORG(                core_logic, "core",  refl_logic);
  auto refl_phys   = PLACE_Z  (tpb_thickness_, refl_logic, "clad", fiber_logic);

  // WARNING: skins affect all interfaces of a given volume, not suitable for the fibers!
  //  new G4LogicalSkinSurface("fiber_vikuiti_surface", fiber_logic, vikuiti_coating);
  new G4LogicalBorderSurface("fiber_vikuiti",      core_phys,      refl_phys, vikuiti_coating);
  new G4LogicalBorderSurface("vikuiti_fiber",      refl_phys,      core_phys, vikuiti_coating);

  auto gas_pad_thick = with_fiber_tpb_ ? tpb_thickness_/2 : tpb_thickness_;
  auto gas_pad_z     = with_fiber_tpb_ ? -fiber_length_/2 - tpb_thickness_/2 : -fiber_length_/2;
  auto gas_pad_solid = new G4Box("fiber_gas", refl_outer/2, refl_outer/2, gas_pad_thick);
  auto gas_pad_logic = new G4LogicalVolume(gas_pad_solid, xe, "gas_pad");
  auto gas_pad_phys  = PLACE_Z(gas_pad_z, gas_pad_logic, "gas_pad", fiber_logic);

  // FIBER TPB COATING ON ENTRANCE
  G4LogicalVolume* fiber_tpb_logic = nullptr;
  if (with_fiber_tpb_) {
    auto fiber_tpb_solid = new G4Box("fiber_tpb", refl_outer/2, refl_outer/2, tpb_thickness_/2);
    /**/ fiber_tpb_logic = new G4LogicalVolume(fiber_tpb_solid, tpb, "fiber_tpb");
    auto fiber_tpb_z     = -fiber_length_/2 + tpb_thickness_/2;
    auto fiber_tpb_phys  = PLACE_Z(fiber_tpb_z, fiber_tpb_logic, "fiber_tpb", fiber_logic);

    new G4LogicalBorderSurface("fiber_tpb",      core_phys, fiber_tpb_phys, tpb_surface);
    new G4LogicalBorderSurface("tpb_fiber", fiber_tpb_phys,      core_phys, tpb_surface);
    new G4LogicalBorderSurface("gas_tpb",   gas_pad_phys, fiber_tpb_phys, tpb_surface);
    new G4LogicalBorderSurface("tpb_gas", fiber_tpb_phys,   gas_pad_phys, tpb_surface);
  }
  else {
    new G4LogicalBorderSurface("gas_tpb",   gas_pad_phys,      core_phys, pmma_surface);
    new G4LogicalBorderSurface("tpb_gas",      core_phys,   gas_pad_phys, pmma_surface);
  }

  // Fiber holder. Holder hole size depends on cladding
  //((G4Box*) fiber_logic -> GetSolid()) -> GetXHalfLength() * 2;
  auto holder_hole_size  = refl_outer;
  auto holder_hole_thick = holder_thick + 4*tpb_thickness_; // hole bigger to make sure we subtract everything
  auto holder_full       = new G4Tubs("fiber_holder", 0, tracking_plane_r, holder_thick/2, 0, TWO_PI);
  auto holder_hole       = new G4Box ("holder_hole", holder_hole_size/2, holder_hole_size/2, holder_hole_thick/2);
  std::cerr << "HOLDER HOLE SIZE " << holder_hole_size/mm << " mm" << std::endl;

  // CREATE ARRAY
  auto max_pos     = (n_sipms_ - 1) / 2.0 * pitch_;
  auto sipm_z      = fiber_length_ + sipm_thick/2;
  auto fiber_z     = fiber_length_/2 - tpb_thickness_;
  auto fiber_tpb_z = -tpb_thickness_/2;

  std::vector<G4ThreeVector> sipm_poss;
  auto holder_holes = new G4MultiUnion("holes");
  auto max_idx      = (n_sipms_ - 1) / 2;
  auto copy_no      = 0;

  for   (auto i=-max_idx; i<=max_idx; ++i) {
    for (auto j=-max_idx; j<=max_idx; ++j) {
      auto x = pitch_ * i;
      auto y = pitch_ * j;
      sipm_poss.emplace_back(x, y, sipm_z);

      auto  sipm_phys = PLACE(x, y,  sipm_z,  sipm_logic,  "sipm", gas_logic, copy_no);
      auto fiber_phys = PLACE(x, y, fiber_z, fiber_logic, "fiber", gas_logic, copy_no);

      holder_holes -> AddNode(*holder_hole, G4Translate3D(x, y, 0));

      copy_no++;
    }
  }
  holder_holes -> Voxelize();

  G4LogicalVolume* holder_logic     = nullptr;
  G4LogicalVolume* holder_tpb_logic = nullptr;
  if (with_holder_) {
    auto holder_z     = d_fiber_holder_ > 0 ? -holder_thick/2 : holder_thick/2;
    auto holder_solid = new G4SubtractionSolid("fiber_holder", holder_full, holder_holes);
    /**/ holder_logic = new G4LogicalVolume(holder_solid, ptfe, "fiber_holder");
    new G4LogicalSkinSurface("holder_surface", holder_logic, ptfe_surface);
    PLACE_Z(holder_z, holder_logic, "fiber_holder", gas_logic);

    if (with_holder_tpb_) {
      auto holder_tpb_z     = -d_fiber_holder_ - tpb_thickness_/2;
      auto holder_tpb_full  = new G4Tubs("fibers_holder_full", 0, tracking_plane_r, tpb_thickness_/2, 0, TWO_PI);
      auto holder_tpb_solid = new G4SubtractionSolid("fibers_holder_tpb", holder_tpb_full, holder_holes);
      /**/ holder_tpb_logic = new G4LogicalVolume(holder_tpb_solid, tpb, "fiber_holder_tpb");
      PLACE_Z(holder_tpb_z, holder_tpb_logic, "fiber_holder_tpb", gas_logic);
    }
  }

  gas_logic                              -> SetVisAttributes(G4VisAttributes::GetInvisible());
  fiber_logic                            -> SetVisAttributes(G4VisAttributes::GetInvisible());
  gas_pad_logic                          -> SetVisAttributes(G4VisAttributes::GetInvisible());
  refl_logic                             -> SetVisAttributes(G4Color{.2, .2, .2, .2});
  sipm_logic                             -> SetVisAttributes(new G4VisAttributes(G4Color::Red   ()));
  core_logic                             -> SetVisAttributes(new G4VisAttributes(G4Color::Yellow()));
  fiber_tpb_logic                        -> SetVisAttributes(new G4VisAttributes(G4Color::Blue  ()));
  if (with_holder_)     holder_logic     -> SetVisAttributes(new G4VisAttributes(G4Color::White ()));
  if (with_holder_ &&
      with_holder_tpb_) holder_tpb_logic -> SetVisAttributes(new G4VisAttributes(G4Color::Blue  ()));


  // SENSITIVE DETECTORS
  auto square_fiber_sd = new SquareFiberSD("square_fiber", sipm_output_file_, tpb_output_file_);
  G4SDManager::GetSDMpointer() -> AddNewDetector(square_fiber_sd);

  fiber_tpb_logic -> SetSensitiveDetector(square_fiber_sd);

  if (with_holder_tpb_ && holder_tpb_logic) {
    holder_tpb_logic -> SetSensitiveDetector(square_fiber_sd);
  }
  sipm_logic -> SetSensitiveDetector(square_fiber_sd);

  this -> SetELzCoord(-d_fiber_holder_ - d_anode_holder_ - el_gap_length_);
  this -> SetELWidth(el_gap_length_);
} // Construct()



G4ThreeVector SquareOpticalFiber::GenerateVertex(const G4String& region) const {
  if (region == "FIBER_ENTRY") { return {0, 0, 1*nm}; }
  if (region == "AD_HOC"     ) { return specific_vertex_; }
  if (region == "EL"         ) {
    // Unit cell
    auto x = (G4UniformRand() - 0.5) * pitch_;
    auto y = (G4UniformRand() - 0.5) * pitch_;
    return {x, y, GetELzCoord()};
  }

  auto x = (G4UniformRand() - 0.5) * sipm_size_;
  auto y = (G4UniformRand() - 0.5) * sipm_size_;
  auto eps = 10 * nm;

  if (region == "TPB_ENTRY_INSIDE" ) { return {x, y, -tpb_thickness_ + eps}; }
  if (region == "TPB_ENTRY_OUTSIDE") { return {x, y, -tpb_thickness_ - eps}; }
  if (region == "TPB_MIDDLE_LAYER" ) { return {x, y, -tpb_thickness_ / 2  }; }

  G4Exception("[SquareOpticalFiber]", "GenerateVertex()",
              FatalException, "Unknown vertex generation region!");
  return {};
} // GenerateVertex


} // close namespace

#undef TWO_PI
#undef CHECK_OVLP
#undef PLACE
#undef PLACE_Z
#undef PLACE_ORG
