// -----------------------------------------------------------------------------
// nexus | FROGXe.cc
//
//
//
// The NEXT Collaboration
// -----------------------------------------------------------------------------

#include "FROGXe.h"
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

  REGISTER_CLASS(FROGXe, GeometryBase)

  using namespace CLHEP;

  FROGXe::FROGXe():
    GeometryBase(),
    fibers_per_wall_(64),
    fiber_diam_(1.0 * mm),
    source_thickness_(2.0 * mm),
    source_diam_(0.3 * 25.4 * mm),
    scintillator_thickness_(1.0 * mm),
    scintillator_size_(10.0 * mm),
    floor_size_(91.0 * mm),
    floor_thickness_(6.0 * mm),
    ceiling_size_(74.0 * mm),
    ceiling_thickness_(30.2 * mm),
    peek_stand_diam_(8.0 * mm),
    peek_stand_height_(60.0 * mm),
    wall_thickness_(8.0 * mm),
    wall_height_(95.0 * mm),
    wall_width_(75.0 * mm),
    vuv_pmt_size_(20.5 * mm),
    vuv_pmt_thickness_(30.0 * mm),
    red_pmt_size_(20.5 * mm),
    red_pmt_thickness_(50.0 * mm),
    acrylic_plate_thickness_(2.0 * mm),
    acrylic_plate_height_(63.0 * mm),
    acrylic_plate_width_(59.0 * mm),
    fibers_stopper_height1_(15.0 * mm), // distance to floor bottom set
    fibers_stopper_height2_(77.0 * mm), // distance to floor top set
    fibers_stopper_gap_(10.0 * mm), // distance within a set
    fibers_stopper_thickness_(3.0 * mm),
    fibers_stopper_height_(4.5 * mm),
    fibers_stopper_width_(74.5 * mm),
    medium_(""),
    source_()
  {
  }

  FROGXe::~FROGXe()
  {
  }

  void FROGXe::Construct()
  {
    //////////////////////////////////////////
    // SOLIDS
    //////////////////////////////////////////
    G4Box* world_solid = new G4Box ("world", 1. * m, 1. * m, 1. * m);



    G4Box* wall = new G4Box( "full_wall"
                           , wall_width_     / 2.
                           , wall_thickness_ / 2.
                           , wall_height_    / 2.);

    G4double gap_width = fibers_per_wall_ * fiber_diam_;
    G4ThreeVector gap_pos = G4ThreeVector{0., -wall_thickness_/2 + fiber_diam_/2, 0.};
    G4Box* gap_in_wall = new G4Box( "gap_in_wall"
                                  , gap_width    / 2.
                                  , fiber_diam_  / 2.
                                  , wall_height_ / 2.);

    G4SubtractionSolid* wall_solid = new G4SubtractionSolid( "wall"
                                                           , wall
                                                           , gap_in_wall
                                                           , nullptr
                                                           , gap_pos);



    G4Box* fibers_stopper_solid = new G4Box( "fibers_stopper"
                                           , fibers_stopper_width_
                                           , fibers_stopper_thickness_
                                           , fibers_stopper_height_);



    G4Tubs* peek_stand = new G4Tubs( "full_peek_stand"
                                   , 0., peek_stand_diam_ / 2.
                                   , 0., 360. * deg
                                   , peek_stand_height_ / 2.);

    G4double gap = 1.0 * mm;
    G4Box* gap_in_peek = new G4Box( "gap_in_peek"
                                  , gap
                                  , gap
                                  , peek_stand_height_ / 2.);

    G4ThreeVector gap_pos_x = G4ThreeVector{ peek_stand_diam_/2. - gap
                                           , acrylic_plate_thickness_
                                           , peek_stand_height_};
    G4ThreeVector gap_pos_y = G4ThreeVector{ acrylic_plate_thickness_
                                           , peek_stand_diam_/2. - gap
                                           , peek_stand_height_};

    G4SubtractionSolid* peek_stand_inter = new G4SubtractionSolid( "peek_stand_solid_intermediate"
                                                                 , peek_stand
                                                                 , gap_in_peek
                                                                 , nullptr
                                                                 , gap_pos_x);

    G4SubtractionSolid* peek_stand_solid = new G4SubtractionSolid( "peek_stand_solid"
                                                                 , peek_stand_inter
                                                                 , gap_in_peek
                                                                 , nullptr
                                                                 , gap_pos_y);




    G4Box* floor_solid = new G4Box( "floor"
                                  , floor_size_      / 2.
                                  , floor_size_      / 2.
                                  , floor_thickness_ / 2.);



    G4Box* ceiling_solid = new G4Box( "full_ceiling"
                                    , ceiling_size_      / 2.
                                    , ceiling_size_      / 2.
                                    , ceiling_thickness_ / 2.);



    G4Box* vuv_pmt_solid = new G4Box( "vuv_pmt"
                                    , vuv_pmt_size_      / 2.
                                    , vuv_pmt_size_      / 2.
                                    , vuv_pmt_thickness_ / 2.);



    G4Box* red_pmt_solid = new G4Box( "red_pmt"
                                    , red_pmt_size_      / 2.
                                    , red_pmt_size_      / 2.
                                    , red_pmt_thickness_ / 2.);



    G4Box* scintillator_solid = new G4Box( "scintillator"
                                         , scintillator_size_      / 2.
                                         , scintillator_size_      / 2.
                                         , scintillator_thickness_ / 2.);



    G4Tubs* source_solid = new G4Tubs( "source"
                                     , 0., source_diam_ / 2.
                                     , 0., 360. * deg
                                     , source_thickness_ / 2.);




    //////////////////////////////////////////
    // MATERIALS
    //////////////////////////////////////////


    G4Material* gas;
    if (medium_ == "Xe") {
      gas = materials::GXe(2 * bar, 170 * kelvin);
      gas->SetMaterialPropertiesTable(opticalprops::GXe( 1.   * bar
                                                       , 273. * kelvin
                                                       , 1. / MeV
                                                       , 1 * second));
    }
    else if (medium_ == "Air") {
      gas = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
    }
    else {
      G4Exception("[FROGXe]", "Construct()", FatalException, "Invalid medium!");
    }

    auto teflon = G4NistManager::Instance()->FindOrBuildMaterial("G4_TEFLON");
    teflon->SetMaterialPropertiesTable(opticalprops::PTFE());

    G4OpticalSurface* teflon_surf = new G4OpticalSurface("teflon_surface"
                                                        , unified
                                                        , ground
                                                        , dielectric_metal
                                                        );
    teflon_surf->SetMaterialPropertiesTable(opticalprops::PTFE());


    //////////////////////////////////////////
    // LOGICAL
    //////////////////////////////////////////


    G4LogicalVolume* world_logic = new G4LogicalVolume( world_solid
                                                      , gas
                                                      , "world");

    world_logic->SetVisAttributes(G4VisAttributes::GetInvisible());

    this->SetLogicalVolume(world_logic);

    auto zero = G4ThreeVector(0., 0., 0.);
    auto x    = G4ThreeVector(1., 0., 0.);
    auto y    = G4ThreeVector(0., 1., 0.);
    auto z    = G4ThreeVector(0., 0., 1.);

    auto source_pos = zero + (source_thickness_ + scintillator_thickness_ / 2) * z;

    // new G4PVPlacement(0,  thgem_pos, thgem_logic       , "thgem" , world_logic, false, 0, false);
    // new G4PVPlacement(0, teflon_pos, teflon_walls_logic, "walls" , world_logic, false, 0, false);
    // new G4PVPlacement(0,    pmt_pos, teflon_pmt_logic  , "holder", world_logic, false, 0, false);
    // new G4PVPlacement(0,    pmt_pos, pmt_logic         , "pmt"   , world_logic, false, 0, false);
    //
    // new G4LogicalSkinSurface( "walls_surface", teflon_walls_logic, teflon_surf);
    // new G4LogicalSkinSurface("holder_surface", teflon_pmt_logic  , teflon_surf);

    source_ = new CylinderPointSampler2020( 0, source_diam_/2
                                          , scintillator_thickness_ / 2
                                          , 0., twopi
                                          , nullptr
                                          , source_pos);
  }


  G4ThreeVector FROGXe::GenerateVertex(const G4String& region) const
  {
    return source_->GenerateVertex(region);
  }

}
