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
    wall_pos_(83.4 / 2. * mm),
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

    G4SubtractionSolid* peek_stand_solid = new G4SubtractionSolid( "peek_stand"
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

    auto plexiglass = G4NistManager::Instance()->FindOrBuildMaterial("G4_PLEXIGLASS");
    auto peek       = materials::PEEK();
    auto lead       = G4NistManager::Instance()->FindOrBuildMaterial("G4_LEAD");
    auto plastic    = G4NistManager::Instance()->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
    auto steel      = materials::Steel();
    auto teflon     = G4NistManager::Instance()->FindOrBuildMaterial("G4_TEFLON");

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
    this->SetLogicalVolume(world_logic);

    G4LogicalVolume* wall_logic = new G4LogicalVolume( wall_solid
                                                     , teflon
                                                     , "wall");

    G4LogicalVolume* fibers_stopper_logic = new G4LogicalVolume( fibers_stopper_solid
                                                               , plexiglass
                                                               , "fibers_stopper");

    G4LogicalVolume* peek_stand_logic = new G4LogicalVolume( peek_stand_solid
                                                           , peek
                                                           , "peek_stand");

    G4LogicalVolume* floor_logic = new G4LogicalVolume( floor_solid
                                                      , teflon
                                                      , "floor");

    G4LogicalVolume* ceiling_logic = new G4LogicalVolume( ceiling_solid
                                                        , teflon
                                                        , "ceiling");

    G4LogicalVolume* vuv_pmt_logic = new G4LogicalVolume( vuv_pmt_solid
                                                        , lead
                                                        , "vuv_pmt");

    G4LogicalVolume* red_pmt_logic = new G4LogicalVolume( red_pmt_solid
                                                        , lead
                                                        , "red_pmt");

    G4LogicalVolume* scintillator_logic = new G4LogicalVolume( vuv_pmt_solid
                                                             , plastic
                                                             , "scintillator");

    G4LogicalVolume* source_logic = new G4LogicalVolume( source_solid
                                                       , steel
                                                       , "source");


    //////////////////////////////////////////
    // ROTATIONS
    //////////////////////////////////////////
    G4RotationMatrix* flip            = new G4RotationMatrix{}; flip->rotateZ(180.0 * deg);
    G4RotationMatrix* rotate_z_cwise  = new G4RotationMatrix{}; flip->rotateZ(-90.0 * deg);
    G4RotationMatrix* rotate_z_ccwise = new G4RotationMatrix{}; flip->rotateZ( 90.0 * deg);

    //////////////////////////////////////////
    // PLACEMENTS
    //////////////////////////////////////////
    G4ThreeVector zero = G4ThreeVector{0.0, 0.0, 0.0};
    G4ThreeVector x    = G4ThreeVector{1.0, 0.0, 0.0};
    G4ThreeVector y    = G4ThreeVector{0.0, 1.0, 0.0};
    G4ThreeVector z    = G4ThreeVector{0.0, 0.0, 1.0};

    G4ThreeVector      floor_pos   = - floor_thickness_/2. * z;
    G4ThreeVector     source_pos   = - floor_thickness_/2. * z;
    G4ThreeVector  wall_left_pos   = -           wall_pos_ * x;
    G4ThreeVector wall_right_pos   =             wall_pos_ * x;
    G4ThreeVector wall_front_pos   = -           wall_pos_ * y;
    G4ThreeVector  wall_back_pos   =             wall_pos_ * y;
    G4ThreeVector    ceiling_pos   = ( fibers_stopper_height2_
                                     + fibers_stopper_gap_
                                     + fibers_stopper_height_ / 2.) * z;
    G4ThreeVector scintillator_pos = (source_thickness_ + scintillator_thickness_ / 2) * z;

    new G4PVPlacement(nullptr, zero     , world_logic, "world",     nullptr, false, 0, false);
    new G4PVPlacement(nullptr, floor_pos, floor_logic, "floor", world_logic, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);
    // new G4PVPlacement(nullptr, zero, world_logic, "world", nullptr, false, 0, false);


    //////////////////////////////////////////
    // VISUALS
    //////////////////////////////////////////
             world_logic->SetVisAttributes(G4VisAttributes::GetInvisible());
              wall_logic->SetVisAttributes(nexus::White());
    fibers_stopper_logic->SetVisAttributes(nexus::YellowAlpha());
        peek_stand_logic->SetVisAttributes(nexus::Brown());
             floor_logic->SetVisAttributes(nexus::White());
           ceiling_logic->SetVisAttributes(nexus::White());
           vuv_pmt_logic->SetVisAttributes(nexus::DarkGrey());
           red_pmt_logic->SetVisAttributes(nexus::Red());
      scintillator_logic->SetVisAttributes(nexus::BlueAlpha());
            source_logic->SetVisAttributes(nexus::TitaniumGrey());

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
