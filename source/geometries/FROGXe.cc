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
#include "WLSFiber.h"
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
    source_thickness_(1.0 * mm),
    source_diam_(0.3 * 25.4 * mm),
    scintillator_thickness_(1.0 * mm),
    scintillator_size_(10.0 * mm),
    floor_thickness_(6.0 * mm),
    floor_size_(91.0 * mm),
    ceiling_thickness_(30.2 * mm),
    ceiling_size_(74.0 * mm),
    peek_stand_diam_(8.0 * mm),
    peek_stand_height_(60.0 * mm),
    peek_stand_pos_(64.4 / 2.0 * mm),
    wall_thickness_(8.0 * mm),
    wall_height_(95.0 * mm),
    wall_pos_(83.4 / 2. * mm),
    wall_width_(75.0 * mm),
    vuv_pmt_size_(20.5 * mm),
    vuv_pmt_pos_(27.7 / 2. * mm),
    acrylic_plate_thickness_(2.0 * mm),
    acrylic_plate_height_(63.0 * mm),
    acrylic_plate_width_(59.0 * mm),
    fibers_stopper_height1_(15.0 * mm), // distance to floor bottom set
    fibers_stopper_height2_(77.0 * mm), // distance to floor top set
    fibers_stopper_gap_(10.0 * mm), // distance within a set
    fibers_stopper_thickness_(3.0 * mm),
    fibers_stopper_height_(4.5 * mm),
    fibers_stopper_width_(74.5 * mm),
    medium_("Air"),
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

    G4double wall_gap_width = fibers_per_wall_ * fiber_diam_;
    G4ThreeVector gap_pos = G4ThreeVector{0., -wall_thickness_/2, 0.};
    G4Box* gap_in_wall = new G4Box( "gap_in_wall"
                                  , wall_gap_width / 2.
                                  , fiber_diam_
                                  , wall_height_   );

    G4SubtractionSolid* wall_solid = new G4SubtractionSolid( "wall"
                                                           , wall
                                                           , gap_in_wall
                                                           , nullptr
                                                           , gap_pos);



    G4Box* fibers_stopper_solid = new G4Box( "fibers_stopper"
                                           , fibers_stopper_width_ / 2.
                                           , fibers_stopper_thickness_ / 2.
                                           , fibers_stopper_height_ / 2.);



    G4Tubs* peek_stand = new G4Tubs( "full_peek_stand"
                                   , 0., peek_stand_diam_ / 2.
                                   , peek_stand_height_ / 2.
                                   , 0., 360. * deg);

    G4double gap = 1.0 * mm;
    G4Box* gap_in_peek = new G4Box( "gap_in_peek"
                                  , gap
                                  , gap
                                  , peek_stand_height_);

    G4ThreeVector gap_pos_x = G4ThreeVector{peek_stand_diam_/2. - gap, 0., 0.};
    G4ThreeVector gap_pos_y = G4ThreeVector{0., peek_stand_diam_/2. - gap, 0.};

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



    G4Box* ceiling_solid = new G4Box( "ceiling"
                                    , ceiling_size_      / 2.
                                    , ceiling_size_      / 2.
                                    , ceiling_thickness_ / 2.);



    G4Box* vuv_pmt_solid = new G4Box( "vuv_pmt"
                                    , vuv_pmt_size_      / 2.
                                    , vuv_pmt_size_      / 2.
                                    , ceiling_thickness_ / 2.);


    G4double red_pmt_thickness = ceiling_thickness_ / 4.;
    G4Box* red_pmt_solid = new G4Box( "red_pmt"
                                    , wall_gap_width    / 2.
                                    , fiber_diam_       / 2.
                                    , red_pmt_thickness / 2.);


    G4Box* scintillator_solid = new G4Box( "scintillator"
                                         , scintillator_size_      / 2.
                                         , scintillator_size_      / 2.
                                         , scintillator_thickness_ / 2.);



    G4Tubs* source_solid = new G4Tubs( "source"
                                     , 0., source_diam_ / 2.
                                     , source_thickness_ / 2.
                                     , 0., 360. * deg);




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
      gas->SetMaterialPropertiesTable(opticalprops::Vacuum());

    }
    else {
      G4Exception("[FROGXe]", "Construct()", FatalException, "Invalid medium!");
    }

    auto plexiglass = G4NistManager::Instance()->FindOrBuildMaterial("G4_PLEXIGLASS");
    auto peek       = materials::PEEK();
    auto lead       = G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb");
    auto plastic    = G4NistManager::Instance()->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
    auto steel      = materials::Steel();
    auto teflon     = G4NistManager::Instance()->FindOrBuildMaterial("G4_TEFLON");

    plexiglass->SetMaterialPropertiesTable(opticalprops::Plexiglass());
    peek      ->SetMaterialPropertiesTable(opticalprops::PEEK());
    plastic   ->SetMaterialPropertiesTable(opticalprops::BC404());
    steel     ->SetMaterialPropertiesTable(opticalprops::StainlessSteel());
    teflon    ->SetMaterialPropertiesTable(opticalprops::PTFE());

    G4OpticalSurface* teflon_surf = new G4OpticalSurface("teflon_surface"
                                                        , unified
                                                        , polished
                                                        , dielectric_metal
                                                        );
    teflon_surf->SetMaterialPropertiesTable(opticalprops::PTFE());

    G4OpticalSurface* peek_surf = new G4OpticalSurface("peek_surface"
                                                      , unified
                                                      , polished
                                                      , dielectric_metal
                                                      );
    peek_surf->SetMaterialPropertiesTable(opticalprops::PEEK());

    G4OpticalSurface* steel_surf = new G4OpticalSurface("steel_surface"
                                                       , unified
                                                       , polished
                                                       , dielectric_metal
                                                       );
    steel_surf->SetMaterialPropertiesTable(opticalprops::StainlessSteel());

    G4OpticalSurface* plexiglass_surf = new G4OpticalSurface("plexiglass_surface"
                                                            , unified
                                                            , polished
                                                            , dielectric_metal
                                                            );
    plexiglass_surf->SetMaterialPropertiesTable(plexiglass->GetMaterialPropertiesTable());

    G4OpticalSurface* scint_surf = new G4OpticalSurface("scint_surface"
                                                       , unified
                                                       , polished
                                                       , dielectric_metal
                                                       );
    scint_surf->SetMaterialPropertiesTable(opticalprops::BC404());

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

    G4LogicalVolume* scintillator_logic = new G4LogicalVolume( scintillator_solid
                                                             , plastic
                                                             , "scintillator");

    G4LogicalVolume* source_logic = new G4LogicalVolume( source_solid
                                                       , steel
                                                       , "source");

    WLSFiber* fiber = new WLSFiber(wall_height_, fiber_diam_);
    fiber->Construct();
    G4LogicalVolume* fiber_logic = fiber->GetLogicalVolume();

    //////////////////////////////////////////
    // ROTATIONS_
    //////////////////////////////////////////
    G4RotationMatrix* rotate_z_1 = new G4RotationMatrix{}; rotate_z_1->rotateZ( 90.0 * deg);
    G4RotationMatrix* rotate_z_2 = new G4RotationMatrix{}; rotate_z_2->rotateZ(180.0 * deg);
    G4RotationMatrix* rotate_z_3 = new G4RotationMatrix{}; rotate_z_3->rotateZ(270.0 * deg);

    //////////////////////////////////////////
    // PLACEMENTS
    //////////////////////////////////////////
    G4ThreeVector zero = G4ThreeVector{0.0, 0.0, 0.0};
    G4ThreeVector x    = G4ThreeVector{1.0, 0.0, 0.0};
    G4ThreeVector y    = G4ThreeVector{0.0, 1.0, 0.0};
    G4ThreeVector z    = G4ThreeVector{0.0, 0.0, 1.0};

    G4ThreeVector  floor_pos = - floor_thickness_/2. * z;
    G4ThreeVector source_pos =  source_thickness_/2. * z;
    G4ThreeVector  scint_pos = (source_thickness_ + scintillator_thickness_ / 2) * z;

    G4ThreeVector  wall_left_pos = -wall_pos_ * x + wall_height_ / 2. * z;
    G4ThreeVector wall_right_pos =  wall_pos_ * x + wall_height_ / 2. * z;
    G4ThreeVector wall_front_pos = -wall_pos_ * y + wall_height_ / 2. * z;
    G4ThreeVector  wall_back_pos =  wall_pos_ * y + wall_height_ / 2. * z;

    G4ThreeVector peek_0_pos = peek_stand_pos_ * ( x + y) + peek_stand_height_ / 2. * z;
    G4ThreeVector peek_1_pos = peek_stand_pos_ * ( x - y) + peek_stand_height_ / 2. * z;
    G4ThreeVector peek_2_pos = peek_stand_pos_ * (-x + y) + peek_stand_height_ / 2. * z;
    G4ThreeVector peek_3_pos = peek_stand_pos_ * (-x - y) + peek_stand_height_ / 2. * z;

    G4double stopper_pos      = wall_pos_ - (wall_thickness_ + fibers_stopper_thickness_) / 2.;
    G4double fibers_pos       = wall_pos_ - (wall_thickness_ - fiber_diam_) / 2.;
    G4double stopper_height_1 = fibers_stopper_height1_ + fibers_stopper_height_ / 2.;
    G4double stopper_height_2 = fibers_stopper_height2_ + fibers_stopper_height_ / 2.;
    G4ThreeVector stopper_0_pos = stopper_pos * ( x) + (stopper_height_1                      ) * z;
    G4ThreeVector stopper_1_pos = stopper_pos * ( x) + (stopper_height_2                      ) * z;
    G4ThreeVector stopper_2_pos = stopper_pos * ( y) + (stopper_height_1 + fibers_stopper_gap_) * z;
    G4ThreeVector stopper_3_pos = stopper_pos * ( y) + (stopper_height_2 + fibers_stopper_gap_) * z;
    G4ThreeVector stopper_4_pos = stopper_pos * (-x) + (stopper_height_1                      ) * z;
    G4ThreeVector stopper_5_pos = stopper_pos * (-x) + (stopper_height_2                      ) * z;
    G4ThreeVector stopper_6_pos = stopper_pos * (-y) + (stopper_height_1 + fibers_stopper_gap_) * z;
    G4ThreeVector stopper_7_pos = stopper_pos * (-y) + (stopper_height_2 + fibers_stopper_gap_) * z;

    G4ThreeVector   ceiling_pos = (fibers_stopper_height2_ + fibers_stopper_gap_ + fibers_stopper_height_ / 2. + ceiling_thickness_ / 2.) * z;
    G4ThreeVector vuv_pmt_0_pos = vuv_pmt_pos_ * ( x + y) - 1 * um * z;
    G4ThreeVector vuv_pmt_1_pos = vuv_pmt_pos_ * ( x - y) - 1 * um * z;
    G4ThreeVector vuv_pmt_2_pos = vuv_pmt_pos_ * (-x + y) - 1 * um * z;
    G4ThreeVector vuv_pmt_3_pos = vuv_pmt_pos_ * (-x - y) - 1 * um * z;

    G4ThreeVector red_pmt_0_pos = fibers_pos * ( x) + (wall_height_ + red_pmt_thickness / 2.) * z;
    G4ThreeVector red_pmt_1_pos = fibers_pos * (-x) + (wall_height_ + red_pmt_thickness / 2.) * z;
    G4ThreeVector red_pmt_2_pos = fibers_pos * ( y) + (wall_height_ + red_pmt_thickness / 2.) * z;
    G4ThreeVector red_pmt_3_pos = fibers_pos * (-y) + (wall_height_ + red_pmt_thickness / 2.) * z;


    new G4PVPlacement(   nullptr, zero                 ,          world_logic, "world"       ,       nullptr, false, 0, false);

    new G4PVPlacement(   nullptr, zero +      floor_pos,          floor_logic, "floor"       ,   world_logic, false, 0, false);
    new G4PVPlacement(   nullptr, zero +     source_pos,         source_logic, "source"      ,   world_logic, false, 0, false);
    new G4PVPlacement(   nullptr, zero +      scint_pos,   scintillator_logic, "scintillator",   world_logic, false, 0, false);

    new G4PVPlacement(   nullptr, zero +    ceiling_pos,        ceiling_logic, "ceiling"     ,   world_logic, false, 0, false);
    new G4PVPlacement(   nullptr, zero +  vuv_pmt_0_pos,        vuv_pmt_logic, "vuv_pmt_0"   , ceiling_logic,  true, 0, false);
    new G4PVPlacement(   nullptr, zero +  vuv_pmt_1_pos,        vuv_pmt_logic, "vuv_pmt_1"   , ceiling_logic,  true, 1, false);
    new G4PVPlacement(   nullptr, zero +  vuv_pmt_2_pos,        vuv_pmt_logic, "vuv_pmt_2"   , ceiling_logic,  true, 2, false);
    new G4PVPlacement(   nullptr, zero +  vuv_pmt_3_pos,        vuv_pmt_logic, "vuv_pmt_3"   , ceiling_logic,  true, 3, false);

    new G4PVPlacement(rotate_z_1, zero +  red_pmt_0_pos,        red_pmt_logic, "red_pmt_0"   ,   world_logic,  true, 0, false);
    new G4PVPlacement(rotate_z_1, zero +  red_pmt_1_pos,        red_pmt_logic, "red_pmt_1"   ,   world_logic,  true, 1, false);
    new G4PVPlacement(   nullptr, zero +  red_pmt_2_pos,        red_pmt_logic, "red_pmt_2"   ,   world_logic,  true, 2, false);
    new G4PVPlacement(   nullptr, zero +  red_pmt_3_pos,        red_pmt_logic, "red_pmt_3"   ,   world_logic,  true, 3, false);

    new G4PVPlacement(rotate_z_3, zero +  wall_left_pos,           wall_logic, "wall_left"   ,   world_logic, true , 0, false);
    new G4PVPlacement(rotate_z_1, zero + wall_right_pos,           wall_logic, "wall_right"  ,   world_logic, true , 1, false);
    new G4PVPlacement(rotate_z_2, zero + wall_front_pos,           wall_logic, "wall_front"  ,   world_logic, true , 2, false);
    new G4PVPlacement(   nullptr, zero +  wall_back_pos,           wall_logic, "wall_back"   ,   world_logic, true , 3, false);

    new G4PVPlacement(rotate_z_2, zero +     peek_0_pos,     peek_stand_logic, "peek_stand_0",   world_logic, true , 0, false);
    new G4PVPlacement(rotate_z_3, zero +     peek_1_pos,     peek_stand_logic, "peek_stand_1",   world_logic, true , 1, false);
    new G4PVPlacement(rotate_z_1, zero +     peek_2_pos,     peek_stand_logic, "peek_stand_2",   world_logic, true , 2, false);
    new G4PVPlacement(   nullptr, zero +     peek_3_pos,     peek_stand_logic, "peek_stand_3",   world_logic, true , 3, false);

    new G4PVPlacement(rotate_z_1, zero +  stopper_0_pos, fibers_stopper_logic, "stopper_0"   ,   world_logic, true , 0, false);
    new G4PVPlacement(rotate_z_1, zero +  stopper_1_pos, fibers_stopper_logic, "stopper_1"   ,   world_logic, true , 1, false);
    new G4PVPlacement(   nullptr, zero +  stopper_2_pos, fibers_stopper_logic, "stopper_2"   ,   world_logic, true , 2, false);
    new G4PVPlacement(   nullptr, zero +  stopper_3_pos, fibers_stopper_logic, "stopper_3"   ,   world_logic, true , 3, false);
    new G4PVPlacement(rotate_z_3, zero +  stopper_4_pos, fibers_stopper_logic, "stopper_4"   ,   world_logic, true , 4, false);
    new G4PVPlacement(rotate_z_3, zero +  stopper_5_pos, fibers_stopper_logic, "stopper_5"   ,   world_logic, true , 5, false);
    new G4PVPlacement(   nullptr, zero +  stopper_6_pos, fibers_stopper_logic, "stopper_6"   ,   world_logic, true , 6, false);
    new G4PVPlacement(   nullptr, zero +  stopper_7_pos, fibers_stopper_logic, "stopper_7"   ,   world_logic, true , 7, false);

    G4int k=0;
    G4ThreeVector alongs[4] = {x,x,y,y};
    G4ThreeVector planes[4] = {y,-y,x,-x};
    for (G4int j=0; j<4; ++j) {
      G4ThreeVector along = alongs[j];
      G4ThreeVector plane = planes[j];
      for (G4int i=0; i<fibers_per_wall_; ++i) {
        G4double x0 =  (i + 0.5 - fibers_per_wall_/ 2.) * fiber_diam_;
        G4ThreeVector fiber_pos = x0 * along + fibers_pos * plane + wall_height_/2. * z;
        new G4PVPlacement( nullptr, zero +      fiber_pos,          fiber_logic, "fiber"       ,   world_logic, true , k, false);
        ++k;
      }
    }


    //////////////////////////////////////////
    // Skins
    //////////////////////////////////////////
    new G4LogicalSkinSurface(     "walls_surface",           wall_logic,     teflon_surf);
    new G4LogicalSkinSurface(     "floor_surface",          floor_logic,     teflon_surf);
    new G4LogicalSkinSurface(   "ceiling_surface",        ceiling_logic,     teflon_surf);
    // new G4LogicalSkinSurface(     "scint_surface",   scintillator_logic,      scint_surf);
    new G4LogicalSkinSurface(      "peek_surface",     peek_stand_logic,       peek_surf);
    new G4LogicalSkinSurface("plexiglass_surface", fibers_stopper_logic, plexiglass_surf);
    new G4LogicalSkinSurface(     "steel_surface",         source_logic,      steel_surf);


    //////////////////////////////////////////
    // VISUALS
    //////////////////////////////////////////
             world_logic->SetVisAttributes(G4VisAttributes::GetInvisible());
              wall_logic->SetVisAttributes(nexus::WhiteAlpha());
              wall_logic->SetVisAttributes(G4VisAttributes::GetInvisible());
    fibers_stopper_logic->SetVisAttributes(nexus::YellowAlpha());
        peek_stand_logic->SetVisAttributes(nexus::Brown());
             floor_logic->SetVisAttributes(nexus::White());
           ceiling_logic->SetVisAttributes(nexus::White());
           vuv_pmt_logic->SetVisAttributes(nexus::DarkGrey());
           red_pmt_logic->SetVisAttributes(nexus::Red());
      scintillator_logic->SetVisAttributes(nexus::BlueAlpha());
            source_logic->SetVisAttributes(nexus::TitaniumGrey());


    source_ = new CylinderPointSampler2020( 0, source_diam_/2
                                          , scintillator_thickness_ / 2
                                          , 0., twopi
                                          , nullptr
                                          , scint_pos);
  }


  G4ThreeVector FROGXe::GenerateVertex(const G4String& region) const
  {
    return source_->GenerateVertex(region);
  }

}
