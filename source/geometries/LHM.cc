// -----------------------------------------------------------------------------
// nexus | LHM.cc
//
//
//
// The NEXT Collaboration
// -----------------------------------------------------------------------------

#include "LHM.h"
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

  REGISTER_CLASS(LHM, GeometryBase)

  using namespace CLHEP;

  LHM::LHM():
    GeometryBase(),
    thgem_diam_(32   * mm),
      csi_diam_(20   * mm),
      pmt_size_(20.0 * mm),
      pmt_gap_ (13.3 * mm)
  {
  }

  LHM::~LHM()
  {
  }

  void LHM::Construct()
  {
    G4double thgem_thickness = 0.4 * mm;
    G4double thgem_r         = thgem_diam_ / 2.0;
    G4double outer_r         = thgem_diam_ / 2.0 + 5.0 * mm;

    G4Box * world_solid = new G4Box ("world", 1 * m, 1 * m, 1 * m);
    G4Tubs* thgem_solid = new G4Tubs( "thgem"
                                    , 0, thgem_r
                                    , thgem_thickness / 2.0
                                    , 0, 360 * deg);
    G4Tubs* teflon_walls_solid = new G4Tubs( "teflonwalls"
                                           , thgem_r, outer_r
                                           , pmt_gap_ / 2.
                                           , 0, 360 * deg);
    G4Tubs* teflon_house_solid = new G4Tubs( "teflonholder"
                                           , 0, outer_r
                                           , pmt_size_/2
                                           , 0, 360 * deg);
    G4Box * pmt_solid = new G4Box ("pmt", pmt_size_/2, pmt_size_/2, pmt_size_/2);

    G4SubtractionSolid *teflon_pmt_solid = new G4SubtractionSolid( "teflonpmt"
                                                                 , teflon_house_solid
                                                                 , pmt_solid);

    auto gas = materials::GXe(2 * bar, 170 * kelvin);
    gas->SetMaterialPropertiesTable(opticalprops::GXe( 2. * bar
						                                         , 170. * kelvin
						                                         , 1. / MeV
                                                     , 1 * second));

    auto teflon = G4NistManager::Instance()->FindOrBuildMaterial("G4_TEFLON");
    teflon->SetMaterialPropertiesTable(new G4MaterialPropertiesTable());
//    teflon->SetMaterialPropertiesTable(opticalprops::PTFE());
//    teflon->SetMaterialPropertiesTable(opticalprops::AdHoc(0.94, 1 * nm));

    auto lead =  G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb");
//    lead->SetMaterialPropertiesTable(opticalprops::AdHoc(1e-6, 1 * nm));

    auto gold =  G4NistManager::Instance()->FindOrBuildMaterial("G4_Au");
//    gold->SetMaterialPropertiesTable(opticalprops::AdHoc(0.2, 1 * nm));

    G4LogicalVolume* world_logic = new G4LogicalVolume( world_solid
                                                      , gas
                                                      , "world");

    G4LogicalVolume* thgem_logic = new G4LogicalVolume( thgem_solid
                                                      , gold
                                                      , "thgem");

    G4LogicalVolume* teflon_walls_logic = new G4LogicalVolume( teflon_walls_solid
                                                             , teflon
                                                             , "teflon_walls");

    G4LogicalVolume* teflon_pmt_logic = new G4LogicalVolume( teflon_pmt_solid
                                                           , teflon
                                                           , "teflon_pmt");

    G4LogicalVolume* pmt_logic = new G4LogicalVolume( pmt_solid
                                                    , lead
                                                    , "pmt");

    G4OpticalSurface* teflon_surf = new G4OpticalSurface("teflon_surface"
                                                        , unified
                                                        , ground
                                                        , dielectric_metal
                                                        );
    teflon_surf->SetMaterialPropertiesTable(opticalprops::PTFE());

           world_logic->SetVisAttributes(G4VisAttributes::GetInvisible());
           thgem_logic->SetVisAttributes(nexus::Yellow());
    teflon_walls_logic->SetVisAttributes(nexus::White());
      teflon_pmt_logic->SetVisAttributes(nexus::White());
             pmt_logic->SetVisAttributes(nexus::DarkGrey());

    this->SetLogicalVolume(world_logic);

    auto zero       = G4ThreeVector(0., 0., 0.);
    auto z          = G4ThreeVector(0., 0., 1.);
    auto  thgem_pos = zero              - 0.5 * thgem_thickness * z;
    auto    csi_pos = zero              + 2.0 * nm              * z;
    auto teflon_pos = zero              + 0.5 * pmt_gap_        * z;
    auto    pmt_pos = zero + (pmt_gap_  + 0.5 * pmt_size_     ) * z;

    new G4PVPlacement(0,  thgem_pos, thgem_logic       , "thgem" , world_logic, false, 0, false);
    new G4PVPlacement(0, teflon_pos, teflon_walls_logic, "walls" , world_logic, false, 0, false);
    new G4PVPlacement(0,    pmt_pos, teflon_pmt_logic  , "holder", world_logic, false, 0, false);
    new G4PVPlacement(0,    pmt_pos, pmt_logic         , "pmt"   , world_logic, false, 0, false);

    new G4LogicalSkinSurface( "walls_surface", teflon_walls_logic, teflon_surf);
    new G4LogicalSkinSurface("holder_surface", teflon_pmt_logic  , teflon_surf);

    source_ = new CylinderPointSampler2020( 0, csi_diam_/2
                                          , 1 * nm
                                          , 0., twopi
                                          , nullptr
                                          , csi_pos);
  }

  G4ThreeVector LHM::GenerateVertex(const G4String& region) const
  {
    return source_->GenerateVertex(region);
  }

}
