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
#include <G4Box.hh>
#include <G4Tubs.hh>
#include <G4VisAttributes.hh>


namespace nexus {

  REGISTER_CLASS(LHM, GeometryBase)

  using namespace CLHEP;

  LHM::LHM():
    GeometryBase(),
    thgem_diam_(36   * mm),
      csi_diam_(20   * mm),
      pmt_size_(25.4 * mm),
      pmt_gap_ (12.5 * mm)
  {
  }

  LHM::~LHM()
  {
  }

  void LHM::Construct()
  {
    G4Box * world_solid = new G4Box ("world", 1 * m, 1 * m, 1 * m);
    G4Tubs* thgem_solid = new G4Tubs( "thgem"
                                    , 0, thgem_diam_ * 1.1
                                    , 1 * mm
                                    , 0, 360 * deg);
    G4Tubs* teflon_walls_solid = new G4Tubs( "teflonwalls"
                                           , thgem_diam_, thgem_diam_ * 1.1
                                           , pmt_gap_
                                           , 0, 360 * deg);
    G4Tubs* teflon_house_solid = new G4Tubs( "teflonholder"
                                           , 0, thgem_diam_ * 1.1
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

    G4LogicalVolume* world_logic = new G4LogicalVolume( world_solid
                                                      , gas
                                                      , "world");
    this->SetLogicalVolume(world_logic);

    G4LogicalVolume* thgem_logic = new G4LogicalVolume( thgem_solid
                                                      , G4NistManager::Instance()->FindOrBuildMaterial("G4_Au")
                                                      , "thgem");

    G4LogicalVolume* teflon_walls_logic = new G4LogicalVolume( teflon_walls_solid
                                                             , G4NistManager::Instance()->FindOrBuildMaterial("G4_TEFLON")
                                                             , "teflon_walls");

    G4LogicalVolume* teflon_pmt_logic = new G4LogicalVolume( teflon_pmt_solid
                                                           , G4NistManager::Instance()->FindOrBuildMaterial("G4_TEFLON")
                                                           , "teflon_pmt");

    G4LogicalVolume* pmt_logic = new G4LogicalVolume( pmt_solid
                                                    , G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb")
                                                    , "pmt");

           world_logic->SetVisAttributes(G4VisAttributes::GetInvisible());
           thgem_logic->SetVisAttributes(nexus::Yellow());
    teflon_walls_logic->SetVisAttributes(nexus::White());
      teflon_pmt_logic->SetVisAttributes(G4VisAttributes::GetInvisible());
             pmt_logic->SetVisAttributes(nexus::DarkGrey());

    auto zero       = G4ThreeVector(0., 0., 0.);
    auto z          = G4ThreeVector(0., 0., 1.);
    auto thgem_pos  = zero              - 0.5 * 1.0 * mm   * z;
    auto teflon_pos = zero              + 0.5 * pmt_gap_   * z;
    auto    pmt_pos = zero + (pmt_gap_  + 0.5 * pmt_size_) * z;

    new G4PVPlacement(0,       zero, thgem_logic       , "world" , world_logic, false, 0, false);
    new G4PVPlacement(0, teflon_pos, teflon_walls_logic, "walls" , world_logic, false, 0, false);
    new G4PVPlacement(0,    pmt_pos, teflon_pmt_logic  , "holder", world_logic, false, 0, false);
    new G4PVPlacement(0,    pmt_pos, pmt_logic         , "pmt"   , world_logic, false, 0, false);

    source_ = new CylinderPointSampler2020( 0, csi_diam_/2
                                          , 2 * nm
                                          , 0., twopi
                                          , nullptr
                                          , zero + 1 * nm * z);
  }

  G4ThreeVector LHM::GenerateVertex(const G4String& region) const
  {
    return source_->GenerateVertex(region);
  }

}
