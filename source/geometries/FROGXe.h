// -----------------------------------------------------------------------------
// nexus | FROGXe.h
//
//
//
// The NEXT Collaboration
// -----------------------------------------------------------------------------

#ifndef FROGXE_H
#define FROGXE_H

#include "GeometryBase.h"

namespace nexus {class CylinderPointSampler2020;}

class G4LogicalVolume;
class G4GenericMessenger;

namespace nexus {

  class FROGXe: public GeometryBase {
  public:
    /// Constructor
    FROGXe();

    /// Destructor
    ~FROGXe();

    /// Generate a vertex within a given region of the geometry
    G4ThreeVector GenerateVertex(const G4String& region) const;

    /// Builder
    void Construct();

  private:

    G4int    fibers_per_wall_;
    G4double fiber_diam_;

    G4double source_thickness_;
    G4double source_diam_;

    G4double scintillator_thickness_;
    G4double scintillator_size_;

    G4double floor_size_;
    G4double floor_thickness_;

    G4double ceiling_size_;
    G4double ceiling_thickness_;

    G4double peek_stand_diam_;
    G4double peek_stand_height_;
    G4double peek_stand_pos_;

    G4double wall_thickness_;
    G4double wall_height_;
    G4double wall_width_;
    G4double wall_pos_;

    G4double vuv_pmt_size_;
    G4double vuv_pmt_pos_;

    G4bool   acrylic_plates_;
    G4double acrylic_plate_thickness_;
    G4double acrylic_plate_height_;
    G4double acrylic_plate_width_;

    G4double fibers_stopper_thickness_;
    G4double fibers_stopper_height_;
    G4double fibers_stopper_width_;
    G4double fibers_stopper_height1_;
    G4double fibers_stopper_height2_;
    G4double fibers_stopper_gap_;

    G4String medium_;

    CylinderPointSampler2020* source_;

  };

}

#endif
