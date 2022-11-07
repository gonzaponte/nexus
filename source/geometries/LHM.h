// -----------------------------------------------------------------------------
// nexus | LHM.h
//
//
//
// The NEXT Collaboration
// -----------------------------------------------------------------------------

#ifndef LHM_H
#define LHM_H

#include "GeometryBase.h"

namespace nexus {class CylinderPointSampler2020;}

class G4LogicalVolume;
class G4GenericMessenger;

namespace nexus {

  class LHM: public GeometryBase {
  public:
    /// Constructor
    LHM();

    /// Destructor
    ~LHM();

    /// Generate a vertex within a given region of the geometry
    G4ThreeVector GenerateVertex(const G4String& region) const;

    /// Builder
    void Construct();

  private:
    // Dimensions
    G4double thgem_diam_;
    G4double   csi_diam_;
    G4double   pmt_size_;
    G4double   pmt_gap_ ;
    CylinderPointSampler2020* source_;

  };

}

#endif
