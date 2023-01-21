// -----------------------------------------------------------------------------
// nexus | FiberBB.h
//
//
//
// The NEXT Collaboration
// -----------------------------------------------------------------------------

#ifndef FIBERBB_H
#define FIBERBB_H

#include "GeometryBase.h"

namespace nexus {

  class FiberBB: public GeometryBase {
  public:
    /// Constructor
    FiberBB();

    /// Destructor
    ~FiberBB();

    /// Generate a vertex within a given region of the geometry
    G4ThreeVector GenerateVertex(const G4String& region) const;

    /// Builder
    void Construct();

  private:
    // Dimensions
    G4double  scint_size_;
    G4double scint_thick_;
    G4double    pmt_size_;
    G4double    air_gap_;

  };

}

#endif
