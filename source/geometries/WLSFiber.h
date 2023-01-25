// -----------------------------------------------------------------------------
// nexus | wls_fiber.h
//
//
//
// The NEXT Collaboration
// -----------------------------------------------------------------------------

#ifndef WLSFIBER_H
#define WLSFIBER_H

#include "GeometryBase.h"

namespace nexus {class CylinderPointSampler2020;}

class G4LogicalVolume;
class G4GenericMessenger;

namespace nexus {

  class WLSFiber: public GeometryBase {
  public:
    /// Constructor
    WLSFiber(G4double, G4double);

    /// Destructor
    ~WLSFiber();

    /// Generate a vertex within a given region of the geometry
    G4ThreeVector GenerateVertex(const G4String& region) const;

    /// Builder
    void Construct();

  private:
    // Dimensions
    G4double length_;
    G4double diameter_;
  };

}

#endif
