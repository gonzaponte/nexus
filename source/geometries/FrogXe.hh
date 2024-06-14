// ----------------------------------------------------------------------------
// nexus | FrogXe.cc
//

// The NEXT Collaboration
// ----------------------------------------------------------------------------

#ifndef BLACK_BOX_H
#define BLACK_BOX_H

#include "GeometryBase.h"

namespace nexus {

  class FrogXe: public GeometryBase
  {
  public:
    /// Constructor
    FrogXe();
    /// Destructor
    ~FrogXe();

    /// Return vertex within region <region> of the chamber
    G4ThreeVector GenerateVertex(const G4String& region) const;

    void Construct();
  };

} // end namespace nexus

#endif
