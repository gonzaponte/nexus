// ----------------------------------------------------------------------------
// nexus | Next100Meshes.h
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#ifndef NEXT100_MESHES_H
#define NEXT100_MESHES_H

#include "GeometryBase.h"

class G4GenericMessenger;

namespace nexus {
  class CylinderPointSampler;

  class Next100Meshes: public GeometryBase
  {
  public:
    Next100Meshes();
    ~Next100Meshes();

    G4ThreeVector GenerateVertex(const G4String& region) const;

  private:
    void Construct();

    const G4double el_diam_;
    const G4double el_thick_;
    const G4double hex_indiam_;
    const G4double el_gap_length_;

    G4double rotation_;
    G4double reflectivity_;

    CylinderPointSampler* sampler_;
    // Messenger for the definition of control commands
    G4GenericMessenger* msg_;
  };

} //end namespace nexus

#endif
