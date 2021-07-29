// ----------------------------------------------------------------------------
// nexus | Irrad.h
//
// Co irrad at WIS
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#ifndef CO_IRRAD_H
#define CO_IRRAD_H

#include "BaseGeometry.h"

class G4GenericMessenger;

namespace nexus {

  class Irrad : public BaseGeometry
  {
  public:
    ///Constructor
    Irrad();

    ///Destructor
    ~Irrad(){}

    void Construct();

    G4ThreeVector GenerateVertex(const G4String& region) const;

  private:

    // Messenger for the definition of control commands
    G4GenericMessenger* msg_;
    G4bool   visibility_;
    G4double source_length_;
    G4double source_diam_  ;
    G4double ss_diam_;
    G4double gap_to_crystal_;
    G4double crystal_size_;
    G4String crystal_type_;
  };
}

#endif
