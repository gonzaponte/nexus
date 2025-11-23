#ifndef XETRAP_HH
#define XETRAP_HH

#include "GeometryBase.h"
#include "G4ThreeVector.hh"

class G4GenericMessenger;

namespace nexus {

    class XeTrap : public GeometryBase {

    public:
       XeTrap();
      ~XeTrap();

      void          Construct();

      G4GenericMessenger* msg_;

      // Controlled from macro
      G4double sipm_size_;
      G4double sipm_thickness_;
      G4double pmma_size_;
      G4double pmma_thickness_;
      G4double tpb_thickness_;
      G4double tpb_surface_roughness_;
      G4double reflector_reflectivity_;

      G4ThreeVector GenerateVertex(const G4String& region) const;
      };
}

#endif // XETRAP_H_
