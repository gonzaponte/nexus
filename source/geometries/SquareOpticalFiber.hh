#ifndef SQUARE_OPTICAL_FIBER_HH
#define SQUARE_OPTICAL_FIBER_HH

#include "GeometryBase.h"

class G4GenericMessenger;

namespace nexus {

    class SquareOpticalFiber : public GeometryBase {

    public:
      SquareOpticalFiber();
      ~SquareOpticalFiber();

      void          Construct();
      G4ThreeVector GenerateVertex(const G4String& region) const;

      G4GenericMessenger* msg_;


      // Controlled from macro
      G4ThreeVector specific_vertex_;
      G4double sipm_size_;
      G4double fiber_length_;
      G4double el_gap_length_;
      G4double pitch_;
      G4double d_fiber_holder_;
      G4double d_anode_holder_;
      G4double holder_thickness_;
      G4double tpb_thickness_;
      G4String sipm_output_file_;
      G4String tpb_output_file_;
      G4double diff_sigma_; // Standard deviation value for transverse diffusion
      G4int    n_sipms_; // per dimension

      G4bool with_holder_;
      G4bool with_fiber_tpb_;
      G4bool with_holder_tpb_;
    };
}

#endif
