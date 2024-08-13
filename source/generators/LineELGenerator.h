// ----------------------------------------------------------------------------
// nexus | LineELGenerator.h
//
// This class is the primary generator for events consisting of a
// photons emitted along a line in the EL region. The user must
// specify via configuration parameters the number of photons, the
// smearing in x,y, and photon energy.
// ----------------------------------------------------------------------------

#ifndef LINE_EL_GENERATOR_H
#define LINE_EL_GENERATOR_H

#include <G4VPrimaryGenerator.hh>

class G4GenericMessenger;
class G4Event;
class G4ParticleDefinition;


namespace nexus {

  class GeometryBase;

  class LineELGenerator: public G4VPrimaryGenerator
  {
  public:
    /// Constructor
    LineELGenerator();
    /// Destructor
    ~LineELGenerator();

    /// This method is invoked at the beginning of the event. It sets
    /// a primary vertex (that is, a particle in a given position and time)
    /// in the event.
    void GeneratePrimaryVertex(G4Event*);

  private:

    void SetParticleDefinition();


  private:
    G4GenericMessenger* msg_;

    G4ParticleDefinition* photon_;

    const GeometryBase* geom_; ///< Pointer to the detector geometry

    G4String region_;

    G4double energy_min_;
    G4double energy_max_;

    G4double costheta_min_;
    G4double costheta_max_;

    G4double sigma_x_;
    G4double sigma_y_;

    G4int n_photons_;
  };

} // end namespace nexus

#endif
