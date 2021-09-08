// ----------------------------------------------------------------------------
// nexus | B10mGenerator.h
//
// This class is the primary generator for the decay chain
// of a source containing B10.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#ifndef B10_GENERATOR_H
#define B10_GENERATOR_H

#include <vector>
#include <G4VPrimaryGenerator.hh>

class G4Event;
class G4ParticleDefinition;
class G4GenericMessenger;

namespace nexus {

  class GeometryBase;


  class B10Generator: public G4VPrimaryGenerator
  {
  public:
    //Constructor
    B10Generator();
    //Destructor
    ~B10Generator();

    void GeneratePrimaryVertex(G4Event* evt);

  private:

    G4GenericMessenger* msg_;
    const GeometryBase* geom_;

    G4bool simulate_xrays_;

    G4String region_;
    G4ParticleDefinition* gammadef_;
  };

}// end namespace nexus
#endif
