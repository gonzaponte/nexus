// ----------------------------------------------------------------------------
// nexus | DefaultEventAction.cc
//
// This is the default event action of the NEXT simulations. Only events with
// deposited energy larger than 0 are saved in the nexus output file.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "DefaultEventAction.h"
#include "Trajectory.h"
#include "PersistencyManager.h"
#include "IonizationHit.h"
#include "FactoryBase.h"

#include <G4Event.hh>
#include <G4VVisManager.hh>
#include <G4Trajectory.hh>
#include <G4GenericMessenger.hh>
#include <G4HCofThisEvent.hh>
#include <G4SDManager.hh>
#include <G4HCtable.hh>
#include <globals.hh>


namespace nexus {

REGISTER_CLASS(DefaultEventAction, G4UserEventAction)

  DefaultEventAction::DefaultEventAction():
    G4UserEventAction(), nevt_(0), nupdate_(10), energy_min_(0.), energy_max_(DBL_MAX)
  {
    msg_ = new G4GenericMessenger(this, "/Actions/DefaultEventAction/");

    G4GenericMessenger::Command& thresh_cmd =
       msg_->DeclareProperty("min_energy", energy_min_,
                             "Minimum deposited energy to save the event to file.");
    thresh_cmd.SetParameterName("min_energy", true);
    thresh_cmd.SetUnitCategory("Energy");
    thresh_cmd.SetRange("min_energy>0.");

    G4GenericMessenger::Command& max_energy_cmd =
      msg_->DeclareProperty("max_energy", energy_max_,
                            "Maximum deposited energy to save the event to file.");
    max_energy_cmd.SetParameterName("max_energy", true);
    max_energy_cmd.SetUnitCategory("Energy");
    max_energy_cmd.SetRange("max_energy>0.");

    PersistencyManager* pm = dynamic_cast<PersistencyManager*>
      (G4VPersistencyManager::GetPersistencyManager());

    pm->SaveNumbOfInteractingEvents(true);
  }



  DefaultEventAction::~DefaultEventAction()
  {
  }



  void DefaultEventAction::BeginOfEventAction(const G4Event* /*event*/)
  {
    // Print out event number info
    G4cout << " >> Event no. " << nevt_++ << G4endl;
  }



  void DefaultEventAction::EndOfEventAction(const G4Event* event)
  {
  }


} // end namespace nexus
