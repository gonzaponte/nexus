// ----------------------------------------------------------------------------
// nexus | JustPrintEventAction.cc
//
// This is the JustPrint event action of the NEXT simulations. Only events with
// deposited energy larger than 0 are saved in the nexus output file.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#include "JustPrintEventAction.h"
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

REGISTER_CLASS(JustPrintEventAction, G4UserEventAction)

  JustPrintEventAction::JustPrintEventAction():
  G4UserEventAction(), print_mod_(1), current_evt_(0)
  {}



  JustPrintEventAction::~JustPrintEventAction()
  {}



  void JustPrintEventAction::BeginOfEventAction(const G4Event* /*event*/)
  {
    // Print out event number info
    if (current_evt_ % print_mod_ == 0) {
      G4cout << " >> Event no. " << std::setw(10) << std::right << current_evt_ << G4endl;
      if (current_evt_  == 10 * print_mod_)
        print_mod_ *= 10;
    }
  }

  void JustPrintEventAction::EndOfEventAction(const G4Event* event)
  {
    current_evt_++;
  }

} // end namespace nexus
