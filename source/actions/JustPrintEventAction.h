// ----------------------------------------------------------------------------
// nexus | JustPrintEventAction.h
//
// This is the JustPrint event action of the NEXT simulations. Only events with
// deposited energy larger than 0 are saved in the nexus output file.
//
// The NEXT Collaboration
// ----------------------------------------------------------------------------

#ifndef JUSTPRINT_EVENT_ACTION_H
#define JUSTPRINT_EVENT_ACTION_H

#include <G4UserEventAction.hh>
#include <globals.hh>

class G4Event;
class G4GenericMessenger;

namespace nexus {

  /// This class is a general-purpose event run action.

  class JustPrintEventAction: public G4UserEventAction
  {
  public:
    /// Constructor
    JustPrintEventAction();
    /// Destructor
    ~JustPrintEventAction();

    /// Hook at the beginning of the event loop
    void BeginOfEventAction(const G4Event*);
    /// Hook at the end of the event loop
    void EndOfEventAction(const G4Event*);

  private:
    G4int print_mod_, current_evt_;
  };

} // namespace nexus

#endif
