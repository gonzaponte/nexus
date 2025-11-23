#!/usr/bin/env python3

init_template = """
/PhysicsList/RegisterPhysics G4EmStandardPhysics_option4
/PhysicsList/RegisterPhysics G4DecayPhysics
/PhysicsList/RegisterPhysics G4RadioactiveDecayPhysics
/PhysicsList/RegisterPhysics NexusPhysics
/PhysicsList/RegisterPhysics G4StepLimiterPhysics

/nexus/RegisterGeometry Next100
/nexus/RegisterGenerator Kr83mGenerator

/nexus/RegisterPersistencyManager PersistencyManager
/nexus/RegisterRunAction DefaultRunAction
/nexus/RegisterEventAction DefaultEventAction
/nexus/RegisterTrackingAction DefaultTrackingAction

/nexus/RegisterMacro {config}
"""

config_template = """
/run/verbose 0
/event/verbose 0
/tracking/verbose 0
/process/em/verbose 0

/Geometry/Next100/elfield false
/Geometry/Next100/max_step_size .05 mm
/Geometry/Next100/pressure {pressure} bar
/Geometry/Next100/gas {gas}
/Geometry/PmtR11410/time_binning 25. nanosecond
/Geometry/Next100/sipm_time_binning  1. microsecond
/Geometry/Next100/drift_v   1.480 mm/microsecond
/Geometry/Next100/EL_drift_v 6.249 mm/microsecond
/Geometry/Next100/drift_transv_diff 1.87 mm/sqrt(cm)
/Geometry/Next100/drift_long_diff 0.65 mm/sqrt(cm)
/Geometry/Next100/ELtransv_diff 0.46 mm/sqrt(cm)
/Geometry/Next100/ELlong_diff 0.30 mm/sqrt(cm)
/Geometry/Next100/e_lifetime 50   ms

/Generator/Kr83mGenerator/region CENTER

/Actions/DefaultEventAction/min_energy 1 keV
/Actions/DefaultEventAction/max_energy 1 MeV

/PhysicsList/Nexus/clustering          false
/PhysicsList/Nexus/drift               false
/PhysicsList/Nexus/electroluminescence false

/nexus/random_seed             {seed}
/nexus/persistency/output_file {output}
/nexus/persistency/start_id    {start}
/nexus/persistency/event_type background
"""

seed  = 12343210
start = 0
for gas in "naturalXe GAr".split():
    for i in range(20):
        pressure = round(10 + i*2) / 10

        config = f"kr_{gas}_{pressure}bar.config"
        init   = config.replace(".config", ".init")
        output = config.replace(".config", ".h5")

        open(  init, "w").write(  init_template.format(**globals()))
        open(config, "w").write(config_template.format(**globals()))

        seed  += 1
        start += 10000
