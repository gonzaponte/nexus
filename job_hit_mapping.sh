
source /gpfs0/arazi/projects/geant4.11.0.2-build/geant4make.sh
cd /gpfs0/arazi/users/gonzalom/sw/nexus

date
time ./build/nexus -b -n 100000000 macros/SquareOpticalFiber.init.mac
date
