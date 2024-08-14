source /gpfs0/arazi/projects/geant4.11.0.2-build/geant4make.sh
cd /gpfs0/arazi/users/gonzalom/sw/nexus

date

time ./build/nexus -b -n {n_events} {ini}

date

. /gpfs0/arazi/projects/miniconda/etc/profile.d/conda.sh
conda activate IC-3.7-2020-06-16

python3 ./scan/process_tpb_hits.py {txt} {pitch}

date
