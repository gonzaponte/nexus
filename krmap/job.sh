source /gpfs0/arazi/projects/geant4.11.0.2-build/geant4make.sh
cd /gpfs0/arazi/users/gonzalom/sw/nexus

date

time ./build/nexus -b -n {n_events} {ini}

date

mv {out_nexus}.h5 {out_nexus}

. /gpfs0/arazi/projects/miniconda_24.7.1-0/etc/profile.d/conda.sh
conda activate IC-3.8-2022-04-13

export ICTDIR=/gpfs0/arazi/projects/IC-v2/
export ICDIR=$ICTDIR/invisible_cities/
export PYTHONPATH=$ICTDIR
export PATH=$ICTDIR/bin:$PATH
export HDF_USE_FILE_LOCKING=FALSE

export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=1

logcity() {{
    echo "$@" 1>&2; printf "\n\n===$@===\n";
}}

logcity "DETSIM"   ; time city detsim    {cnf_detsim}
logcity "DIOMIRA"  ; time city diomira   {cnf_diomira}
logcity "IRENE"    ; time city irene     {cnf_irene}
logcity "DOROTHEA" ; time city dorothea  {cnf_dorothea}
logcity "SOPHRONIA"; time city sophronia {cnf_sophronia}
logcity "EUTROPIA" ; time city eutropia  {cnf_eutropia}

date
