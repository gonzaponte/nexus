
. /gpfs0/arazi/projects/miniconda/etc/profile.d/conda.sh
conda activate IC-3.7-2020-06-16

date

cd /gpfs0/arazi/users/gonzalom/sw/nexus/scan
python3 deconvolve.py {filename}

date
