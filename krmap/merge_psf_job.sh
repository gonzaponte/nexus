
. /gpfs0/arazi/projects/miniconda_24.7.1-0/etc/profile.d/conda.sh
conda activate IC-3.8-2022-04-13

export ICTDIR=/gpfs0/arazi/projects/IC-v2/
export ICDIR=$ICTDIR/invisible_cities/
export PYTHONPATH=$ICTDIR
export PATH=$ICTDIR/bin:$PATH
export HDF_USE_FILE_LOCKING=FALSE

krmap=/gpfs0/arazi/users/gonzalom/sw/nexus/krmap/
input=$krmap/output/data/eutropia/
output=$krmap/output/data/new_psf.h5
python3.8 $krmap/merge_psfs.py $input $output
